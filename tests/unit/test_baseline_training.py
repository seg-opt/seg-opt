import json

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.train.losses import SegmentationLoss
from src.train.metrics import SegmentationMetrics
from src.train.trainer import (
    BaselineModule,
    cosine_with_warmup_multiplier,
    create_trainer,
    parameter_groups,
)
from src.train.profiling import NvtxRangesCallback


CLASS_NAMES = ["background", "sky", "small_rock", "large_rock"]


class TinyBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 5, kernel_size=1)
        self.head = nn.Conv2d(5, 4, kernel_size=1)
        self.forward_calls = 0
        self.loss_calls = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.head(self.encoder(images))

    def loss_and_scores(self, images, labels, criterion):
        self.loss_calls += 1
        scores = self(images)
        return criterion(scores, labels), scores

    def encoder_parameters(self):
        return self.encoder.parameters()


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.randn(3, 8, 12),
            "labels": torch.randint(0, 4, (8, 12)),
        }


def make_batch() -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.randn(2, 3, 8, 12),
        "labels": torch.randint(0, 4, (2, 8, 12)),
    }


def make_module(model: nn.Module | None = None) -> BaselineModule:
    return BaselineModule(
        model=model or TinyBaseline(),
        criterion=SegmentationLoss(4),
        head_learning_rate=3e-4,
        backbone_learning_rate=3e-5,
        weight_decay=0.01,
        class_names=CLASS_NAMES,
        total_steps=100,
    )


def test_metrics_ignore_labels_and_export_confusion_matrix() -> None:
    metrics = SegmentationMetrics(CLASS_NAMES)
    predictions = torch.tensor([[[0, 2], [2, 1]]])
    targets = torch.tensor([[[0, 1], [2, 255]]])

    metrics.update("test", predictions, targets)
    result = metrics.serializable("test")

    assert result["miou"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(5 / 9)
    assert result["pixel_accuracy"] == pytest.approx(2 / 3)
    assert result["iou_small_rock"] == pytest.approx(0.5)
    assert result["precision_small_rock"] == pytest.approx(0.5)
    assert result["recall_small_rock"] == pytest.approx(1.0)
    assert result["confusion_matrix"] == [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ]
    json.dumps(result)


def test_metrics_export_gt_trimap_diagnostics_without_changing_standard_iou() -> None:
    metrics = SegmentationMetrics(CLASS_NAMES, trimap_kernel_size=3)
    targets = torch.zeros(1, 5, 5, dtype=torch.long)
    targets[:, 2, 2] = 2
    predictions = targets.clone()
    predictions[:, 2, 2] = 0

    metrics.update("val", predictions, targets)
    result = metrics.serializable("val")

    assert result["miou"] < 1
    assert result["trimap_miou"] < 1
    assert "trimap_iou_small_rock" in result
    assert result["trimap_iou_small_rock"] == pytest.approx(0)


def test_all_ignored_loss_is_finite_and_differentiable() -> None:
    scores = torch.randn(1, 4, 3, 5, requires_grad=True)
    labels = torch.full((1, 3, 5), 255)

    loss = SegmentationLoss(4)(scores, labels)
    loss.backward()

    assert loss.item() == 0
    assert scores.grad is not None


def test_training_and_evaluation_use_model_loss_contract() -> None:
    model = TinyBaseline()
    module = make_module(model)
    logged = set()
    module.log = lambda name, *args, **kwargs: logged.add(name)
    batch = make_batch()

    train_loss = module.training_step(batch, batch_idx=0)
    assert train_loss.ndim == 0
    assert model.loss_calls == 1
    assert model.forward_calls == 1

    validation_output = module.validation_step(batch, batch_idx=0)
    assert validation_output.ndim == 0
    assert model.loss_calls == 2
    assert model.forward_calls == 2

    test_output = module.test_step(batch, batch_idx=0)
    assert test_output.ndim == 0
    assert model.loss_calls == 3
    assert model.forward_calls == 3
    assert {"train/loss", "val/loss", "test/loss"} <= logged
    assert not any(name.endswith(("cross_entropy_loss", "dice_loss")) for name in logged)


def test_latest_metrics_are_json_safe_and_defensive_copy() -> None:
    module = make_module()
    module.log = lambda *args, **kwargs: None
    module.on_validation_epoch_start()
    module.validation_step(make_batch(), batch_idx=0)
    module.on_validation_epoch_end()

    first = module.latest_metrics("val")
    first["confusion_matrix"][0][0] = -1
    second = module.latest_metrics("val")

    assert second["confusion_matrix"][0][0] >= 0
    json.dumps(second)


def test_parameter_groups_are_disjoint_and_use_configured_rates() -> None:
    model = TinyBaseline()
    groups = parameter_groups(
        model,
        head_learning_rate=3e-4,
        backbone_learning_rate=3e-5,
    )

    assert [group["name"] for group in groups] == ["head", "backbone"]
    assert [group["lr"] for group in groups] == pytest.approx([3e-4, 3e-5])
    assert {id(parameter) for parameter in groups[0]["params"]} == {
        id(parameter) for parameter in model.head.parameters()
    }
    assert {id(parameter) for parameter in groups[1]["params"]} == {
        id(parameter) for parameter in model.encoder.parameters()
    }


def test_optimizer_uses_stepwise_cosine_schedule_after_five_percent_warmup() -> None:
    configured = make_module().configure_optimizers()
    optimizer = configured["optimizer"]
    scheduler_config = configured["lr_scheduler"]

    assert isinstance(optimizer, torch.optim.AdamW)
    assert scheduler_config["interval"] == "step"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4 / 5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(3e-5 / 5)
    assert cosine_with_warmup_multiplier(
        4, total_steps=100, warmup_steps=5
    ) == pytest.approx(1)
    assert cosine_with_warmup_multiplier(
        99, total_steps=100, warmup_steps=5
    ) == pytest.approx(0)


def test_create_trainer_selects_one_best_miou_checkpoint_and_accumulates(
    tmp_path,
) -> None:
    trainer = create_trainer(
        tmp_path,
        max_epochs=20,
        precision="32-true",
        devices=1,
        accumulate_grad_batches=8,
        enable_wandb=False,
    )
    checkpoint = next(
        callback
        for callback in trainer.callbacks
        if isinstance(callback, ModelCheckpoint)
    )

    assert checkpoint.monitor == "val/miou"
    assert checkpoint.mode == "max"
    assert checkpoint.save_top_k == 1
    assert checkpoint.filename == "best"
    assert trainer.accumulate_grad_batches == 8


def test_create_trainer_honors_explicit_max_steps(tmp_path) -> None:
    trainer = create_trainer(
        tmp_path,
        max_epochs=20,
        max_steps=25,
        precision="32-true",
        devices=1,
        enable_wandb=False,
    )

    assert trainer.max_steps == 25


def test_create_trainer_adds_nvtx_callback_only_when_requested(tmp_path) -> None:
    trainer = create_trainer(
        tmp_path,
        max_epochs=1,
        precision="32-true",
        devices=1,
        enable_nvtx_ranges=True,
        enable_wandb=False,
    )

    assert any(isinstance(callback, NvtxRangesCallback) for callback in trainer.callbacks)


def test_smoke_mode_runs_one_batch_and_preserves_best_checkpoint(tmp_path) -> None:
    module = make_module()
    loader = DataLoader(TinyDataset(), batch_size=2)
    trainer = create_trainer(
        tmp_path,
        max_epochs=20,
        precision="32-true",
        devices=1,
        fast_dev_run=True,
        enable_wandb=False,
    )

    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    assert not trainer.fast_dev_run
    assert trainer.max_epochs == 1
    assert trainer.num_training_batches == 1
    assert trainer.num_val_batches == [1]
    checkpoint_path = tmp_path / "checkpoints" / "best.ckpt"
    assert checkpoint_path.is_file()

    restored = BaselineModule.load_from_checkpoint(
        checkpoint_path,
        model=TinyBaseline(),
        criterion=SegmentationLoss(4),
    )
    assert restored(torch.randn(1, 3, 8, 12)).shape == (1, 4, 8, 12)
