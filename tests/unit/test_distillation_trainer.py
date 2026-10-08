import torch
from torch import nn
from torch.utils.data import DataLoader

from src.distillation.bpkd import PixelWiseKDLoss
from src.train.losses import SegmentationLoss
from src.train.trainer import (
    BaselineModule,
    DistillationModule,
    create_trainer,
)


class TinySegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.projection = nn.Conv2d(3, num_classes, kernel_size=1)
        self.forward_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        return self.projection(inputs)

    def loss_and_scores(self, inputs, labels, criterion):
        scores = self(inputs)
        return criterion(scores, labels), scores

    def encoder_parameters(self):
        return iter(())


def make_batch():
    return {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "labels": torch.randint(0, 4, (2, 8, 8)),
    }


def test_canonical_distillation_trains_only_student_and_saves_student_weights():
    teacher = TinySegmentationModel(4)
    student = TinySegmentationModel(4)
    module = DistillationModule(
        student,
        SegmentationLoss(4),
        teacher=teacher,
        distillation_loss=PixelWiseKDLoss(num_classes=4, spatial_stride=1),
        head_learning_rate=1e-3,
        class_names=["background", "sky", "small_rock", "large_rock"],
        trimap_kernel_size=7,
        total_steps=4,
    )
    logged: dict[str, torch.Tensor] = {}
    module.log = lambda name, value, **kwargs: logged.setdefault(name, value)
    module.train()
    teacher.train()
    module.on_train_epoch_start()

    loss = module.training_step(make_batch(), batch_idx=0)
    loss.backward()
    module.on_train_epoch_end()

    assert student.projection.weight.grad is not None
    assert teacher.projection.weight.grad is None
    assert not teacher.training
    assert teacher.forward_calls == 1
    assert not any(key.startswith("teacher.") for key in module.state_dict())
    assert {
        "train/teacher_miou",
        "train/teacher_iou_small_rock",
        "train/teacher_trimap_iou_small_rock",
    } <= set(logged)
    assert torch.isfinite(logged["train/teacher_student_agreement"])
    assert torch.isfinite(logged["train/kd_to_task_ratio"])

    baseline = BaselineModule(
        TinySegmentationModel(4),
        SegmentationLoss(4),
        head_learning_rate=1e-3,
        class_names=["background", "sky", "small_rock", "large_rock"],
        total_steps=4,
    )
    baseline.load_state_dict(module.state_dict(), strict=True)


def test_canonical_distillation_validation_does_not_call_teacher():
    teacher = TinySegmentationModel(4)
    module = DistillationModule(
        TinySegmentationModel(4),
        SegmentationLoss(4),
        teacher=teacher,
        distillation_loss=PixelWiseKDLoss(num_classes=4, spatial_stride=1),
        head_learning_rate=1e-3,
        class_names=["background", "sky", "small_rock", "large_rock"],
        total_steps=4,
    )
    module.log = lambda *args, **kwargs: None

    loss = module.validation_step(make_batch(), batch_idx=0)

    assert loss.ndim == 0
    assert teacher.forward_calls == 0


def test_distillation_lightning_checkpoint_is_evaluable_without_teacher(tmp_path):
    module = DistillationModule(
        TinySegmentationModel(4),
        SegmentationLoss(4),
        teacher=TinySegmentationModel(4),
        distillation_loss=PixelWiseKDLoss(num_classes=4, spatial_stride=1),
        head_learning_rate=1e-3,
        class_names=["background", "sky", "small_rock", "large_rock"],
        total_steps=1,
    )
    loader = DataLoader([make_batch()], batch_size=None)
    trainer = create_trainer(
        tmp_path,
        max_epochs=1,
        precision="32-true",
        devices=1,
        fast_dev_run=True,
        enable_wandb=False,
    )

    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    checkpoint = tmp_path / "checkpoints" / "best.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    assert not any(key.startswith("teacher.") for key in state)

    baseline = BaselineModule(
        TinySegmentationModel(4),
        SegmentationLoss(4),
        head_learning_rate=1e-3,
        class_names=["background", "sky", "small_rock", "large_rock"],
        total_steps=1,
    )
    baseline.load_state_dict(state, strict=True)
