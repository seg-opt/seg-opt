import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
from lightning.pytorch.loggers import WandbLogger
from dataclasses import replace
from pathlib import Path

import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.evaluate import (
    _distillation_artifact,
    evaluate_checkpoint,
    prepare_artifact_directory,
)
from src.baselines import BaselineComponents
from src.data.dataset import SplitManifest
from src.train.losses import SegmentationLoss
from src.train.trainer import BaselineModule, LusnarExamplesCallback, create_trainer
from src.utils.utils import TeacherConfig, load_config, write_resolved_config


CLASS_NAMES = ("background", "sky", "small_rock", "large_rock")


class TinyBaseline(nn.Module):
    def __init__(self, num_classes=4) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, images):
        return self.projection(images)

    def loss_and_scores(self, images, labels, criterion):
        scores = self(images)
        return criterion(scores, labels), scores

    def encoder_parameters(self):
        return iter(())


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return {
            "pixel_values": torch.randn(3, 8, 12, generator=generator),
            "labels": torch.randint(0, 4, (8, 12), generator=generator),
        }


@pytest.mark.parametrize("lusnar", [False, True])
def test_checkpoint_evaluation_writes_complete_local_artifact(tmp_path, lusnar):
    config = load_config(
        "experiments/lusnar/segformer_b0.yaml" if lusnar
        else "experiments/baselines_benchmark/fast_scnn.yaml"
    )
    config = replace(
        config,
        training=replace(config.training, output_dir=str(tmp_path)),
        metrics=replace(config.metrics, trimap_kernel_size=None),
    )
    if lusnar:
        _write_examples(tmp_path / "examples")
        config = replace(config, data=replace(
            config.data, dataset_root=str(tmp_path), image_height=32, image_width=32,
        ))
    write_resolved_config(config, tmp_path / "config.yaml")
    model = TinyBaseline(config.num_classes)
    module = BaselineModule(
        model,
        SegmentationLoss(config.num_classes),
        head_learning_rate=5e-4,
        class_names=config.data.class_names,
        total_steps=1,
    )
    dataset = TinyDataset()
    loader = DataLoader(dataset, batch_size=2)
    manifest = SplitManifest(
        path=Path("split.json"),
        sha256="manifest-sha256",
        seed=42,
        excluded_id_files=(),
        included_id_files=(),
        train_ids=("0001", "0002"),
        validation_ids=("0003", "0004"),
        test_ids=("0005", "0006"),
    )
    components = BaselineComponents(
        model=model,
        module=module,
        train_dataset=dataset,
        validation_dataset=dataset,
        test_dataset=dataset,
        train_loader=loader,
        validation_loader=loader,
        test_loader=loader,
        split_manifest=manifest,
    )
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

    trainer.test(module, dataloaders=loader, ckpt_path=str(checkpoint), verbose=False)
    reference_metrics = module.latest_metrics("test")
    callbacks_before = list(trainer.callbacks)
    artifact = evaluate_checkpoint(
        config,
        components,
        checkpoint,
        tmp_path,
        devices=1,
        run_kind="smoke",
        trainer=trainer,
    )

    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved == artifact
    assert artifact["schema_version"] == 2
    assert artifact["run_kind"] == "smoke"
    assert artifact["selection"]["metric"] == "val/miou"
    assert trainer.callbacks == callbacks_before
    assert module.latest_metrics("test") == reference_metrics
    assert artifact["data"]["mask_variant"] == config.data.mask_variant
    assert artifact["data"]["split_sha256"] == "manifest-sha256"
    assert set(artifact["test"]["per_class"]) == set(config.data.class_names)
    assert len(artifact["test"]["confusion_matrix"]) == config.num_classes
    if lusnar:
        assert len(list((tmp_path / "segmentation_samples").glob("*.png"))) == 6
    else:
        assert not (tmp_path / "segmentation_samples").exists()
    assert artifact["checkpoint_sha256"]


def test_artifact_preparation_refuses_to_mix_runs(tmp_path):
    config = load_config("experiments/baselines_benchmark/fast_scnn.yaml")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "old-metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must be empty"):
        prepare_artifact_directory(config, output_dir)


def test_distillation_artifact_copies_teacher_config_and_hashes_provenance(tmp_path):
    config = load_config("experiments/bpkd/fast_scnn_bpkd_w7_clean.yaml")
    checkpoint = tmp_path / "teacher.ckpt"
    checkpoint.write_bytes(b"trained-teacher")
    assert config.distillation is not None
    config = replace(
        config,
        distillation=replace(
            config.distillation,
            teacher=TeacherConfig(
                config_path=config.distillation.teacher.config_path,
                checkpoint_path=str(checkpoint),
            ),
        ),
    )

    prepared = prepare_artifact_directory(config, tmp_path / "artifact")
    metadata = _distillation_artifact(prepared)

    assert (tmp_path / "artifact" / "teacher_config.yaml").read_text(
        encoding="utf-8"
    ) == Path(config.distillation.teacher.config_path).read_text(encoding="utf-8")
    assert metadata is not None
    assert metadata["teacher"]["checkpoint_sha256"]
    assert metadata["teacher"]["config_sha256"]


def _write_examples(directory):
    samples = []
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[:, 16:] = 1
    for index in range(6):
        timestamp = str(1000 + index)
        sample_dir = directory / timestamp
        sample_dir.mkdir(parents=True)
        Image.fromarray(np.full((32, 32, 3), 100 + index, dtype=np.uint8)).save(sample_dir / "rgb.png")
        mask = Image.fromarray(labels).convert("P")
        mask.putpalette([255, 0, 0, 0, 0, 255] + [0] * (256 * 3 - 6))
        mask.save(sample_dir / "label.png")
        samples.append({
            "scene": "Moon_2", "timestamp": timestamp,
            "files": [{"path": f"{timestamp}/rgb.png"}, {"path": f"{timestamp}/label.png"}],
        })
    (directory / "manifest.json").write_text(json.dumps({"sample_count": 6, "samples": samples}))


def test_examples_use_indexed_labels_and_upload_one_wandb_artifact(tmp_path, monkeypatch):
    _write_examples(tmp_path / "examples")
    config = load_config("experiments/lusnar/segformer_b0.yaml")
    model = TinyBaseline(5)
    with torch.no_grad():
        model.projection.weight.zero_()
        model.projection.bias.zero_()
        model.projection.bias[0] = 1
    batch_norm = nn.BatchNorm2d(5)
    model.projection = nn.Sequential(model.projection, batch_norm)
    module = BaselineModule(
        model, SegmentationLoss(5), head_learning_rate=5e-4,
        class_names=config.data.class_names,
    )
    module.eval()
    logger = MagicMock(spec=WandbLogger)
    logger.experiment.id = "example-run"
    artifact = MagicMock()
    artifact_type = MagicMock(return_value=artifact)
    monkeypatch.setattr("wandb.Artifact", artifact_type)
    context = MagicMock(side_effect=nullcontext)
    trainer = SimpleNamespace(
        is_global_zero=True, loggers=[logger],
        precision_plugin=SimpleNamespace(forward_context=context),
    )
    callback = LusnarExamplesCallback(
        tmp_path / "examples", tmp_path / "output", config.data.class_names, 32, 32,
    )
    forward_states = []
    handle = model.register_forward_pre_hook(
        lambda model, _inputs: forward_states.append((model.training, torch.is_inference_mode_enabled()))
    )
    running_mean = batch_norm.running_mean.clone()
    callback.on_test_start(trainer, module)
    handle.remove()
    assert forward_states == [(False, True)] * 6
    torch.testing.assert_close(batch_norm.running_mean, running_mean)
    directory = tmp_path / "output" / "segmentation_samples"
    assert len(list(directory.glob("*.png"))) == 6
    assert (directory / "manifest.json").read_bytes() == (tmp_path / "examples" / "manifest.json").read_bytes()
    with Image.open(directory / "Moon_2_1000.png") as image:
        pixels = np.asarray(image)
        np.testing.assert_array_equal(pixels[24:56, :32], np.full((32, 32, 3), 100))
        np.testing.assert_array_equal(pixels[24:56, 32:48], pixels[24:56, 64:80])
        assert np.any(pixels[24:56, 48:64] != pixels[24:56, 80:96])
    context.assert_called_once()
    artifact_type.assert_called_once_with(
        name="lusnar-examples-example-run", type="evaluation",
        metadata={"source": "lusnar/examples", "sample_count": 6},
    )
    artifact.add_dir.assert_called_once_with(str(directory))
    logger.experiment.log_artifact.assert_called_once_with(artifact)
    trainer.is_global_zero = False
    callback.on_test_start(trainer, module)
    logger.experiment.log_artifact.assert_called_once()


def test_examples_callback_is_removed_if_evaluation_fails(tmp_path):
    config = load_config("experiments/lusnar/segformer_b0.yaml")
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.touch()
    module = MagicMock()
    module.latest_metrics.return_value = {}
    trainer = SimpleNamespace(
        validate=MagicMock(), test=MagicMock(side_effect=RuntimeError("test failed")),
        callbacks=[],
    )
    with pytest.raises(RuntimeError, match="test failed"):
        evaluate_checkpoint(
            config, SimpleNamespace(module=module, validation_loader=[], test_loader=[]),
            checkpoint, tmp_path, trainer=trainer,
        )
    assert trainer.callbacks == []
