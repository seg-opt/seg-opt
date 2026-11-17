import json
import random
from types import SimpleNamespace
from unittest.mock import patch

import lightning as L
import numpy as np
from PIL import Image
from dataclasses import replace
from pathlib import Path

import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.evaluate import (
    _segmentation_sample_writer,
    _distillation_artifact,
    evaluate_checkpoint,
    prepare_artifact_directory,
)
from src.baselines import BaselineComponents
from src.data.dataset import SplitManifest
from src.train.losses import SegmentationLoss
from src.train.trainer import BaselineModule, create_trainer
from src.utils.utils import TeacherConfig, load_config, write_resolved_config


CLASS_NAMES = ("background", "sky", "small_rock", "large_rock")


class TinyBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 4, kernel_size=1)

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


def test_checkpoint_evaluation_writes_complete_local_artifact(tmp_path):
    config = load_config("experiments/baselines_benchmark/fast_scnn.yaml")
    config = replace(
        config,
        training=replace(config.training, output_dir=str(tmp_path)),
        metrics=replace(config.metrics, trimap_kernel_size=None),
    )
    write_resolved_config(config, tmp_path / "config.yaml")
    model = TinyBaseline()
    module = BaselineModule(
        model,
        SegmentationLoss(4),
        head_learning_rate=5e-4,
        class_names=CLASS_NAMES,
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
    assert len(list((tmp_path / "segmentation_samples").glob("*.png"))) == 2
    assert saved == artifact
    assert artifact["schema_version"] == 2
    assert artifact["run_kind"] == "smoke"
    assert artifact["selection"]["metric"] == "val/miou"
    assert artifact["data"]["mask_variant"] == "clean"
    assert artifact["data"]["split_sha256"] == "manifest-sha256"
    assert set(artifact["test"]["per_class"]) == set(CLASS_NAMES)
    assert len(artifact["test"]["confusion_matrix"]) == 4
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


@pytest.mark.parametrize("size", [2, 12])
def test_segmentation_samples_are_reproducible_and_match_predictions(tmp_path, size):
    config = load_config("experiments/baselines_benchmark/fast_scnn.yaml")
    config = replace(config, data=replace(config.data, class_names=CLASS_NAMES))

    class PreviewDataset(TinyDataset):
        def __len__(self):
            return size

        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["labels"] = torch.zeros(8, 12, dtype=torch.long)
            sample["labels"][0, 0] = 255
            return sample

    model = TinyBaseline()
    with torch.no_grad():
        model.projection.weight.zero_()
        model.projection.bias.zero_()
        model.projection.bias[0] = 1
    module = BaselineModule(
        model, SegmentationLoss(4), head_learning_rate=5e-4, class_names=CLASS_NAMES
    )
    module.train()
    dataset = PreviewDataset()
    loader = DataLoader(dataset, batch_size=2)
    components = SimpleNamespace(test_dataset=dataset, test_loader=loader, module=module)
    trainer = SimpleNamespace(
        is_global_zero=True,
        world_size=1,
    )
    test_trainer = L.Trainer(
        accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False,
    )
    for run in ("first", "second"):
        state = random.getstate()
        module._test_sample_writer = _segmentation_sample_writer(
            config, components, tmp_path / run, trainer, "full"
        )
        assert random.getstate() == state
        with patch.object(model, "forward", wraps=model.forward) as forward:
            test_trainer.test(module, dataloaders=loader, verbose=False)
        assert forward.call_count == len(loader)
        module._test_sample_writer = None
    first = tmp_path / "first" / "segmentation_samples"
    second = tmp_path / "second" / "segmentation_samples"
    paths = sorted(first.glob("*.png"))
    expected = random.Random(config.training.seed).sample(range(size), min(5, size))
    assert {path.name for path in paths} == {f"sample_{index:06d}.png" for index in expected}
    for path in paths:
        assert path.read_bytes() == (second / path.name).read_bytes()
        with Image.open(path) as image:
            pixels = np.asarray(image)
            assert image.format == "PNG"
            assert image.width >= 36
            np.testing.assert_array_equal(pixels[24:32, 13:24], pixels[24:32, 25:36])
            np.testing.assert_array_equal(pixels[24, 12], [128, 128, 128])
    with pytest.raises(FileExistsError):
        _segmentation_sample_writer(config, components, tmp_path / "first", trainer, "full")
