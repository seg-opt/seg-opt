import json
from dataclasses import replace
from pathlib import Path

import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.evaluate import evaluate_checkpoint, prepare_artifact_directory
from src.baselines import BaselineComponents
from src.data.dataset import SplitManifest
from src.train.losses import SegmentationLoss
from src.train.trainer import BaselineModule, create_trainer
from src.utils.utils import load_config, write_resolved_config


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
    config = load_config("experiments/baselines/fast_scnn.yaml")
    config = replace(
        config,
        training=replace(config.training, output_dir=str(tmp_path)),
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
    assert saved == artifact
    assert artifact["run_kind"] == "smoke"
    assert artifact["selection"]["metric"] == "val/miou"
    assert artifact["data"]["split_sha256"] == "manifest-sha256"
    assert set(artifact["test"]["per_class"]) == set(CLASS_NAMES)
    assert len(artifact["test"]["confusion_matrix"]) == 4
    assert artifact["checkpoint_sha256"]


def test_artifact_preparation_refuses_to_mix_runs(tmp_path):
    config = load_config("experiments/baselines/fast_scnn.yaml")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "old-metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must be empty"):
        prepare_artifact_directory(config, output_dir)
