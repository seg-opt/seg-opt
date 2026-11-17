import argparse
import logging
import random
from contextlib import contextmanager
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch import Callback
from torch.nn import functional as F

from scripts.evaluate import PRECISION, prepare_artifact_directory
from src.baselines import build_baseline
from src.distillation.common import valid_mask_at_size
from src.distillation.sard import SARDLoss, _standardized, capture_features, structure_weights
from src.train.trainer import DistillationModule, create_trainer
from src.utils.artifacts import sha256_file, write_json
from src.utils.logging import configure_logging
from src.utils.utils import load_config


REGION_NAMES = (
    "small_boundary",
    "large_boundary",
    "sky_boundary",
    "other_boundary",
    "background_interior",
    "sky_interior",
    "small_interior",
    "large_interior",
)


def validation_indices(length: int, count: int = 32) -> list[int]:
    if not 1 < count <= length:
        raise ValueError("diagnostic sample count must be between 2 and validation size")
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


def region_occupancies(
    labels: torch.Tensor,
    size: tuple[int, int],
    kernel_size: int,
    ignore_index: int,
) -> torch.Tensor:
    valid = (labels != ignore_index).unsqueeze(1).float()
    safe_labels = labels.masked_fill(labels == ignore_index, 0)
    one_hot = F.one_hot(safe_labels, num_classes=4).movedim(-1, 1).float() * valid
    padding = (kernel_size // 2,) * 4
    dilated = F.max_pool2d(
        F.pad(one_hot, padding, mode="replicate"), kernel_size, stride=1
    )
    eroded = 1 - F.max_pool2d(
        F.pad(1 - one_hot, padding, mode="replicate"), kernel_size, stride=1
    )
    valid_region = 1 - F.max_pool2d(
        F.pad(1 - valid, padding, mode="replicate"), kernel_size, stride=1
    )
    edges = ((dilated - eroded).clamp(0, 1) * valid_region).bool()
    all_edges = edges.any(dim=1)
    small = edges[:, 2]
    large = edges[:, 3] & ~small
    sky = edges[:, 1] & ~small & ~large
    other = all_edges & ~small & ~large & ~sky
    regions = [small, large, sky, other]
    regions.extend((labels == c) & ~all_edges for c in range(4))
    return F.adaptive_avg_pool2d(torch.stack(regions, dim=1).float(), size)


def regional_feature_loss(
    loss: SARDLoss,
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    labels: torch.Tensor,
    kernel_size: int = 7,
) -> list[dict]:
    size = tuple((length + loss.spatial_stride - 1) // loss.spatial_stride
                 for length in labels.shape[-2:])
    with torch.autocast(device_type=labels.device.type, enabled=False):
        student = loss._grid(student_features, size)
        teacher = loss._grid(teacher_features.detach(), size)
        projected_student = _standardized(loss.student_projection(student))
        projected_teacher = _standardized(loss.teacher_projection(teacher))
        error = (projected_student - projected_teacher).square().mean(dim=1)
        valid = valid_mask_at_size(labels, size, ignore_index=loss.ignore_index)[:, 0]
        regions = region_occupancies(labels, size, kernel_size, loss.ignore_index)
        regions = regions * valid[:, None]
        weightings = {
            "uniform": structure_weights(
                teacher, valid, edge_weight=0, corner_weight=0, density_weight=0
            ),
            "full": structure_weights(
                teacher, valid, edge_weight=2, corner_weight=1, density_weight=1,
                density_radius=loss.density_radius,
            ),
        }
        reports = []
        for image in range(labels.shape[0]):
            report = {}
            count = valid[image].sum().clamp_min(1)
            for name, weights in weightings.items():
                weighted_error = weights[image] * error[image]
                region_values = {}
                for index, region_name in enumerate(REGION_NAMES):
                    occupancy = regions[image, index]
                    cells = occupancy.sum()
                    region_values[region_name] = {
                        "area": float(cells / count),
                        "weight_mass": float((weights[image] * occupancy).sum()),
                        "mean_feature_error": (
                            float((error[image] * occupancy).sum() / cells)
                            if cells > 0 else None
                        ),
                        "feature_loss_contribution": float(
                            (weighted_error * occupancy).sum()
                        ),
                    }
                report[name] = {
                    "raw_feature": float(weighted_error.sum()),
                    "regions": region_values,
                }
            reports.append(report)
    return reports


@contextmanager
def diagnostic_state(module: DistillationModule):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    modes = [(child, child.training) for child in module.modules()]
    device = next(module.model.parameters()).device
    cuda_devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        try:
            module.eval()
            yield
        finally:
            for child, training in modes:
                child.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def measure_snapshot(module, dataset, indices: list[int]) -> dict:
    device = next(module.model.parameters()).device
    rows = []
    with diagnostic_state(module), torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=False):
            for index in indices:
                batch = dataset[index]
                images = batch["pixel_values"].unsqueeze(0).to(device)
                labels = batch["labels"].unsqueeze(0).to(device)
                with capture_features(module.model) as student:
                    module.model(images)
                with capture_features(module.teacher) as teacher:
                    module.teacher(images)
                reports = regional_feature_loss(
                    module.distillation_loss, student.value, teacher.value, labels
                )
                rows.append({"frame_id": dataset.frame_ids[index],
                             "weightings": reports[0]})
    summary = {}
    for weighting in ("uniform", "full"):
        total = sum(row["weightings"][weighting]["raw_feature"] for row in rows) / len(rows)
        regions = {}
        for name in REGION_NAMES:
            values = [row["weightings"][weighting]["regions"][name] for row in rows]
            errors = [v["mean_feature_error"] for v in values
                      if v["mean_feature_error"] is not None]
            regions[name] = {
                key: sum(v[key] for v in values) / len(values)
                for key in ("area", "weight_mass", "feature_loss_contribution")
            }
            regions[name]["mean_feature_error_when_present"] = (
                sum(errors) / len(errors) if errors else None
            )
            regions[name]["feature_loss_share"] = (
                regions[name]["feature_loss_contribution"] / total if total else None
            )
        summary[weighting] = {"raw_feature": total, "regions": regions}
    return {"per_image": rows, "mean_per_image": summary}


class SARDLossDiagnostics(Callback):
    def __init__(self, dataset, indices, output_dir, snapshot_epochs=(0, 1, 5, 10)):
        self.dataset = dataset
        self.indices = indices
        self.output_dir = Path(output_dir)
        self.snapshot_epochs = snapshot_epochs

    def record(self, trainer, module, completed_epochs: int) -> None:
        directory = self.output_dir / f"snapshot_{completed_epochs:02d}"
        directory.mkdir()
        state = {
            "schema_version": 1,
            "kind": "sard_diagnostic_not_training_resume",
            "completed_epochs": completed_epochs,
            "global_step": trainer.global_step,
            "student": {k: v.detach().cpu().clone()
                        for k, v in module.model.state_dict().items()},
            "distillation_loss": {k: v.detach().cpu().clone()
                                  for k, v in module.distillation_loss.state_dict().items()},
        }
        with (directory / "state.pt").open("xb") as handle:
            torch.save(state, handle)
        report = measure_snapshot(module, self.dataset, self.indices)
        report.update({
            "schema_version": 1,
            "completed_epochs": completed_epochs,
            "global_step": trainer.global_step,
            "learning_rates": [group["lr"] for group in trainer.optimizers[0].param_groups],
            "diagnostic_precision": "fp32",
            "region_priority": ["small_boundary", "large_boundary", "sky_boundary"],
            "boundary_kernel_size": 7,
            "feature_weight": module.distillation_loss.feature_weight,
            "snapshot_sha256": sha256_file(directory / "state.pt"),
        })
        write_json(directory / "regions.json", report)
        logging.getLogger("seg_opt").info(
            "SARD snapshot epoch=%d step=%d uniform FC=%.6f full FC=%.6f",
            completed_epochs, trainer.global_step,
            report["mean_per_image"]["uniform"]["raw_feature"],
            report["mean_per_image"]["full"]["raw_feature"],
        )

    def on_train_start(self, trainer, pl_module) -> None:
        self.record(trainer, pl_module, 0)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        completed_epochs = trainer.current_epoch + 1
        if completed_epochs in self.snapshot_epochs:
            self.record(trainer, pl_module, completed_epochs)
        if completed_epochs >= max(self.snapshot_epochs):
            trainer.should_stop = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure regional SARD feature losses")
    parser.add_argument("--variant", choices=("uniform", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = Path("experiments/strong_teacher_distillation") / (
        f"fast_scnn_sard_{args.variant}_b4_80e.yaml"
    )
    config = load_config(config_path)
    if (config.training.epochs != 80 or config.training.devices != 1
            or config.training.seed != 42 or config.distillation.method != "sard"
            or config.distillation.spatial_stride != 8):
        raise ValueError("diagnostic requires the seed-42, stride-8, single-GPU 80e recipe")
    config = prepare_artifact_directory(config, args.output_dir)
    output_dir = Path(config.training.output_dir)
    log = configure_logging(output_dir)
    L.seed_everything(config.training.seed, workers=True)
    components = build_baseline(config)
    indices = validation_indices(len(components.validation_dataset))
    script_root = Path(__file__).resolve().parents[1]
    write_json(output_dir / "diagnostics.json", {
        "schema_version": 1,
        "variant": args.variant,
        "stop_after_epochs": 10,
        "schedule_epochs": 80,
        "snapshot_epochs": [0, 1, 5, 10],
        "validation_indices": indices,
        "frame_ids": [components.validation_dataset.frame_ids[i] for i in indices],
        "split_sha256": components.split_manifest.sha256,
        "config_sha256": sha256_file(output_dir / "config.yaml"),
        "teacher_checkpoint_sha256": sha256_file(config.distillation.teacher.checkpoint_path),
        "teacher_config_sha256": sha256_file(output_dir / "teacher_config.yaml"),
        "source_sha256": {
            path: sha256_file(script_root / path) for path in (
                "scripts/debug_sard_loss.py", "src/distillation/sard.py",
                "src/distillation/common.py", "src/train/trainer.py", "src/baselines.py",
            )
        },
        "region_priority": ["small_boundary", "large_boundary", "sky_boundary"],
        "diagnostic_precision": "fp32",
    })
    trainer = create_trainer(
        output_dir=output_dir,
        max_epochs=config.training.epochs,
        precision=PRECISION[config.training.precision],
        devices=1,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        log_every_n_steps=config.training.logging_steps,
        enable_wandb=False,
    )
    trainer.callbacks.append(SARDLossDiagnostics(
        components.validation_dataset, indices, output_dir,
    ))
    log.info("Starting %s diagnostic: 10 epochs of the 80-epoch schedule", args.variant)
    trainer.fit(
        components.module,
        train_dataloaders=components.train_loader,
        val_dataloaders=components.validation_loader,
    )
    log.info("Diagnostic complete; no test evaluation; artifacts=%s", output_dir)


if __name__ == "__main__":
    main()
