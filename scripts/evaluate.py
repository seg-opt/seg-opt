import argparse
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import lightning as L
import torch

from src.baselines import BaselineComponents, build_baseline
from src.data.dataset import MASK_VARIANTS, MaskVariant
from src.utils.artifacts import (
    parameter_counts,
    sha256_file,
    write_json,
)
from src.utils.logging import configure_logging
from src.utils.utils import BaselineConfig, load_config, write_resolved_config


PRECISION = {"fp32": "32-true", "fp16": "16-mixed", "bf16": "bf16-mixed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a canonical lunar segmentation baseline checkpoint"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--mask-variant",
        choices=MASK_VARIANTS,
        help="override the config mask variant for this evaluation",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="override the config split manifest for evaluation",
    )
    parser.add_argument("--devices", default="auto")
    parser.add_argument(
        "--smoke-run",
        action="store_true",
        help="evaluate one validation and one test batch",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _structured_metrics(
    metrics: dict[str, Any],
    class_names: tuple[str, ...],
    *,
    trimap_kernel_size: int | None,
) -> dict[str, Any]:
    structured = {
        "miou": metrics["miou"],
        "macro_f1": metrics["macro_f1"],
        "pixel_accuracy": metrics["pixel_accuracy"],
        "per_class": {
            name: {
                "iou": metrics[f"iou_{name}"],
                "precision": metrics[f"precision_{name}"],
                "recall": metrics[f"recall_{name}"],
            }
            for name in class_names
        },
        "confusion_matrix": metrics["confusion_matrix"],
    }
    if trimap_kernel_size is not None:
        structured["trimap"] = {
            "kernel_size": trimap_kernel_size,
            "miou": metrics["trimap_miou"],
            "per_class": {
                name: {
                    "iou": metrics[f"trimap_iou_{name}"],
                    "precision": metrics[f"trimap_precision_{name}"],
                    "recall": metrics[f"trimap_recall_{name}"],
                }
                for name in class_names
            },
        }
    return structured


def _checkpoint_epoch(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch", -1)
    return int(epoch)


def prepare_artifact_directory(
    config: BaselineConfig,
    output_dir: Path,
) -> BaselineConfig:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"artifact path is not a directory: {output_dir}")
        if next(output_dir.iterdir(), None) is not None:
            raise FileExistsError(
                f"artifact directory must be empty for a new run: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True)

    split_path = output_dir / "split.json"
    shutil.copy2(config.data.split_manifest, split_path)
    if config.distillation is not None:
        teacher_config = Path(config.distillation.teacher.config_path)
        if not teacher_config.is_file():
            raise FileNotFoundError(f"teacher config not found: {teacher_config}")
        shutil.copy2(teacher_config, output_dir / "teacher_config.yaml")
    resolved = replace(
        config,
        data=replace(config.data, split_manifest=str(split_path)),
        training=replace(config.training, output_dir=str(output_dir)),
    )
    write_resolved_config(resolved, output_dir / "config.yaml")
    return resolved


def _distillation_artifact(config: BaselineConfig) -> dict[str, Any] | None:
    if config.distillation is None:
        return None
    settings = asdict(config.distillation)
    teacher = settings.pop("teacher")
    teacher_config = Path(teacher["config_path"])
    teacher_checkpoint = Path(teacher["checkpoint_path"])
    return {
        "method": config.distillation.method,
        "settings": settings,
        "teacher": {
            **teacher,
            "config_sha256": (
                sha256_file(teacher_config) if teacher_config.is_file() else None
            ),
            "checkpoint_sha256": (
                sha256_file(teacher_checkpoint)
                if teacher_checkpoint.is_file()
                else None
            ),
        },
    }


def evaluate_checkpoint(
    config: BaselineConfig,
    components: BaselineComponents,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    devices: int | list[int] | str = "auto",
    run_kind: str = "full",
    mask_variant: MaskVariant | None = None,
    trainer: L.Trainer | None = None,
) -> dict[str, Any]:
    """Restore one checkpoint, run validation/test, and write metrics.json."""
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if run_kind not in {"full", "smoke"}:
        raise ValueError("run_kind must be 'full' or 'smoke'")
    selected_mask_variant = (
        config.data.mask_variant if mask_variant is None else mask_variant
    )
    if selected_mask_variant not in MASK_VARIANTS:
        raise ValueError(f"mask_variant must be one of {list(MASK_VARIANTS)}")

    if trainer is None:
        limit = 1 if run_kind == "smoke" else 1.0
        trainer = L.Trainer(
            default_root_dir=output_dir,
            accelerator="auto",
            devices=devices,
            precision=PRECISION[config.training.precision],
            logger=False,
            enable_checkpointing=False,
            deterministic="warn",
            limit_val_batches=limit,
            limit_test_batches=limit,
            num_sanity_val_steps=0,
        )

    trainer.validate(
        components.module,
        dataloaders=components.validation_loader,
        ckpt_path=str(checkpoint_path),
        verbose=False,
    )
    validation = components.module.latest_metrics("val")
    trainer.test(
        components.module,
        dataloaders=components.test_loader,
        ckpt_path=str(checkpoint_path),
        verbose=False,
    )
    test = components.module.latest_metrics("test")

    config_path = output_dir / "config.yaml"
    artifact = {
        "schema_version": 2,
        "run_kind": run_kind,
        "model": {
            "name": config.model.name,
            "model_id": config.model.model_id,
            "pretrained": config.model.pretrained,
            **parameter_counts(components.model),
        },
        "data": {
            "class_names": list(config.data.class_names),
            "image_size": [config.data.image_height, config.data.image_width],
            "mask_variant": selected_mask_variant,
            "split_sha256": components.split_manifest.sha256,
            "counts": components.split_manifest.counts,
        },
        "selection": {
            "metric": "val/miou",
            "best_epoch": _checkpoint_epoch(checkpoint_path),
            "best_value": validation["miou"],
        },
        "validation": _structured_metrics(
            validation,
            config.data.class_names,
            trimap_kernel_size=config.metrics.trimap_kernel_size,
        ),
        "test": _structured_metrics(
            test,
            config.data.class_names,
            trimap_kernel_size=config.metrics.trimap_kernel_size,
        ),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
    }
    distillation = _distillation_artifact(config)
    if distillation is not None:
        artifact["distillation"] = distillation
    write_json(output_dir / "metrics.json", artifact)
    return artifact


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    config = load_config(args.config)
    if args.mask_variant is not None:
        config = replace(
            config,
            data=replace(config.data, mask_variant=args.mask_variant),
        )
    if args.split_manifest is not None:
        split_manifest = args.split_manifest.expanduser().resolve()
        if not split_manifest.is_file():
            raise FileNotFoundError(f"split manifest not found: {split_manifest}")
        config = replace(
            config,
            data=replace(config.data, split_manifest=str(split_manifest)),
        )
    default_evaluation_dir = (
        "evaluation" if config.data.mask_variant == "ground" else "evaluation_clean"
    )
    output_dir = args.output_dir or checkpoint_path.parent.parent / default_evaluation_dir
    config = prepare_artifact_directory(config, output_dir)
    log = configure_logging(output_dir, args.verbose)
    L.seed_everything(config.training.seed, workers=True)
    log.info(
        "Loading baseline %s with %s masks",
        config.model.name,
        config.data.mask_variant,
    )
    components = build_baseline(config, include_distillation=False)
    artifact = evaluate_checkpoint(
        config,
        components,
        checkpoint_path,
        output_dir,
        devices=args.devices,
        run_kind="smoke" if args.smoke_run else "full",
        mask_variant=config.data.mask_variant,
    )
    log.info(
        "Evaluation complete: val mIoU=%.4f, test mIoU=%.4f",
        artifact["validation"]["miou"],
        artifact["test"]["miou"],
    )


if __name__ == "__main__":
    main()
