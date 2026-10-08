"""Compare a teacher, task-only control, and distilled student checkpoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from src.baselines import build_baseline
from src.data.dataloaders import move_batch_to_device
from src.data.dataset import SplitManifest, load_split_manifest
from src.distillation.comparison import PairedErrorAnalysis
from src.models.loaders import load_model
from src.utils.artifacts import sha256_file, write_json
from src.utils.utils import BaselineConfig, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure where KD transfers (or loses) teacher decisions"
    )
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--kd-config", type=Path, required=True)
    parser.add_argument("--kd-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"), default=("validation", "test"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--examples", type=int, default=20)
    return parser.parse_args()


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no state_dict mapping: {path}")
    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.") and isinstance(value, torch.Tensor)
    }
    if not model_state:
        raise ValueError(f"checkpoint has no model.* tensors: {path}")
    return model_state


def _load_model(config: BaselineConfig, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = load_model(
        config.model.name,
        model_id=config.model.model_id,
        pretrained=config.model.pretrained,
        class_names=config.data.class_names,
        ignore_index=config.loss.ignore_index,
    )
    model.load_state_dict(_checkpoint_state(checkpoint), strict=True)
    return model.to(device).eval()


def _validate_configs(configs: dict[str, BaselineConfig]) -> SplitManifest:
    reference_name, reference = next(iter(configs.items()))
    reference_manifest = load_split_manifest(reference.data.split_manifest)
    for name, config in configs.items():
        manifest = load_split_manifest(config.data.split_manifest)
        if config.data.class_names != reference.data.class_names:
            raise ValueError(f"{name} class names do not match {reference_name}")
        if config.data.mask_variant != reference.data.mask_variant:
            raise ValueError(f"{name} mask variant does not match {reference_name}")
        if (config.data.image_height, config.data.image_width) != (
            reference.data.image_height,
            reference.data.image_width,
        ):
            raise ValueError(f"{name} image size does not match {reference_name}")
        if config.loss.ignore_index != reference.loss.ignore_index:
            raise ValueError(f"{name} ignore index does not match {reference_name}")
        if manifest.sha256 != reference_manifest.sha256:
            raise ValueError(f"{name} split manifest does not match {reference_name}")
    return reference_manifest


def _per_image_miou(prediction: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int) -> float:
    valid = targets != ignore_index
    values: list[torch.Tensor] = []
    for class_id in range(num_classes):
        predicted = prediction.eq(class_id) & valid
        target = targets.eq(class_id) & valid
        union = (predicted | target).sum()
        if union.item() > 0:
            values.append((predicted & target).sum().float() / union.float())
    return float(torch.stack(values).mean()) if values else 0.0


def _image_record(
    frame_id: str,
    targets: torch.Tensor,
    teacher: torch.Tensor,
    control: torch.Tensor,
    kd: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
) -> dict[str, Any]:
    valid = targets != ignore_index
    pixels = int(valid.sum())
    control_correct = (control.eq(targets) & valid).sum()
    kd_correct = (kd.eq(targets) & valid).sum()
    return {
        "frame_id": frame_id,
        "teacher_miou": _per_image_miou(teacher, targets, num_classes, ignore_index),
        "control_miou": _per_image_miou(control, targets, num_classes, ignore_index),
        "kd_miou": _per_image_miou(kd, targets, num_classes, ignore_index),
        "kd_minus_control_miou": _per_image_miou(kd, targets, num_classes, ignore_index)
        - _per_image_miou(control, targets, num_classes, ignore_index),
        "teacher_control_agreement": float(
            (teacher.eq(control) & valid).sum() / max(pixels, 1)
        ),
        "kd_minus_control_pixel_accuracy": float(
            (kd_correct - control_correct).float() / max(pixels, 1)
        ),
    }


def _rank_examples(records: list[dict[str, Any]], count: int) -> dict[str, list[dict[str, Any]]]:
    by_delta = sorted(records, key=lambda record: record["kd_minus_control_miou"])
    return {"most_degraded_by_kd": by_delta[:count], "most_improved_by_kd": by_delta[-count:][::-1]}


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: path.expanduser().resolve()
        for name, path in {
            "teacher_config": args.teacher_config,
            "teacher_checkpoint": args.teacher_checkpoint,
            "control_config": args.control_config,
            "control_checkpoint": args.control_checkpoint,
            "kd_config": args.kd_config,
            "kd_checkpoint": args.kd_checkpoint,
        }.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required analysis files are missing: {missing}")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"analysis output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "teacher": load_config(paths["teacher_config"]),
        "control": load_config(paths["control_config"]),
        "kd": load_config(paths["kd_config"]),
    }
    manifest = _validate_configs(configs)
    device = _device(args.device)
    if args.precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 analysis requires a CUDA device")

    # The control's deterministic evaluation loader is the single source of images,
    # labels, and manifest order for all three predictions.
    components = build_baseline(configs["control"], include_distillation=False)
    models = {
        "teacher": _load_model(configs["teacher"], paths["teacher_checkpoint"], device),
        "control": components.model.to(device).eval(),
        "kd": _load_model(configs["kd"], paths["kd_checkpoint"], device),
    }
    models["control"].load_state_dict(_checkpoint_state(paths["control_checkpoint"]), strict=True)

    result: dict[str, Any] = {
        "schema_version": 1,
        "data": {
            "split_sha256": manifest.sha256,
            "class_names": list(configs["control"].data.class_names),
            "image_size": [configs["control"].data.image_height, configs["control"].data.image_width],
            "mask_variant": configs["control"].data.mask_variant,
        },
        "models": {
            name: {
                "config": str(paths[f"{name}_config"]),
                "config_sha256": sha256_file(paths[f"{name}_config"]),
                "checkpoint": str(paths[f"{name}_checkpoint"]),
                "checkpoint_sha256": sha256_file(paths[f"{name}_checkpoint"]),
            }
            for name in ("teacher", "control", "kd")
        },
        "splits": {},
    }
    loaders = {
        "validation": (components.validation_loader, components.validation_dataset.frame_ids),
        "test": (components.test_loader, components.test_dataset.frame_ids),
    }
    for split in args.splits:
        loader, frame_ids = loaders[split]
        analysis = PairedErrorAnalysis(
            configs["control"].data.class_names,
            ignore_index=configs["control"].loss.ignore_index,
            trimap_kernel_size=configs["control"].metrics.trimap_kernel_size,
        )
        records: list[dict[str, Any]] = []
        offset = 0
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            precision_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.precision == "bf16"
                else nullcontext()
            )
            with torch.inference_mode(), precision_context:
                predictions = {
                    name: model(batch["pixel_values"]).argmax(dim=1)
                    for name, model in models.items()
                }
            targets = batch["labels"]
            analysis.update(targets, **predictions)
            for index in range(targets.shape[0]):
                records.append(
                    _image_record(
                        frame_ids[offset + index],
                        targets[index],
                        predictions["teacher"][index],
                        predictions["control"][index],
                        predictions["kd"][index],
                        num_classes=len(configs["control"].data.class_names),
                        ignore_index=configs["control"].loss.ignore_index,
                    )
                )
            offset += targets.shape[0]
        if offset != len(frame_ids):
            raise RuntimeError(f"{split} loader yielded {offset} samples, expected {len(frame_ids)}")
        result["splits"][split] = {**analysis.serializable(), "examples": _rank_examples(records, args.examples)}

    write_json(output_dir / "paired_error_analysis.json", result)
    return result


def main() -> None:
    result = run_analysis(parse_args())
    for split, values in result["splits"].items():
        overall = values["all"]["overall"]
        print(
            f"{split}: KD-control accuracy delta={overall['kd_net_accuracy_delta']:+.4%}; "
            f"KD recovery of teacher advantage={overall['kd_recovery_rate']:.2%}"
        )


if __name__ == "__main__":
    main()
