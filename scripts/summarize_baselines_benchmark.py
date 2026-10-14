"""Summarize the clean teacher-candidate versus Fast-SCNN benchmark."""

import argparse
import json
from pathlib import Path
from typing import Any


STUDENT = "fast_scnn"
TEACHER_CANDIDATES = (
    "resnet34_unet",
    "segformer_b0",
    "segformer_b2",
    "segformer_b4",
    "dinov3_vitl16",
    "mask2former_swinl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare clean benchmark metrics and select a teacher by validation mIoU"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/baselines_benchmark"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional Markdown output path; otherwise print to standard output",
    )
    return parser.parse_args()


def _metrics_path(results_root: Path, name: str) -> Path:
    return results_root / name / "seed42" / "metrics.json"


def load_metrics(results_root: Path, name: str) -> dict[str, Any]:
    path = _metrics_path(results_root, name)
    if not path.is_file():
        raise FileNotFoundError(f"missing completed benchmark metrics: {path}")
    with path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    if metrics.get("run_kind") != "full":
        raise ValueError(f"benchmark must be a full run, not {metrics.get('run_kind')!r}: {path}")
    if metrics["model"]["name"] != name:
        raise ValueError(f"model name does not match benchmark path: {path}")
    if metrics["data"]["mask_variant"] != "clean":
        raise ValueError(f"benchmark does not use clean masks: {path}")
    return metrics


def _value(metrics: dict[str, Any], split: str, key: str) -> float:
    return float(metrics[split][key])


def _class_iou(metrics: dict[str, Any], split: str, name: str) -> float:
    return float(metrics[split]["per_class"][name]["iou"])


def _trimap_iou(metrics: dict[str, Any], split: str, name: str) -> float:
    return float(metrics[split]["trimap"]["per_class"][name]["iou"])


def render_summary(metrics_by_name: dict[str, dict[str, Any]]) -> str:
    required = {STUDENT, *TEACHER_CANDIDATES}
    missing = required - metrics_by_name.keys()
    if missing:
        raise ValueError(f"missing benchmark metrics for: {', '.join(sorted(missing))}")

    student = metrics_by_name[STUDENT]
    teacher_name = max(
        TEACHER_CANDIDATES,
        key=lambda name: _value(metrics_by_name[name], "validation", "miou"),
    )
    teacher = metrics_by_name[teacher_name]

    lines = [
        "# Clean baseline benchmark",
        "",
        "Teacher selection: highest validation mIoU (test metrics were not used for selection).",
        "",
        "| Model | Val mIoU | Val small-rock IoU | Val trimap small-rock IoU | Test mIoU | Test small-rock IoU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in (STUDENT, *TEACHER_CANDIDATES):
        metrics = metrics_by_name[name]
        lines.append(
            "| {name} | {val_miou:.4f} | {val_rock:.4f} | {val_trimap:.4f} | {test_miou:.4f} | {test_rock:.4f} |".format(
                name=name,
                val_miou=_value(metrics, "validation", "miou"),
                val_rock=_class_iou(metrics, "validation", "small_rock"),
                val_trimap=_trimap_iou(metrics, "validation", "small_rock"),
                test_miou=_value(metrics, "test", "miou"),
                test_rock=_class_iou(metrics, "test", "small_rock"),
            )
        )

    margin_miou = _value(teacher, "validation", "miou") - _value(student, "validation", "miou")
    margin_rock = _class_iou(teacher, "validation", "small_rock") - _class_iou(student, "validation", "small_rock")
    margin_trimap = _trimap_iou(teacher, "validation", "small_rock") - _trimap_iou(student, "validation", "small_rock")
    lines.extend(
        [
            "",
            f"Selected teacher: **{teacher_name}**.",
            f"Validation margin over Fast-SCNN: **{margin_miou:+.4f} mIoU**, **{margin_rock:+.4f} small-rock IoU**, and **{margin_trimap:+.4f} trimap small-rock IoU**.",
        ]
    )
    if margin_miou > 0 and margin_rock > 0:
        lines.append("There is a task-quality margin worth testing with distillation.")
    else:
        lines.append("Do not proceed to distillation yet: the selected teacher lacks a positive task-quality margin over Fast-SCNN.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    metrics_by_name = {
        name: load_metrics(args.results_root, name)
        for name in (STUDENT, *TEACHER_CANDIDATES)
    }
    summary = render_summary(metrics_by_name)
    if args.output is None:
        print(summary, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
