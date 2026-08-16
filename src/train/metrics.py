from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


_STAGES = ("train", "val", "test")


class SegmentationMetrics(nn.Module):
    """Accumulate target-by-prediction confusion matrices for segmentation."""

    def __init__(self, class_names: Sequence[str], ignore_index: int = 255) -> None:
        super().__init__()
        if not class_names:
            raise ValueError("class_names must contain at least one class")
        if len(set(class_names)) != len(class_names):
            raise ValueError("class_names must be unique")

        self.class_names = tuple(class_names)
        self.ignore_index = ignore_index
        for stage in _STAGES:
            self.register_buffer(
                f"{stage}_confusion",
                torch.zeros(
                    len(self.class_names),
                    len(self.class_names),
                    dtype=torch.long,
                ),
                persistent=False,
            )

    def _confusion(self, stage: str) -> torch.Tensor:
        if stage not in _STAGES:
            raise ValueError(f"unknown metric stage: {stage!r}")
        return getattr(self, f"{stage}_confusion")

    @torch.no_grad()
    def update(
        self,
        stage: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if predictions.shape != targets.shape:
            raise ValueError(
                "predictions and targets must have the same shape; "
                f"got {tuple(predictions.shape)} and {tuple(targets.shape)}"
            )

        confusion = self._confusion(stage)
        num_classes = len(self.class_names)
        valid = targets != self.ignore_index
        if not torch.any(valid):
            return

        valid_targets = targets[valid].long()
        valid_predictions = predictions[valid].long()
        if torch.any((valid_targets < 0) | (valid_targets >= num_classes)):
            raise ValueError("targets contain a non-ignored class outside the class range")
        if torch.any((valid_predictions < 0) | (valid_predictions >= num_classes)):
            raise ValueError("predictions contain a class outside the class range")

        indices = valid_targets * num_classes + valid_predictions
        confusion += torch.bincount(
            indices,
            minlength=num_classes**2,
        ).reshape_as(confusion)

    @torch.no_grad()
    def reset(self, stage: str) -> None:
        self._confusion(stage).zero_()

    @torch.no_grad()
    def snapshot(
        self,
        stage: str,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Return scalar metrics and a synchronized confusion-matrix snapshot."""
        confusion = self._confusion(stage).clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(confusion)

        intersection = confusion.diag().float()
        targets = confusion.sum(dim=1).float()
        predictions = confusion.sum(dim=0).float()
        union = targets + predictions - intersection
        present = union > 0

        iou = intersection / union.clamp_min(1)
        precision = intersection / predictions.clamp_min(1)
        recall = intersection / targets.clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)

        metrics = {
            "miou": iou[present].mean() if present.any() else iou.new_zeros(()),
            "macro_f1": f1[present].mean() if present.any() else f1.new_zeros(()),
            "pixel_accuracy": intersection.sum() / confusion.sum().clamp_min(1),
        }
        for index, name in enumerate(self.class_names):
            metrics[f"iou_{name}"] = iou[index]
            metrics[f"precision_{name}"] = precision[index]
            metrics[f"recall_{name}"] = recall[index]
        return metrics, confusion

    def compute(self, stage: str) -> dict[str, torch.Tensor]:
        return self.snapshot(stage)[0]

    def serializable(self, stage: str) -> dict[str, Any]:
        """Return JSON-safe metrics, including the target-by-prediction matrix."""
        metrics, confusion = self.snapshot(stage)
        return {
            **{name: float(value.detach().cpu()) for name, value in metrics.items()},
            "confusion_matrix": confusion.detach().cpu().tolist(),
        }
