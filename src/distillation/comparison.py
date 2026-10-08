"""Paired, ground-truth-aware diagnostics for segmentation checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from src.distillation.bpkd import trimap_mask


class PairedErrorAnalysis:
    """Accumulate teacher/control/KD prediction relationships by target class."""

    _REGIONS = ("all", "trimap")

    def __init__(
        self,
        class_names: Sequence[str],
        *,
        ignore_index: int = 255,
        trimap_kernel_size: int | None = None,
    ) -> None:
        if not class_names:
            raise ValueError("class_names must contain at least one class")
        self.class_names = tuple(class_names)
        self.ignore_index = ignore_index
        self.trimap_kernel_size = trimap_kernel_size
        self._counts = {
            region: torch.zeros(len(self.class_names), 10, dtype=torch.long)
            for region in self._REGIONS
        }

    def _update_region(
        self,
        region: str,
        targets: torch.Tensor,
        teacher: torch.Tensor,
        control: torch.Tensor,
        kd: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        region_valid = valid
        if region == "trimap":
            if self.trimap_kernel_size is None:
                return
            region_valid = region_valid & trimap_mask(
                targets,
                num_classes=len(self.class_names),
                edge_kernel_size=self.trimap_kernel_size,
                ignore_index=self.ignore_index,
            )
        if not torch.any(region_valid):
            return

        target = targets[region_valid].long()
        teacher_correct = teacher[region_valid].eq(target)
        control_correct = control[region_valid].eq(target)
        kd_correct = kd[region_valid].eq(target)
        values = torch.stack(
            (
                torch.ones_like(target),
                teacher[region_valid].eq(control[region_valid]),
                teacher[region_valid].eq(kd[region_valid]),
                control[region_valid].eq(kd[region_valid]),
                teacher_correct & ~control_correct,
                teacher_correct & ~kd_correct,
                teacher_correct & ~control_correct & kd_correct,
                teacher_correct & ~control_correct & ~kd_correct,
                ~control_correct & kd_correct,
                control_correct & ~kd_correct,
            ),
            dim=1,
        ).to(dtype=torch.long)
        counts = self._counts[region].to(values.device)
        counts.index_add_(0, target, values)
        self._counts[region] = counts.cpu()

    @torch.no_grad()
    def update(
        self,
        targets: torch.Tensor,
        teacher: torch.Tensor,
        control: torch.Tensor,
        kd: torch.Tensor,
    ) -> None:
        expected = targets.shape
        if targets.ndim != 3 or any(
            prediction.shape != expected for prediction in (teacher, control, kd)
        ):
            raise ValueError("targets and predictions must share [B, H, W] shape")
        valid = targets != self.ignore_index
        if torch.any(
            valid & ((targets < 0) | (targets >= len(self.class_names)))
        ):
            raise ValueError("targets contain a class outside the configured range")
        for prediction in (teacher, control, kd):
            if torch.any(
                valid & ((prediction < 0) | (prediction >= len(self.class_names)))
            ):
                raise ValueError("predictions contain a class outside the configured range")
        for region in self._REGIONS:
            self._update_region(region, targets, teacher, control, kd, valid)

    @staticmethod
    def _rate(count: int, denominator: int) -> float:
        return count / denominator if denominator else 0.0

    def _summary(self, counts: torch.Tensor) -> dict[str, Any]:
        names = (
            "pixels",
            "teacher_control_agreement",
            "teacher_kd_agreement",
            "control_kd_agreement",
            "teacher_correct_control_wrong",
            "teacher_correct_kd_wrong",
            "kd_recovers_teacher_advantage",
            "teacher_advantage_not_recovered_by_kd",
            "kd_corrects_control_error",
            "kd_introduces_control_error",
        )

        def row(values: torch.Tensor) -> dict[str, Any]:
            raw = {name: int(value) for name, value in zip(names, values.tolist())}
            pixels = raw["pixels"]
            recoverable = raw["teacher_correct_control_wrong"]
            return {
                **raw,
                "teacher_control_agreement_rate": self._rate(
                    raw["teacher_control_agreement"], pixels
                ),
                "teacher_kd_agreement_rate": self._rate(
                    raw["teacher_kd_agreement"], pixels
                ),
                "control_kd_agreement_rate": self._rate(
                    raw["control_kd_agreement"], pixels
                ),
                "teacher_advantage_rate": self._rate(recoverable, pixels),
                "kd_recovery_rate": self._rate(
                    raw["kd_recovers_teacher_advantage"], recoverable
                ),
                "kd_improvement_rate": self._rate(
                    raw["kd_corrects_control_error"], pixels
                ),
                "kd_regression_rate": self._rate(
                    raw["kd_introduces_control_error"], pixels
                ),
                "kd_net_accuracy_delta": self._rate(
                    raw["kd_corrects_control_error"]
                    - raw["kd_introduces_control_error"],
                    pixels,
                ),
            }

        total = counts.sum(dim=0)
        return {
            "overall": row(total),
            "per_ground_truth_class": {
                name: row(counts[index])
                for index, name in enumerate(self.class_names)
            },
        }

    def serializable(self) -> dict[str, Any]:
        """Return JSON-safe all-pixel and optional trimap diagnostics."""
        regions = {"all": self._summary(self._counts["all"])}
        if self.trimap_kernel_size is not None:
            regions["trimap"] = self._summary(self._counts["trimap"])
        return regions
