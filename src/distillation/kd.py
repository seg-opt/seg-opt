from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DistillationLossResult:
    """A scalar distillation objective and detached-name-friendly components."""

    total: torch.Tensor
    components: Mapping[str, torch.Tensor]


_KD_COMPONENT_KEYS = (
    "raw_vanilla", "weighted_vanilla",
)


def _kd_result(total: torch.Tensor, **overrides: torch.Tensor) -> DistillationLossResult:
    zero = total * 0
    return DistillationLossResult(
        total=total,
        components={key: overrides.get(key, zero) for key in _KD_COMPONENT_KEYS},
    )


@dataclass(frozen=True)
class BoundaryMasks:
    """Soft per-class edge masks and valid output locations."""

    edge: torch.Tensor
    valid: torch.Tensor


def _same_max_pool(inputs: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Square, replicate-padded max-pooling with unchanged spatial shape."""
    padding = kernel_size // 2
    return F.max_pool2d(
        F.pad(inputs, (padding, padding, padding, padding), mode="replicate"),
        kernel_size=kernel_size,
        stride=1,
    )


def _validate_kernel_size(kernel_size: int) -> None:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("edge_kernel_size must be a positive odd integer")


def _validate_labels(
    labels: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
) -> None:
    if labels.ndim != 3:
        raise ValueError("labels must have shape [B, H, W]")
    valid = labels != ignore_index
    if valid.any() and (
        labels[valid].min() < 0 or labels[valid].max() >= num_classes
    ):
        raise ValueError("labels contain a class outside the configured range")


def valid_mask_at_size(
    labels: torch.Tensor,
    output_size: tuple[int, int],
    *,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Return output locations composed entirely of non-ignored source pixels."""
    if labels.ndim != 3:
        raise ValueError("labels must have shape [B, H, W]")
    if min(output_size) <= 0:
        raise ValueError("output_size dimensions must be positive")
    valid = (labels != ignore_index).unsqueeze(1).float()
    pooled = F.adaptive_avg_pool2d(valid, output_size)
    return (pooled >= 1.0 - 1e-6).to(dtype=valid.dtype)


def boundary_masks(
    labels: torch.Tensor,
    *,
    num_classes: int,
    output_size: tuple[int, int],
    edge_kernel_size: int,
    ignore_index: int = 255,
) -> BoundaryMasks:
    """Build soft class-aware edge masks from dense ground truth."""
    _validate_kernel_size(edge_kernel_size)
    _validate_labels(labels, num_classes=num_classes, ignore_index=ignore_index)

    valid = labels != ignore_index
    safe_labels = labels.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_labels, num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=torch.float32)
    one_hot = one_hot * valid.unsqueeze(1)

    dilated = _same_max_pool(one_hot, edge_kernel_size)
    eroded = 1.0 - _same_max_pool(1.0 - one_hot, edge_kernel_size)
    edge = (dilated - eroded).clamp_(0.0, 1.0)

    # Void pixels and their morphology neighbourhood are excluded so an ignore
    # region cannot masquerade as a semantic boundary.
    valid_region = 1.0 - _same_max_pool(
        1.0 - valid.unsqueeze(1).float(),
        edge_kernel_size,
    )
    edge = edge * valid_region
    valid_output = valid_mask_at_size(
        labels,
        output_size,
        ignore_index=ignore_index,
    )
    edge = F.adaptive_avg_pool2d(edge, output_size) * valid_output
    return BoundaryMasks(edge=edge, valid=valid_output)


def trimap_mask(
    labels: torch.Tensor,
    *,
    num_classes: int,
    edge_kernel_size: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Return a single-channel GT Trimap mask at label resolution."""
    masks = boundary_masks(
        labels,
        num_classes=num_classes,
        output_size=tuple(labels.shape[-2:]),
        edge_kernel_size=edge_kernel_size,
        ignore_index=ignore_index,
    )
    return masks.edge.amax(dim=1).to(dtype=torch.bool) & masks.valid[:, 0].bool()


def _validate_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
) -> None:
    if student_logits.ndim != 4 or teacher_logits.ndim != 4:
        raise ValueError("student_logits and teacher_logits must have shape [B, C, H, W]")
    if student_logits.shape[0] != labels.shape[0]:
        raise ValueError("student logits and labels must have matching batch size")
    if teacher_logits.shape[0] != student_logits.shape[0]:
        raise ValueError("student and teacher logits must have matching batch size")
    if student_logits.shape[1] != num_classes or teacher_logits.shape[1] != num_classes:
        raise ValueError("logit channels must match the configured class count")


def _distillation_size(logits: torch.Tensor, spatial_stride: int) -> tuple[int, int]:
    height, width = logits.shape[-2:]
    return (
        (height + spatial_stride - 1) // spatial_stride,
        (width + spatial_stride - 1) // spatial_stride,
    )


def _prepared_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    spatial_stride: int,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_size = _distillation_size(student_logits, spatial_stride)
    student = F.interpolate(
        student_logits.float(),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    teacher = F.interpolate(
        teacher_logits.detach().float(),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    valid = valid_mask_at_size(labels, output_size, ignore_index=ignore_index)
    return student, teacher, valid


def _pixel_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    scaled_student = student_logits / temperature
    scaled_teacher = teacher_logits / temperature
    teacher_probabilities = F.softmax(scaled_teacher, dim=1)
    teacher_log_probabilities = F.log_softmax(scaled_teacher, dim=1)
    student_log_probabilities = F.log_softmax(scaled_student, dim=1)
    return (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=1) * temperature**2


class PixelWiseKDLoss(nn.Module):
    """Target-aware vanilla semantic KD with valid-pixel normalization."""

    def __init__(
        self,
        *,
        num_classes: int,
        weight: float = 1.0,
        temperature: float = 1.0,
        spatial_stride: int = 8,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if weight < 0:
            raise ValueError("weight must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if spatial_stride <= 0:
            raise ValueError("spatial_stride must be positive")
        self.num_classes = num_classes
        self.weight = float(weight)
        self.temperature = float(temperature)
        self.spatial_stride = spatial_stride
        self.ignore_index = ignore_index

    @property
    def enabled(self) -> bool:
        return bool(self.weight)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> DistillationLossResult:
        _validate_logits(
            student_logits,
            teacher_logits,
            labels,
            num_classes=self.num_classes,
        )
        _validate_labels(
            labels,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        if not self.enabled:
            return _kd_result(student_logits.sum() * 0)
        student, teacher, valid = _prepared_logits(
            student_logits,
            teacher_logits,
            labels,
            spatial_stride=self.spatial_stride,
            ignore_index=self.ignore_index,
        )
        pixel_kl = _pixel_kl(student, teacher, self.temperature)
        count = valid.sum()
        raw_vanilla = torch.where(
            count > 0,
            (pixel_kl * valid[:, 0]).sum() / count.clamp_min(1),
            student.sum() * 0,
        )
        weighted_vanilla = raw_vanilla * self.weight
        return _kd_result(
            weighted_vanilla,
            raw_vanilla=raw_vanilla,
            weighted_vanilla=weighted_vanilla,
        )
