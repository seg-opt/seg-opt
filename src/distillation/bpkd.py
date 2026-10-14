from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DistillationLossResult:
    """A scalar distillation objective and detached-name-friendly components."""

    total: torch.Tensor
    components: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class BoundaryMasks:
    """Soft per-class edge masks and valid output locations."""

    edge: torch.Tensor
    valid: torch.Tensor

    @property
    def body(self) -> torch.Tensor:
        return (1.0 - self.edge) * self.valid


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
    """Build BPKD's soft class-aware edge masks from dense ground truth.

    Morphology operates at target resolution.  The binary gradient is then
    average-pooled to the requested logit resolution, exactly preserving the
    intended soft-mask construction when output stride is greater than one.
    """
    _validate_kernel_size(edge_kernel_size)
    _validate_labels(labels, num_classes=num_classes, ignore_index=ignore_index)
    if min(output_size) <= 0:
        raise ValueError("output_size dimensions must be positive")

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


def _as_alpha(
    alpha: float | Sequence[float],
    *,
    num_classes: int,
) -> tuple[float, ...]:
    if isinstance(alpha, (int, float)):
        values = (float(alpha),) * num_classes
    else:
        values = tuple(float(value) for value in alpha)
        if len(values) != num_classes:
            raise ValueError("edge_alpha must have one value per class")
    if any(value < 0 for value in values):
        raise ValueError("edge_alpha values must be non-negative")
    return values


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
    if spatial_stride <= 0:
        raise ValueError("spatial_stride must be positive")
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
    num_classes: int,
    spatial_stride: int,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_logits(
        student_logits,
        teacher_logits,
        labels,
        num_classes=num_classes,
    )
    _validate_labels(labels, num_classes=num_classes, ignore_index=ignore_index)
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
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled_student = student_logits / temperature
    scaled_teacher = teacher_logits / temperature
    teacher_probabilities = F.softmax(scaled_teacher, dim=1)
    teacher_log_probabilities = F.log_softmax(scaled_teacher, dim=1)
    student_log_probabilities = F.log_softmax(scaled_student, dim=1)
    return (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=1) * temperature**2


def _spatial_channel_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """CWD-style KL over pixels within each semantic channel."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch, channels, height, width = student_logits.shape
    valid_positions = valid.bool().expand(-1, channels, -1, -1)
    student = (student_logits / temperature).reshape(batch * channels, height * width)
    teacher = (teacher_logits / temperature).reshape(batch * channels, height * width)
    valid_positions = valid_positions.reshape(batch * channels, height * width)

    active = valid_positions.any(dim=1)
    if not active.any():
        return student_logits.sum() * 0

    student = student.masked_fill(~valid_positions, float("-inf"))
    teacher = teacher.masked_fill(~valid_positions, float("-inf"))
    # Avoid all--inf rows while retaining their zero contribution below.
    inactive = ~active
    if inactive.any():
        student[inactive, 0] = 0
        teacher[inactive, 0] = 0

    teacher_probabilities = F.softmax(teacher, dim=1)
    teacher_log_probabilities = F.log_softmax(teacher, dim=1)
    student_log_probabilities = F.log_softmax(student, dim=1)
    divergence = (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=1)
    return divergence[active].mean() * temperature**2


class BPKDLoss(nn.Module):
    """Paper-faithful body/edge KD using only final class potentials."""

    def __init__(
        self,
        *,
        num_classes: int,
        body_weight: float = 20.0,
        edge_weight: float = 50.0,
        edge_alpha: float | Sequence[float] = 2.0,
        edge_kernel_size: int = 7,
        spatial_stride: int = 8,
        body_temperature: float = 1.0,
        edge_temperature: float = 1.0,
        prm: bool = True,
        pom: bool = True,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if body_weight < 0 or edge_weight < 0:
            raise ValueError("distillation weights must be non-negative")
        if body_temperature <= 0 or edge_temperature <= 0:
            raise ValueError("distillation temperatures must be positive")
        if spatial_stride <= 0:
            raise ValueError("spatial_stride must be positive")
        _validate_kernel_size(edge_kernel_size)
        self.num_classes = num_classes
        self.body_weight = float(body_weight)
        self.edge_weight = float(edge_weight)
        self.edge_kernel_size = edge_kernel_size
        self.spatial_stride = spatial_stride
        self.body_temperature = float(body_temperature)
        self.edge_temperature = float(edge_temperature)
        self.prm = prm
        self.pom = pom
        self.ignore_index = ignore_index
        self.register_buffer(
            "edge_alpha",
            torch.tensor(_as_alpha(edge_alpha, num_classes=num_classes)),
            persistent=False,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.body_weight or self.edge_weight)

    def _body_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        masks: BoundaryMasks,
    ) -> torch.Tensor:
        if not self.body_weight:
            return student_logits.sum() * 0
        return _spatial_channel_kl(
            student_logits * masks.body,
            teacher_logits * masks.body,
            masks.valid,
            self.body_temperature,
        )

    def _edge_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        masks: BoundaryMasks,
    ) -> torch.Tensor:
        if not self.edge_weight:
            return student_logits.sum() * 0

        if self.prm:
            student_logits = student_logits * masks.edge
            teacher_logits = teacher_logits * masks.edge
        pixel_kl = _pixel_kl(
            student_logits,
            teacher_logits,
            self.edge_temperature,
        )

        if not self.pom:
            # With POM disabled this is ordinary pixel-wise KL over the valid
            # output grid.  When PRM is on, non-edge locations carry the
            # identically zero-masked class potentials; keeping them in the
            # reduction is what makes PRM and POM independently switchable.
            support = masks.valid[:, 0]
            count = support.sum(dim=(-2, -1))
            values = (pixel_kl * support).sum(dim=(-2, -1))
            per_sample = torch.where(count > 0, values / count.clamp_min(1), 0.0)
            return per_sample.mean()

        counts = (masks.edge > 0).sum(dim=(-2, -1))
        weighted = (pixel_kl.unsqueeze(1) * masks.edge).sum(dim=(-2, -1))
        alpha = self.edge_alpha.to(student_logits)
        per_class = alpha * weighted / counts.clamp_min(1)
        per_class = torch.where(counts > 0, per_class, torch.zeros_like(per_class))
        return per_class.sum(dim=1).mean()

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
            zero = student_logits.sum() * 0
            return DistillationLossResult(
                total=zero,
                components={
                    "raw_body": zero,
                    "raw_edge": zero,
                    "weighted_body": zero,
                    "weighted_edge": zero,
                    "raw_vanilla": zero,
                    "weighted_vanilla": zero,
                },
            )
        student, teacher, _ = _prepared_logits(
            student_logits,
            teacher_logits,
            labels,
            num_classes=self.num_classes,
            spatial_stride=self.spatial_stride,
            ignore_index=self.ignore_index,
        )
        masks = boundary_masks(
            labels,
            num_classes=self.num_classes,
            output_size=tuple(student.shape[-2:]),
            edge_kernel_size=self.edge_kernel_size,
            ignore_index=self.ignore_index,
        )
        raw_body = self._body_loss(student, teacher, masks)
        raw_edge = self._edge_loss(student, teacher, masks)
        weighted_body = raw_body * self.body_weight
        weighted_edge = raw_edge * self.edge_weight
        total = weighted_body + weighted_edge
        zero = total * 0
        return DistillationLossResult(
            total=total,
            components={
                "raw_body": raw_body,
                "raw_edge": raw_edge,
                "weighted_body": weighted_body,
                "weighted_edge": weighted_edge,
                "raw_vanilla": zero,
                "weighted_vanilla": zero,
            },
        )


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
            zero = student_logits.sum() * 0
            return DistillationLossResult(
                total=zero,
                components={
                    "raw_body": zero,
                    "raw_edge": zero,
                    "weighted_body": zero,
                    "weighted_edge": zero,
                    "raw_vanilla": zero,
                    "weighted_vanilla": zero,
                },
            )
        student, teacher, valid = _prepared_logits(
            student_logits,
            teacher_logits,
            labels,
            num_classes=self.num_classes,
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
        zero = weighted_vanilla * 0
        return DistillationLossResult(
            total=weighted_vanilla,
            components={
                "raw_body": zero,
                "raw_edge": zero,
                "weighted_body": zero,
                "weighted_edge": zero,
                "raw_vanilla": raw_vanilla,
                "weighted_vanilla": weighted_vanilla,
            },
        )
