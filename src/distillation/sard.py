"""Structure-Aware Representation Distillation (SARD, Liu et al., CVPR 2026).

SARD aligns student and teacher *features* rather than outputs and weights
every location by a structure-importance map computed from the frozen
teacher's features: structure-tensor anisotropy (edges), structure-tensor
geometric mean (corners/junctions), and local feature dispersion (crowding).

The paper leaves several details open; this implementation fixes them as:

* Both feature maps are resized to the label grid divided by
  ``spatial_stride`` (the same distillation grid as BPKD and vanilla KD).
* ``P_T`` is a frozen random orthogonal 1x1 projection.  The paper states that
  gradients flow only through the student, ``P_S`` and the denoising head, and
  a trainable ``P_T`` would let both projections collapse to zero.
* Projected features are standardized per sample and channel (no affine) so
  the loss weights do not depend on the teacher's activation scale.
* ``E``, ``C`` and ``D`` are each divided by their per-image mean before being
  combined, so ``beta`` weights mix terms of comparable magnitude.
* The structure tensor is integrated over a 3x3 window before its
  eigenvalues are taken; the paper's per-pixel sum over channels alone is
  rank-one whenever channels share a gradient direction.
* ``D`` is the RMS feature dispersion in the window, which has a closed form
  through average pooling, instead of the mean of L2 norms.
* The denoising head sees the noisy teacher features concatenated with the
  student features.  Predicting the noise from student features alone, as the
  paper's equation literally reads, is impossible because the noise is
  independent of the student; conditioning on the noisy target makes the noise
  recoverable exactly when the student features encode the clean teacher
  features.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.distillation.bpkd import DistillationLossResult, valid_mask_at_size


class _CapturedFeatures:
    value: torch.Tensor | None = None


@contextmanager
def capture_features(model: nn.Module) -> Iterator[_CapturedFeatures]:
    """Record the model's distillation feature map during forwards in scope."""
    tap_getter = getattr(model, "distillation_feature_tap", None)
    if not callable(tap_getter):
        raise TypeError("model must implement distillation_feature_tap for SARD")
    tap = tap_getter()
    captured = _CapturedFeatures()

    def hook(_module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        captured.value = inputs[0] if tap.use_input else output

    handle = tap.module.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
) / 8.0


def _sobel_gradients(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channels = features.shape[1]
    kernel_x = _SOBEL_X.to(features)
    kernel_x = kernel_x.expand(channels, 1, 3, 3)
    kernel_y = kernel_x.transpose(-2, -1)
    padded = F.pad(features, (1, 1, 1, 1), mode="replicate")
    return (
        F.conv2d(padded, kernel_x, groups=channels),
        F.conv2d(padded, kernel_y, groups=channels),
    )


def _per_image_mean_normalized(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    count = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
    mean = (values * valid).sum(dim=(-2, -1), keepdim=True) / count
    return values / mean.clamp_min(1e-12)


@dataclass(frozen=True)
class StructureCues:
    edge: torch.Tensor
    corner: torch.Tensor
    density: torch.Tensor


def structure_cues(
    teacher_features: torch.Tensor,
    *,
    density_radius: int = 7,
) -> StructureCues:
    """Return the raw ``E``, ``C`` and ``D`` maps, each shaped ``[B, H, W]``."""
    if teacher_features.ndim != 4:
        raise ValueError("teacher_features must have shape [B, C, H, W]")
    if density_radius < 0:
        raise ValueError("density_radius must be non-negative")
    features = teacher_features.detach().float()
    gradient_x, gradient_y = _sobel_gradients(features)
    tensor = torch.stack(
        [
            (gradient_x * gradient_x).sum(dim=1),
            (gradient_y * gradient_y).sum(dim=1),
            (gradient_x * gradient_y).sum(dim=1),
        ],
        dim=1,
    )
    # Integrate J over a 3x3 window, as in classical structure-tensor
    # analysis.  Without it J is a sum of rank-one terms that stays rank-one
    # whenever channels share a gradient direction, so corners give C = 0.
    tensor = F.avg_pool2d(F.pad(tensor, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    j_xx, j_yy, j_xy = tensor.unbind(dim=1)
    half_trace = (j_xx + j_yy) / 2
    root = torch.sqrt(((j_xx - j_yy) / 2) ** 2 + j_xy**2)
    largest = half_trace + root
    smallest = (half_trace - root).clamp_min(0)

    kernel = 2 * density_radius + 1
    padding = (density_radius,) * 4
    local_mean = F.avg_pool2d(
        F.pad(features, padding, mode="replicate"), kernel, stride=1
    )
    local_square = F.avg_pool2d(
        F.pad(features.square(), padding, mode="replicate"), kernel, stride=1
    )
    variance = (local_square - local_mean.square()).sum(dim=1).clamp_min(0)
    return StructureCues(
        edge=largest - smallest,
        corner=torch.sqrt(largest * smallest),
        density=torch.sqrt(variance),
    )


def structure_weights(
    teacher_features: torch.Tensor,
    valid: torch.Tensor,
    *,
    edge_weight: float = 2.0,
    corner_weight: float = 1.0,
    density_weight: float = 1.0,
    density_radius: int = 7,
) -> torch.Tensor:
    """Return ``W``: non-negative ``[B, H, W]`` weights summing to one per image.

    Ignored locations receive zero weight.  With every beta at zero the map is
    uniform over valid locations, which is the paper's ``W(i) = 1`` ablation.
    """
    if valid.ndim != 3:
        raise ValueError("valid must have shape [B, H, W]")
    valid = valid.float()
    betas = (edge_weight, corner_weight, density_weight)
    if any(beta < 0 for beta in betas):
        raise ValueError("structure weights must be non-negative")
    if not any(betas):
        score = torch.ones_like(valid)
    else:
        cues = structure_cues(teacher_features, density_radius=density_radius)
        score = torch.zeros_like(valid)
        for beta, cue in zip(betas, (cues.edge, cues.corner, cues.density)):
            if beta:
                score = score + beta * _per_image_mean_normalized(cue, valid)
    score = score * valid
    total = score.sum(dim=(-2, -1), keepdim=True)
    return torch.where(total > 0, score / total.clamp_min(1e-12), score)


def _standardized(features: torch.Tensor) -> torch.Tensor:
    mean = features.mean(dim=(-2, -1), keepdim=True)
    std = features.std(dim=(-2, -1), keepdim=True, unbiased=False)
    return (features - mean) / (std + 1e-5)


class SARDLoss(nn.Module):
    """Structure-weighted feature consistency plus distribution alignment."""

    requires_features = True

    def __init__(
        self,
        *,
        student_channels: int,
        teacher_channels: int,
        projection_channels: int = 256,
        denoiser_hidden_channels: int = 128,
        feature_weight: float = 1.0,
        distribution_weight: float = 0.5,
        edge_weight: float = 2.0,
        corner_weight: float = 1.0,
        density_weight: float = 1.0,
        density_radius: int = 7,
        noise_alpha: float = 0.5,
        spatial_stride: int = 8,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        if min(student_channels, teacher_channels, projection_channels) <= 0:
            raise ValueError("feature channel counts must be positive")
        if denoiser_hidden_channels <= 0:
            raise ValueError("denoiser_hidden_channels must be positive")
        if feature_weight < 0 or distribution_weight < 0:
            raise ValueError("SARD loss weights must be non-negative")
        if not 0 < noise_alpha < 1:
            raise ValueError("noise_alpha must be in (0, 1)")
        if spatial_stride <= 0:
            raise ValueError("spatial_stride must be positive")
        self.feature_weight = float(feature_weight)
        self.distribution_weight = float(distribution_weight)
        self.edge_weight = float(edge_weight)
        self.corner_weight = float(corner_weight)
        self.density_weight = float(density_weight)
        self.density_radius = density_radius
        self.noise_alpha = float(noise_alpha)
        self.spatial_stride = spatial_stride
        self.ignore_index = ignore_index

        self.student_projection = nn.Conv2d(
            student_channels, projection_channels, kernel_size=1
        )
        self.teacher_projection = nn.Conv2d(
            teacher_channels, projection_channels, kernel_size=1, bias=False
        )
        nn.init.orthogonal_(self.teacher_projection.weight)
        self.teacher_projection.requires_grad_(False)
        self.denoiser = nn.Sequential(
            nn.Conv2d(
                2 * projection_channels,
                denoiser_hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(denoiser_hidden_channels, projection_channels, kernel_size=1),
        )

    def _grid(self, features: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        features = features.float()
        if tuple(features.shape[-2:]) == size:
            return features
        if features.shape[-2] >= size[0] and features.shape[-1] >= size[1]:
            return F.adaptive_avg_pool2d(features, size)
        return F.interpolate(features, size=size, mode="bilinear", align_corners=False)

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> DistillationLossResult:
        if student_features.ndim != 4 or teacher_features.ndim != 4:
            raise ValueError("SARD features must have shape [B, C, H, W]")
        if labels.ndim != 3:
            raise ValueError("labels must have shape [B, H, W]")
        if not student_features.shape[0] == teacher_features.shape[0] == labels.shape[0]:
            raise ValueError("features and labels must have matching batch size")
        size = (
            (labels.shape[-2] + self.spatial_stride - 1) // self.spatial_stride,
            (labels.shape[-1] + self.spatial_stride - 1) // self.spatial_stride,
        )
        with torch.autocast(device_type=labels.device.type, enabled=False):
            student = self._grid(student_features, size)
            teacher = self._grid(teacher_features.detach(), size)
            valid = valid_mask_at_size(
                labels, size, ignore_index=self.ignore_index
            )[:, 0]
            with torch.no_grad():
                weights = structure_weights(
                    teacher,
                    valid,
                    edge_weight=self.edge_weight,
                    corner_weight=self.corner_weight,
                    density_weight=self.density_weight,
                    density_radius=self.density_radius,
                )
                projected_teacher = _standardized(self.teacher_projection(teacher))
            projected_student = _standardized(self.student_projection(student))

            raw_feature = (
                weights * (projected_student - projected_teacher).square().mean(dim=1)
            ).sum(dim=(-2, -1)).mean()

            if self.distribution_weight:
                noise = torch.randn_like(projected_teacher)
                noisy_teacher = (
                    self.noise_alpha**0.5 * projected_teacher
                    + (1.0 - self.noise_alpha) ** 0.5 * noise
                )
                predicted = self.denoiser(
                    torch.cat([projected_student, noisy_teacher], dim=1)
                )
                raw_distribution = (
                    weights * (predicted - noise).square().mean(dim=1)
                ).sum(dim=(-2, -1)).mean()
            else:
                raw_distribution = raw_feature * 0

        weighted_feature = raw_feature * self.feature_weight
        weighted_distribution = raw_distribution * self.distribution_weight
        total = weighted_feature + weighted_distribution
        return DistillationLossResult(
            total=total,
            components={
                "raw_feature": raw_feature,
                "raw_distribution": raw_distribution,
                "weighted_feature": weighted_feature,
                "weighted_distribution": weighted_distribution,
            },
        )
