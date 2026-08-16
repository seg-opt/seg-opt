from collections.abc import Sequence

import torch
from kornia.losses import dice_loss
from torch import nn
from torch.nn import functional as F


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_weights: Sequence[float] | None = None,
        cross_entropy_weight: float = 1.0,
        dice_weight: float = 0.5,
        ignore_index: int = 255,
        dice_eps: float = 1e-8,
    ) -> None:
        super().__init__()

        weights = (
            torch.ones(num_classes, dtype=torch.float32)
            if class_weights is None
            else torch.tensor(class_weights, dtype=torch.float32)
        )
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.dice_eps = dice_eps
        self.register_buffer("class_weights", weights)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 4 or targets.ndim != 3:
            raise ValueError("expected logits [B, C, H, W] and targets [B, H, W]")
        if (
            logits.shape[0] != targets.shape[0]
            or logits.shape[-2:] != targets.shape[-2:]
        ):
            raise ValueError("logits and targets must have matching batch and spatial shapes")
        if logits.shape[1] != self.class_weights.numel():
            raise ValueError("logit channels must match the configured class count")

        valid = targets != self.ignore_index
        differentiable_zero = logits.sum() * 0
        if not valid.any():
            return differentiable_zero

        total = differentiable_zero
        if self.cross_entropy_weight:
            cross_entropy = F.cross_entropy(
                logits,
                targets,
                weight=self.class_weights.to(logits),
                ignore_index=self.ignore_index,
            )
            total = total + self.cross_entropy_weight * cross_entropy
        if self.dice_weight:
            soft_dice = dice_loss(
                logits,
                targets,
                average="macro",
                eps=self.dice_eps,
                weight=self.class_weights.to(logits),
                ignore_index=self.ignore_index,
            )
            total = total + self.dice_weight * soft_dice
        return total
