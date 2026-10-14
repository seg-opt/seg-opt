from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _macro_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    ignore_index: int,
    eps: float,
) -> torch.Tensor:
    """Compute weighted macro Dice loss from segmentation logits."""
    probabilities = logits.softmax(dim=1)
    valid = targets != ignore_index
    safe_targets = targets.masked_fill(~valid, 0)
    target_one_hot = F.one_hot(
        safe_targets,
        num_classes=logits.shape[1],
    ).movedim(-1, 1).to(dtype=logits.dtype)
    valid_mask = valid.unsqueeze(1).to(dtype=logits.dtype)
    probabilities = probabilities * valid_mask
    target_one_hot = target_one_hot * valid_mask

    intersection = (probabilities * target_one_hot).sum(dim=(2, 3))
    cardinality = (probabilities + target_one_hot).sum(dim=(2, 3))
    per_class_loss = 1.0 - 2.0 * intersection / (cardinality + eps)
    weights = class_weights.to(device=logits.device, dtype=logits.dtype)
    return (per_class_loss * weights).sum(dim=1).div(weights.sum()).mean()


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
            soft_dice = _macro_dice_loss(
                logits,
                targets,
                class_weights=self.class_weights,
                ignore_index=self.ignore_index,
                eps=self.dice_eps,
            )
            total = total + self.dice_weight * soft_dice
        return total
