import torch
from torch import nn


def _window_starts(size: int, crop_size: int, stride: int) -> list[int]:
    if size <= crop_size:
        return [0]
    starts = list(range(0, size - crop_size + 1, stride))
    if starts[-1] != size - crop_size:
        starts.append(size - crop_size)
    return starts


def sliding_window_logits(
    model: nn.Module,
    pixel_values: torch.Tensor,
    crop_size: int | None = None,
    stride: int | None = None,
) -> torch.Tensor:
    """Average overlapping logits for bounded-memory segmentation inference."""
    if crop_size is None: return model(pixel_values)
    if crop_size <= 0: raise ValueError("crop_size must be positive")
    stride = crop_size if stride is None else stride
    if not 0 < stride <= crop_size: raise ValueError("stride must be positive and no larger than crop_size")

    height, width = pixel_values.shape[-2:]
    if height <= crop_size and width <= crop_size:
        return model(pixel_values)

    logits_sum = None
    counts = pixel_values.new_zeros((1, 1, height, width))
    for top in _window_starts(height, crop_size, stride):
        bottom = min(top + crop_size, height)
        for left in _window_starts(width, crop_size, stride):
            right = min(left + crop_size, width)
            crop_logits = model(pixel_values[:, :, top:bottom, left:right])
            if logits_sum is None:
                logits_sum = crop_logits.new_zeros(
                    (
                        pixel_values.shape[0],
                        crop_logits.shape[1],
                        height,
                        width,
                    )
                )
            logits_sum[:, :, top:bottom, left:right] += crop_logits
            counts[:, :, top:bottom, left:right] += 1

    if logits_sum is None: raise RuntimeError("sliding-window inference produced no crops")
    return logits_sum / counts.clamp_min(1)
