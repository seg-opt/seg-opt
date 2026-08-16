from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F


class DINOv3SegmentationProcessor:
    """Add class-index mask processing to a DINOv3 image processor."""

    def __init__(self, image_processor: Any) -> None:
        self.image_processor = image_processor

    def __call__(
        self,
        images: Any,
        segmentation_maps: Sequence[Any] | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> Any:
        encoded = self.image_processor(
            images=images,
            return_tensors=return_tensors,
            **kwargs,
        )
        if segmentation_maps is None:
            return encoded
        if return_tensors != "pt":
            raise ValueError(
                "segmentation maps require return_tensors='pt' for DINOv3"
            )

        batch_size = encoded["pixel_values"].shape[0]
        if len(segmentation_maps) != batch_size:
            raise ValueError(
                "the number of segmentation maps must match the number of images"
            )

        output_size = encoded["pixel_values"].shape[-2:]
        labels = []
        for segmentation_map in segmentation_maps:
            label = torch.as_tensor(
                np.asarray(segmentation_map).copy(),
                dtype=torch.long,
            )
            if label.ndim != 2:
                raise ValueError("segmentation maps must contain 2D class indices")
            label = F.interpolate(
                label[None, None].float(),
                size=output_size,
                mode="nearest",
            )[0, 0].long()
            labels.append(label)

        encoded["labels"] = torch.stack(labels)
        return encoded
