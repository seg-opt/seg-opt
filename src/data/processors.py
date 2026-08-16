from typing import Any, Sequence

import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegmentationImageProcessor:
    def __init__(self) -> None:
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]

    @staticmethod
    def _image_tensor(image: Any) -> torch.Tensor:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 image, got shape {array.shape}")
        return torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0

    def __call__(
        self,
        images: Sequence[Any],
        segmentation_maps: Sequence[Any] | None = None,
        return_tensors: str | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("SegmentationImageProcessor requires return_tensors='pt'")

        pixels = [
            (self._image_tensor(image) - self.mean) / self.std
            for image in images
        ]
        encoded = {"pixel_values": torch.stack(pixels)}

        if segmentation_maps is not None:
            labels = [
                torch.from_numpy(np.asarray(mask).copy()).long()
                for mask in segmentation_maps
            ]
            if any(label.shape != pixels[0].shape[-2:] for label in labels):
                raise ValueError("segmentation maps must match the image spatial shape")
            encoded["labels"] = torch.stack(labels)
        return encoded
