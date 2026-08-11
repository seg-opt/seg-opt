from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image


class SegmentationImageProcessor:
    def __init__(self, image_size: int):
        self.image_size = image_size
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __call__(
        self,
        images: Sequence[np.ndarray],
        segmentation_maps: Sequence[np.ndarray] | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("SegmentationImageProcessor requires return_tensors='pt'")

        pixels = []
        for image in images:
            resized = Image.fromarray(image).resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BILINEAR,
            )
            tensor = (
                torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float()
                / 255
            )
            pixels.append((tensor - self.mean) / self.std)

        encoded = {"pixel_values": torch.stack(pixels)}
        if segmentation_maps is not None:
            labels = []
            for segmentation_map in segmentation_maps:
                resized = Image.fromarray(
                    segmentation_map.astype(np.int32), mode="I"
                ).resize(
                    (self.image_size, self.image_size),
                    resample=Image.Resampling.NEAREST,
                )
                labels.append(torch.from_numpy(np.asarray(resized).copy()).long())
            encoded["labels"] = torch.stack(labels)
        return encoded


class StudentTeacherProcessor:
    def __init__(self, student_processor: Any, teacher_processor: Any) -> None:
        self.student_processor = student_processor
        self.teacher_processor = teacher_processor

    def __call__(
        self,
        images: Sequence[np.ndarray],
        segmentation_maps: Sequence[np.ndarray] | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, torch.Tensor]:
        student = self.student_processor(
            images=images,
            segmentation_maps=segmentation_maps,
            return_tensors=return_tensors,
        )
        teacher = self.teacher_processor(images=images, return_tensors=return_tensors)
        return {
            "student_pixel_values": student["pixel_values"],
            "teacher_pixel_values": teacher["pixel_values"],
            "labels": student["labels"],
        }