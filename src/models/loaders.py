from __future__ import annotations

from typing import Sequence

from torch import nn

from src.models.processors import DINOv3SegmentationProcessor
from src.models.segmentation import (
    BackboneSegmenter,
    DINOv3Segmenter,
    QuerySegmenter,
)


def load_mask2former(
    model_id: str,
    num_classes: int,
    pretrained_head: bool = True,
) -> tuple[nn.Module, object]:
     raise NotImplementedError("Mask2Former segmentation loader is not implemented yet")


def load_dinov3(
    model_id: str,
    num_classes: int,
    freeze_backbone: bool = True,
) -> tuple[nn.Module, object]:
    """Load a pretrained DINOv3 backbone with a new linear segmentation head."""
    from transformers import AutoImageProcessor, AutoModel

    backbone = AutoModel.from_pretrained(model_id)
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = DINOv3Segmenter(
        backbone=backbone,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
    )
    return model, DINOv3SegmentationProcessor(image_processor)


def load_sam3(
    model_id: str,
    class_prompts: Sequence[str],
) -> tuple[nn.Module, object]:
    raise NotImplementedError("SAM3 segmentation loader is not implemented yet")


def _load_torchvision_backbone(
    model_name: str,
    pretrained: bool,
) -> tuple[nn.Module, int]:
    from torchvision import models

    model = models.get_model(
        model_name,
        weights="DEFAULT" if pretrained else None,
    )

    if model_name.startswith("resnet"):
        backbone = nn.Sequential(*list(model.children())[:-2])
        return backbone, model.fc.in_features

    backbone = model.features
    first_linear = next(
        layer for layer in model.classifier.modules() if isinstance(layer, nn.Linear)
    )
    return backbone, first_linear.in_features


def load_student(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    backbone, in_channels = _load_torchvision_backbone(model_name, pretrained)
    return BackboneSegmenter(backbone, in_channels, num_classes)


def load_teacher(
    model_name: str,
    model_id: str,
    num_classes: int,
    class_prompts: Sequence[str] | None = None,
) -> tuple[nn.Module, object]:
    if model_name == "mask2former":
        return load_mask2former(model_id, num_classes)
    if model_name == "dinov3":
        return load_dinov3(model_id, num_classes)
    if model_name == "sam3":
        if class_prompts is None:
            raise ValueError("SAM3 requires one text prompt per class")
        return load_sam3(model_id, class_prompts)

    raise ValueError(f"unsupported teacher {model_name!r}")
