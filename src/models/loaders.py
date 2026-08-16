from collections.abc import Sequence

from torch import nn

from src.models.fast_scnn import FastSCNN
from src.models.processors import DINOv3SegmentationProcessor
from src.models.resnet_unet import ResNet34UNet
from src.models.segmentation import (
    BackboneSegmenter,
    DINOv3Segmenter,
    Mask2FormerSegmenter,
    SegFormerSegmenter,
    SemanticSegmenter,
)


BASELINE_MODEL_NAMES = (
    "fast_scnn",
    "resnet34_unet",
    "segformer_b0",
    "dinov3_vitl16",
    "mask2former_swinl",
)

DEFAULT_MODEL_IDS = {
    "segformer_b0": "nvidia/mit-b0",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "mask2former_swinl": "facebook/mask2former-swin-large-ade-semantic",
}


def _class_names(
    num_classes: int | None,
    class_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if class_names is None:
        if num_classes is None:
            raise ValueError("provide class_names or num_classes")
        class_names = tuple(str(index) for index in range(num_classes))
    else:
        class_names = tuple(class_names)
        if num_classes is not None and len(class_names) != num_classes:
            raise ValueError("num_classes must match the length of class_names")
    if not class_names:
        raise ValueError("class_names cannot be empty")
    if len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be unique")
    return tuple(class_names)


def _label_maps(
    class_names: Sequence[str],
) -> tuple[dict[int, str], dict[str, int]]:
    id2label = dict(enumerate(class_names))
    return id2label, {name: class_id for class_id, name in id2label.items()}


def load_resnet34_unet(
    class_names: Sequence[str],
    pretrained: bool = True,
) -> ResNet34UNet:
    from torchvision import models

    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    encoder = models.resnet34(weights=weights)
    return ResNet34UNet(encoder, class_names)


def load_segformer_b0(
    model_id: str,
    class_names: Sequence[str],
    pretrained: bool = True,
) -> SegFormerSegmenter:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    id2label, label2id = _label_maps(class_names)
    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_id,
            num_labels=len(class_names),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    else:
        config = SegformerConfig.from_pretrained(
            model_id,
            num_labels=len(class_names),
            id2label=id2label,
            label2id=label2id,
        )
        model = SegformerForSemanticSegmentation(config)
    return SegFormerSegmenter(model, class_names)


def _build_dinov3(
    model_id: str,
    class_names: Sequence[str],
    freeze_backbone: bool = True,
    head_type: str = "linear",
    head_batch_norm: bool = True,
    head_hidden_channels: int = 256,
    head_dropout: float = 0.1,
    unfreeze_last_blocks: int = 0,
) -> DINOv3Segmenter:
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(model_id)
    return DINOv3Segmenter(
        backbone=backbone,
        num_classes=len(class_names),
        freeze_backbone=freeze_backbone,
        head_type=head_type,
        head_batch_norm=head_batch_norm,
        head_hidden_channels=head_hidden_channels,
        head_dropout=head_dropout,
        unfreeze_last_blocks=unfreeze_last_blocks,
        class_names=class_names,
    )


def _build_mask2former(
    model_id: str,
    class_names: Sequence[str],
    ignore_index: int = 255,
) -> Mask2FormerSegmenter:
    from transformers import Mask2FormerForUniversalSegmentation

    id2label, label2id = _label_maps(class_names)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_id,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return Mask2FormerSegmenter(
        model,
        class_names=class_names,
        ignore_index=ignore_index,
    )


def load_model(
    model_name: str,
    *,
    class_names: Sequence[str] | None = None,
    num_classes: int | None = None,
    pretrained: bool | None = None,
    model_id: str | None = None,
    ignore_index: int = 255,
) -> SemanticSegmenter:
    """Build one of the five canonical supervised segmentation baselines."""
    if model_name not in BASELINE_MODEL_NAMES:
        allowed = ", ".join(BASELINE_MODEL_NAMES)
        raise ValueError(f"unsupported baseline {model_name!r}; choose one of: {allowed}")

    names = _class_names(num_classes, class_names)
    use_pretrained = model_name != "fast_scnn" if pretrained is None else pretrained

    if model_name == "fast_scnn":
        if use_pretrained:
            raise ValueError(
                "Fast-SCNN does not have pretrained weights; set pretrained=false"
            )
        return FastSCNN(len(names), class_names=names)
    if model_name == "resnet34_unet":
        return load_resnet34_unet(names, pretrained=use_pretrained)
    if model_name == "segformer_b0":
        return load_segformer_b0(
            model_id or DEFAULT_MODEL_IDS[model_name],
            names,
            pretrained=use_pretrained,
        )
    if not use_pretrained:
        raise ValueError(f"{model_name} requires pretrained=true")
    if model_name == "dinov3_vitl16":
        return _build_dinov3(
            model_id or DEFAULT_MODEL_IDS[model_name],
            names,
            freeze_backbone=True,
            head_type="linear",
            head_batch_norm=True,
        )
    return _build_mask2former(
        model_id or DEFAULT_MODEL_IDS[model_name],
        names,
        ignore_index=ignore_index,
    )


# Compatibility helpers for existing experiments. The canonical trainer uses
# load_model above, whose dispatch is intentionally limited to five baselines.
def load_mask2former(
    model_id: str,
    num_classes: int,
    pretrained_head: bool = True,
) -> tuple[nn.Module, object]:
    del pretrained_head
    from transformers import AutoImageProcessor

    names = _class_names(num_classes, None)
    model = _build_mask2former(model_id, names)
    return model, AutoImageProcessor.from_pretrained(model_id)


def load_dinov3(
    model_id: str,
    num_classes: int,
    freeze_backbone: bool = True,
    head_type: str = "linear",
    head_batch_norm: bool = True,
    head_hidden_channels: int = 256,
    head_dropout: float = 0.1,
    unfreeze_last_blocks: int = 0,
) -> tuple[nn.Module, object]:
    from transformers import AutoImageProcessor

    names = _class_names(num_classes, None)
    model = _build_dinov3(
        model_id=model_id,
        class_names=names,
        freeze_backbone=freeze_backbone,
        head_type=head_type,
        head_batch_norm=head_batch_norm,
        head_hidden_channels=head_hidden_channels,
        head_dropout=head_dropout,
        unfreeze_last_blocks=unfreeze_last_blocks,
    )
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    return model, DINOv3SegmentationProcessor(image_processor)


def load_sam3(
    model_id: str,
    class_prompts: Sequence[str],
) -> tuple[nn.Module, object]:
    del model_id, class_prompts
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
    if model_name == "fast_scnn":
        if pretrained:
            raise ValueError(
                "Fast-SCNN does not have pretrained weights; set pretrained=false"
            )
        return FastSCNN(num_classes)
    if model_name == "resnet34_unet":
        names = _class_names(num_classes, None)
        return load_resnet34_unet(names, pretrained=pretrained)
    if model_name == "segformer_b0":
        names = _class_names(num_classes, None)
        return load_segformer_b0(
            DEFAULT_MODEL_IDS[model_name], names, pretrained=pretrained
        )

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
