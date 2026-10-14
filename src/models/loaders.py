from collections.abc import Sequence

from src.models.fast_scnn import FastSCNN
from src.models.resnet_unet import ResNet34UNet
from src.models.segmentation import (
    DINOv3Segmenter,
    Mask2FormerSegmenter,
    SegFormerSegmenter,
    SemanticSegmenter,
)


BASELINE_MODEL_NAMES = (
    "fast_scnn",
    "resnet34_unet",
    "segformer_b0",
    "segformer_b2",
    "segformer_b4",
    "dinov3_vitl16",
    "mask2former_swinl",
)

DEFAULT_MODEL_IDS = {
    "segformer_b0": "nvidia/mit-b0",
    "segformer_b2": "nvidia/mit-b2",
    "segformer_b4": "nvidia/mit-b4",
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


def load_segformer(
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
    """Build one of the supervised segmentation benchmark models."""
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
    if model_name in {"segformer_b0", "segformer_b2", "segformer_b4"}:
        return load_segformer(
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
