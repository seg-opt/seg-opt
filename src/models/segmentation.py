from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class SemanticSegmenter(nn.Module):
    """Common interface implemented by every supervised baseline."""

    def __init__(self, class_names: Sequence[str]) -> None:
        super().__init__()
        if not class_names:
            raise ValueError("class_names cannot be empty")
        if len(set(class_names)) != len(class_names):
            raise ValueError("class_names must be unique")
        self.class_names = tuple(class_names)
        self.num_classes = len(self.class_names)

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        """Return parameters that should use the lower backbone learning rate."""
        return iter(())

    def loss_and_scores(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self(images)
        return criterion(scores, labels), scores


def make_segmentation_head(
    in_channels: int,
    num_classes: int,
    head_type: str = "linear",
    batch_norm: bool = True,
    hidden_channels: int = 256,
    dropout: float = 0.1,
) -> nn.Module:
    if head_type == "linear":
        layers: list[nn.Module] = []
        if batch_norm:
            layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.Conv2d(in_channels, num_classes, kernel_size=1))
        return nn.Sequential(*layers)
    if head_type == "lightweight":
        groups = min(32, hidden_channels)
        while hidden_channels % groups:
            groups -= 1
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )
    raise ValueError(f"unsupported segmentation head {head_type!r}")


class BackboneSegmenter(SemanticSegmenter):
    def __init__(self, backbone: nn.Module, in_channels: int, num_classes: int):
        super().__init__([str(index) for index in range(num_classes)])
        self.backbone = backbone
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.backbone.parameters()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output_size = pixel_values.shape[-2:]
        features = self.backbone(pixel_values)
        logits = self.classifier(features)
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class DINOv3Segmenter(SemanticSegmenter):
    """Linear semantic-segmentation head over dense DINOv3 features."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        freeze_backbone: bool = True,
        head_type: str = "linear",
        head_batch_norm: bool = True,
        head_hidden_channels: int = 256,
        head_dropout: float = 0.1,
        unfreeze_last_blocks: int = 0,
        class_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            class_names or [str(index) for index in range(num_classes)]
        )
        if self.num_classes != num_classes:
            raise ValueError("num_classes must match the length of class_names")
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_blocks = unfreeze_last_blocks

        config = backbone.config
        self.backbone_type = config.model_type
        if self.backbone_type == "dinov3_vit":
            self.in_channels = config.hidden_size
            self.patch_size = config.patch_size
            self.num_prefix_tokens = 1 + config.num_register_tokens
        elif self.backbone_type == "dinov3_convnext":
            self.in_channels = config.hidden_sizes[-1]
        else:
            raise ValueError(
                "DINOv3Segmenter requires a DINOv3 ViT or ConvNeXt backbone, "
                f"got {self.backbone_type!r}"
            )

        self.classifier = make_segmentation_head(
            self.in_channels,
            self.num_classes,
            head_type=head_type,
            batch_norm=head_batch_norm,
            hidden_channels=head_hidden_channels,
            dropout=head_dropout,
        )
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            if unfreeze_last_blocks:
                self._unfreeze_last_blocks(unfreeze_last_blocks)
            self.backbone.eval()

    def _unfreeze_last_blocks(self, count: int) -> None:
        layers = getattr(self.backbone, "layer", None)
        if layers is None:
            raise ValueError("this DINOv3 backbone does not expose transformer blocks")
        if not 0 < count <= len(layers):
            raise ValueError(f"unfreeze_last_blocks must be between 0 and {len(layers)}")
        for layer in layers[-count:]:
            layer.requires_grad_(True)
        norm = getattr(self.backbone, "norm", None)
        if norm is not None:
            norm.requires_grad_(True)

    def train(self, mode: bool = True) -> "DINOv3Segmenter":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return (
            parameter
            for parameter in self.backbone.parameters()
            if parameter.requires_grad
        )

    def _backbone_context(self) -> Any:
        has_trainable_parameters = any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )
        return nullcontext() if has_trainable_parameters else torch.no_grad()

    def _vit_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with self._backbone_context():
            outputs = self.backbone(pixel_values=pixel_values)
        patch_tokens = outputs.last_hidden_state[:, self.num_prefix_tokens :]
        grid_height = pixel_values.shape[-2] // self.patch_size
        grid_width = pixel_values.shape[-1] // self.patch_size
        expected_tokens = grid_height * grid_width
        if patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                "DINOv3 patch-token count does not match the input dimensions: "
                f"expected {expected_tokens}, got {patch_tokens.shape[1]}"
            )
        return patch_tokens.transpose(1, 2).reshape(
            pixel_values.shape[0],
            self.in_channels,
            grid_height,
            grid_width,
        )

    def _convnext_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with self._backbone_context():
            outputs = self.backbone(
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
        features = outputs.hidden_states[-1]
        if features.ndim != 4:
            raise ValueError(
                "DINOv3 ConvNeXt backbone did not return a spatial feature map"
            )
        return features

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output_size = pixel_values.shape[-2:]
        if self.backbone_type == "dinov3_vit":
            features = self._vit_features(pixel_values)
        else:
            features = self._convnext_features(pixel_values)

        logits = self.classifier(features)
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class SegFormerSegmenter(SemanticSegmenter):
    """Full-resolution wrapper around the official SegFormer model."""

    def __init__(self, model: nn.Module, class_names: Sequence[str]) -> None:
        super().__init__(class_names)
        self.model = model

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.model.segformer.parameters()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output_size = pixel_values.shape[-2:]
        outputs = self.model(pixel_values=pixel_values)
        return F.interpolate(
            outputs.logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class Mask2FormerSegmenter(SemanticSegmenter):
    """Semantic wrapper retaining Mask2Former's native matching objective."""

    def __init__(
        self,
        model: nn.Module,
        class_names: Sequence[str],
        ignore_index: int = 255,
    ) -> None:
        super().__init__(class_names)
        self.model = model
        self.ignore_index = ignore_index

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.model.model.pixel_level_module.encoder.parameters()

    def dense_targets(
        self,
        labels: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if labels.ndim != 3:
            raise ValueError("labels must have shape [batch, height, width]")

        mask_labels: list[torch.Tensor] = []
        class_labels: list[torch.Tensor] = []
        for sample_index, label in enumerate(labels):
            present = torch.unique(label[label != self.ignore_index])
            if not present.numel():
                raise ValueError(
                    "Mask2Former cannot train on an all-ignore target "
                    f"(sample {sample_index})"
                )
            if present.numel() and (
                present.min() < 0 or present.max() >= self.num_classes
            ):
                raise ValueError("labels contain a class outside the configured range")
            present = present.to(dtype=torch.long)
            masks = torch.stack(
                [(label == class_id) for class_id in present], dim=0
            )
            mask_labels.append(masks.to(dtype=torch.float32))
            class_labels.append(present)
        return mask_labels, class_labels

    def query_outputs_to_dense(
        self,
        class_queries_logits: torch.Tensor,
        masks_queries_logits: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if class_queries_logits.ndim != 3 or masks_queries_logits.ndim != 4:
            raise ValueError("Mask2Former query outputs have invalid dimensions")
        if class_queries_logits.shape[:2] != masks_queries_logits.shape[:2]:
            raise ValueError("class and mask query counts do not match")
        if class_queries_logits.shape[-1] != self.num_classes + 1:
            raise ValueError("class queries must include one no-object class")

        class_probabilities = class_queries_logits.softmax(dim=-1)[
            ..., : self.num_classes
        ]
        resized_mask_logits = F.interpolate(
            masks_queries_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        mask_probabilities = resized_mask_logits.sigmoid()
        return torch.einsum(
            "bqc,bqhw->bchw", class_probabilities, mask_probabilities
        )

    def _scores_from_outputs(
        self,
        outputs: Any,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        return self.query_outputs_to_dense(
            outputs.class_queries_logits,
            outputs.masks_queries_logits,
            output_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return self._scores_from_outputs(outputs, pixel_values.shape[-2:])

    def loss_and_scores(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del criterion
        mask_labels, class_labels = self.dense_targets(labels)
        outputs = self.model(
            pixel_values=images,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        if outputs.loss is None:
            raise RuntimeError("Mask2Former did not return its native training loss")
        scores = self._scores_from_outputs(outputs, images.shape[-2:])
        return outputs.loss, scores


class QuerySegmenter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "query-based segmentation teachers are not implemented yet"
        )
