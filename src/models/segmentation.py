import torch
from torch import nn
from torch.nn import functional as F


class BackboneSegmenter(nn.Module):
    def __init__(self, backbone: nn.Module, in_channels: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

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


class DINOv3Segmenter(nn.Module):
    """Linear semantic-segmentation head over dense DINOv3 features."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

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

        self.classifier = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "DINOv3Segmenter":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _vit_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
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


class QuerySegmenter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "query-based segmentation teachers are not implemented yet"
        )
