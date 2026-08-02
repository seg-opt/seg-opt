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


class QuerySegmenter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "query-based segmentation teachers are not implemented yet"
        )
