from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.models.segmentation import SemanticSegmenter


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(inputs))


class LinearBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion: int = 6,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.block(inputs)
        return inputs + outputs if self.use_residual else outputs


def _bottleneck_stage(
    in_channels: int,
    out_channels: int,
    blocks: int,
    stride: int,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        LinearBottleneck(in_channels, out_channels, stride=stride),
    ]
    layers.extend(
        LinearBottleneck(out_channels, out_channels, stride=1)
        for _ in range(1, blocks)
    )
    return nn.Sequential(*layers)


class PyramidPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_sizes: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        branch_channels = in_channels // len(pool_sizes)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                ConvBNReLU(in_channels, branch_channels, kernel_size=1),
            )
            for pool_size in pool_sizes
        )
        self.bottleneck = ConvBNReLU(
            in_channels + len(pool_sizes) * branch_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        size = inputs.shape[-2:]
        pooled = [
            F.interpolate(
                branch(inputs),
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            for branch in self.branches
        ]
        return self.bottleneck(torch.cat([inputs, *pooled], dim=1))


class LearningToDownsample(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            ConvBNReLU(3, 32, stride=2),
            DepthwiseSeparableConv(32, 48, stride=2),
            DepthwiseSeparableConv(48, 64, stride=2),
        )


class GlobalFeatureExtractor(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            _bottleneck_stage(64, 64, blocks=3, stride=2),
            _bottleneck_stage(64, 96, blocks=3, stride=2),
            _bottleneck_stage(96, 128, blocks=3, stride=1),
            PyramidPooling(128, 128),
        )


class FeatureFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_depthwise = nn.Sequential(
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=4,
                dilation=4,
                groups=128,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.context_projection = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        spatial_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        context_features = F.interpolate(
            context_features,
            size=spatial_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        context_features = self.context_projection(
            self.context_depthwise(context_features)
        )
        spatial_features = self.spatial_projection(spatial_features)
        return self.activation(context_features + spatial_features)


class Classifier(nn.Sequential):
    def __init__(self, num_classes: int) -> None:
        super().__init__(
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 128),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )


class FastSCNN(SemanticSegmenter):
    """Fast-SCNN semantic segmenter with a single full-resolution output."""

    def __init__(
        self,
        num_classes: int,
        class_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            class_names or [str(index) for index in range(num_classes)]
        )
        if self.num_classes != num_classes:
            raise ValueError("num_classes must match the length of class_names")
        self.learning_to_downsample = LearningToDownsample()
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.feature_fusion = FeatureFusion()
        self.classifier = Classifier(self.num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output_size = pixel_values.shape[-2:]
        spatial_features = self.learning_to_downsample(pixel_values)
        context_features = self.global_feature_extractor(spatial_features)
        fused_features = self.feature_fusion(spatial_features, context_features)
        logits = self.classifier(fused_features)
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
