from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.models.segmentation import Mask2FormerSegmenter


def _group_count(channels: int) -> int:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return groups


class PyramidProjection(nn.Sequential):
    """Project one ViT token grid before resizing it into a feature pyramid."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.GELU(),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )


class DINOv3PyramidBackbone(nn.Module):
    """Expose frozen DINOv3 ViT layers as Mask2Former /4-/32 features.

    Mask2Former expects a hierarchical convolutional backbone. DINOv3 ViTs
    instead return same-resolution patch tokens, so four evenly spaced
    transformer layers are normalized, projected, and resized to the spatial
    strides expected by the pretrained pixel decoder.
    """

    strides = (4, 8, 16, 32)

    def __init__(
        self,
        backbone: nn.Module,
        output_channels: Sequence[int],
    ) -> None:
        super().__init__()
        config = backbone.config
        if config.model_type != "dinov3_vit":
            raise ValueError("DINOv3PyramidBackbone requires a DINOv3 ViT")
        if len(output_channels) != len(self.strides):
            raise ValueError("Mask2Former requires four pyramid channel counts")

        hidden_layers = int(config.num_hidden_layers)
        layer_indices = tuple(
            hidden_layers * position // len(self.strides)
            for position in range(1, len(self.strides) + 1)
        )

        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.patch_size = int(config.patch_size)
        self.num_prefix_tokens = 1 + int(config.num_register_tokens)
        self.hidden_size = int(config.hidden_size)
        self.layer_indices = tuple(int(index) for index in layer_indices)
        self.channels = tuple(int(channels) for channels in output_channels)
        self.norms = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for _ in self.strides
        )
        self.projections = nn.ModuleList(
            PyramidProjection(self.hidden_size, channels)
            for channels in self.channels
        )

    def train(self, mode: bool = True) -> "DINOv3PyramidBackbone":
        super().train(mode)
        self.backbone.eval()
        return self

    def _backbone_context(self) -> Any:
        trainable = any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )
        return nullcontext() if trainable else torch.no_grad()

    def _spatial_tokens(
        self,
        hidden_state: torch.Tensor,
        *,
        batch_size: int,
        grid_size: tuple[int, int],
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        tokens = hidden_state[:, self.num_prefix_tokens :]
        expected = grid_size[0] * grid_size[1]
        if tokens.shape[1] != expected:
            raise ValueError(
                "DINOv3 patch-token count does not match the input dimensions: "
                f"expected {expected}, got {tokens.shape[1]}"
            )
        tokens = norm(tokens)
        return tokens.transpose(1, 2).reshape(
            batch_size,
            self.hidden_size,
            *grid_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        height, width = pixel_values.shape[-2:]
        if height % 32 or width % 32:
            raise ValueError("DINOv3 Mask2Former inputs must be divisible by 32")
        grid_size = (height // self.patch_size, width // self.patch_size)
        with self._backbone_context():
            outputs = self.backbone(
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
        if outputs.hidden_states is None:
            raise RuntimeError("DINOv3 did not return intermediate hidden states")

        feature_maps = []
        for layer_index, stride, norm, projection in zip(
            self.layer_indices,
            self.strides,
            self.norms,
            self.projections,
            strict=True,
        ):
            feature = self._spatial_tokens(
                outputs.hidden_states[layer_index],
                batch_size=pixel_values.shape[0],
                grid_size=grid_size,
                norm=norm,
            )
            feature = projection(feature)
            target_size = (height // stride, width // stride)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            feature_maps.append(feature)
        return SimpleNamespace(feature_maps=tuple(feature_maps))


class DINOv3Mask2FormerSegmenter(Mask2FormerSegmenter):
    """Mask2Former semantic model backed by a frozen DINOv3 feature pyramid."""

    def __init__(
        self,
        model: nn.Module,
        class_names: Sequence[str],
        ignore_index: int = 255,
    ) -> None:
        super().__init__(model, class_names, ignore_index)
        encoder = self.model.model.pixel_level_module.encoder
        if not isinstance(encoder, DINOv3PyramidBackbone):
            raise TypeError("Mask2Former encoder must be DINOv3PyramidBackbone")

    @property
    def pyramid_backbone(self) -> DINOv3PyramidBackbone:
        return self.model.model.pixel_level_module.encoder

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return (
            parameter
            for parameter in self.pyramid_backbone.backbone.parameters()
            if parameter.requires_grad
        )
