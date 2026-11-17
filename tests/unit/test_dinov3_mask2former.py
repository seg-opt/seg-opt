from types import SimpleNamespace

import torch
from torch import nn

from src.models.dinov3_mask2former import (
    DINOv3Mask2FormerSegmenter,
    DINOv3PyramidBackbone,
)
from src.utils.utils import load_config


CLASS_NAMES = ("background", "sky", "small_rock", "large_rock")


class TinyDINOv3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="dinov3_vit",
            hidden_size=8,
            patch_size=4,
            num_register_tokens=2,
            num_hidden_layers=4,
        )
        self.projection = nn.Conv2d(3, 8, kernel_size=4, stride=4)

    def forward(self, pixel_values, output_hidden_states=False):
        assert output_hidden_states
        patches = self.projection(pixel_values).flatten(2).transpose(1, 2)
        prefix = patches.new_zeros(patches.shape[0], 3, patches.shape[2])
        tokens = torch.cat([prefix, patches], dim=1)
        hidden_states = tuple(tokens + index for index in range(5))
        return SimpleNamespace(hidden_states=hidden_states)


class TinyUniversalSegmenter(nn.Module):
    def __init__(self, encoder: DINOv3PyramidBackbone) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.pixel_level_module = nn.Module()
        self.model.pixel_level_module.encoder = encoder
        self.class_queries = nn.Parameter(torch.randn(3, 5))
        self.mask_projection = nn.Conv2d(4, 3, kernel_size=1)

    def forward(self, pixel_values, output_hidden_states=False, **_):
        feature_maps = self.model.pixel_level_module.encoder(pixel_values).feature_maps
        batch_size = pixel_values.shape[0]
        masks = self.mask_projection(feature_maps[0])
        return SimpleNamespace(
            loss=None,
            class_queries_logits=self.class_queries.unsqueeze(0).expand(
                batch_size, -1, -1
            ),
            masks_queries_logits=masks,
            encoder_hidden_states=(feature_maps if output_hidden_states else None),
        )


def test_dinov3_pyramid_returns_mask2former_feature_shapes() -> None:
    pyramid = DINOv3PyramidBackbone(
        TinyDINOv3(),
        output_channels=(4, 8, 12, 16),
    )

    features = pyramid(torch.randn(2, 3, 32, 64)).feature_maps

    assert [feature.shape for feature in features] == [
        (2, 4, 8, 16),
        (2, 8, 4, 8),
        (2, 12, 2, 4),
        (2, 16, 1, 2),
    ]
    assert not any(parameter.requires_grad for parameter in pyramid.backbone.parameters())
    assert all(parameter.requires_grad for parameter in pyramid.projections.parameters())


def test_dinov3_pyramid_keeps_frozen_backbone_in_eval_mode() -> None:
    pyramid = DINOv3PyramidBackbone(TinyDINOv3(), (4, 8, 12, 16))

    pyramid.train()

    assert not pyramid.backbone.training


def test_dinov3_mask2former_returns_scores_and_cap_feature() -> None:
    pyramid = DINOv3PyramidBackbone(TinyDINOv3(), (4, 8, 12, 16))
    model = DINOv3Mask2FormerSegmenter(
        TinyUniversalSegmenter(pyramid),
        CLASS_NAMES,
    )
    images = torch.randn(2, 3, 32, 64)

    scores, features = model.scores_and_features(images)

    assert scores.shape == (2, 4, 32, 64)
    assert features.shape == (2, 16, 1, 2)
    assert model.feature_channels_at(-1) == 16
    assert model.feature_stride_at(-1) == 32


def test_lunar_teacher_config_uses_frozen_dinov3_mask2former() -> None:
    config = load_config(
        "experiments/dinov3_mask2former/lunar_teacher_80e.yaml"
    )

    assert config.model.name == "dinov3_vitl16_mask2former"
    assert config.model.freeze_backbone
    assert config.model.decoder_model_id == (
        "facebook/mask2former-swin-large-ade-semantic"
    )
    assert config.training.epochs == 80
    assert config.training.batch_size == 1
    assert config.training.accumulate_grad_batches == 16
