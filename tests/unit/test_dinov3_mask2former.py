from types import SimpleNamespace

import torch
from torch import nn

from src.models.dinov3_mask2former import DINOv3PyramidBackbone
from src.utils.utils import load_config


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
