from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.models import DINOv3Segmenter


class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="dinov3_vit",
            hidden_size=8,
            patch_size=4,
            num_register_tokens=2,
        )
        self.projection = nn.Conv2d(3, 8, kernel_size=4, stride=4)

    def forward(self, pixel_values):
        patches = self.projection(pixel_values).flatten(2).transpose(1, 2)
        prefix = patches.new_zeros(patches.shape[0], 3, patches.shape[2])
        return SimpleNamespace(last_hidden_state=torch.cat([prefix, patches], dim=1))


class ConvNextBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="dinov3_convnext",
            hidden_sizes=[4, 8],
        )
        self.projection = nn.Conv2d(3, 8, kernel_size=4, stride=4)

    def forward(self, pixel_values, output_hidden_states):
        assert output_hidden_states
        features = self.projection(pixel_values)
        return SimpleNamespace(hidden_states=(features,))


@pytest.mark.parametrize("backbone", [ViTBackbone(), ConvNextBackbone()])
def test_dinov3_segmenter_returns_full_resolution_logits(backbone):
    model = DINOv3Segmenter(backbone, num_classes=4, freeze_backbone=False)

    logits = model(torch.randn(2, 3, 16, 20))

    assert logits.shape == (2, 4, 16, 20)


def test_dinov3_segmenter_keeps_frozen_backbone_in_eval_mode():
    model = DINOv3Segmenter(ViTBackbone(), num_classes=4, freeze_backbone=True)

    model.train()

    assert not model.backbone.training
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
