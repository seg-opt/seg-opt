from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.models import DINOv3Segmenter
from src.models.loaders import load_model


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


def test_dinov3_segmenter_returns_full_resolution_logits():
    model = DINOv3Segmenter(ViTBackbone(), num_classes=4, freeze_backbone=False)

    logits = model(torch.randn(2, 3, 16, 20))

    assert logits.shape == (2, 4, 16, 20)


def test_dinov3_segmenter_keeps_frozen_backbone_in_eval_mode():
    model = DINOv3Segmenter(ViTBackbone(), num_classes=4, freeze_backbone=True)

    model.train()

    assert not model.backbone.training
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


@pytest.mark.parametrize(
    "model_name, message",
    [("fast_scnn", "no pretrained weights"), ("dinov3", "unsupported DINOv3 backbone")],
)
def test_model_loader_errors(model_name, message, monkeypatch):
    backbone = ViTBackbone()
    monkeypatch.setattr(backbone.config, "model_type", "unsupported")
    with pytest.raises(ValueError, match=message):
        if model_name == "fast_scnn":
            load_model(model_name, num_classes=4, pretrained=True)
        else:
            DINOv3Segmenter(backbone, num_classes=4)
