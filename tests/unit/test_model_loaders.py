from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.models import (
    DINOv3SegmentationProcessor,
    DINOv3Segmenter,
    FastSCNN,
    load_dinov3,
    load_sam3,
    load_student,
)
from src.models.segmentation import (
    BackboneSegmenter,
    QuerySegmenter,
)


class QueryModel(nn.Module):
    pass


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


class ImageProcessor:
    def __call__(self, images, return_tensors, **kwargs):
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros(len(images), 3, 8, 12)}


def test_backbone_segmenter_returns_full_resolution_logits():
    model = BackboneSegmenter(
        backbone=nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
        in_channels=8,
        num_classes=4,
    )

    logits = model(torch.randn(2, 3, 16, 16))

    assert logits.shape == (2, 4, 16, 16)


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


def test_dinov3_processor_resizes_class_maps_with_nearest_neighbor():
    processor = DINOv3SegmentationProcessor(ImageProcessor())
    class_map = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.int64)

    encoded = processor(
        images=[np.zeros((2, 3, 3)), np.zeros((2, 3, 3))],
        segmentation_maps=[class_map, class_map],
        return_tensors="pt",
    )

    assert encoded["labels"].shape == (2, 8, 12)
    assert encoded["labels"].dtype == torch.long
    assert set(encoded["labels"].unique().tolist()) == {0, 1, 2, 3}


def test_dinov3_loader_builds_model_and_processor(monkeypatch):
    backbone = ViTBackbone()
    image_processor = ImageProcessor()
    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained",
        lambda model_id: backbone,
    )
    monkeypatch.setattr(
        "transformers.AutoImageProcessor.from_pretrained",
        lambda model_id: image_processor,
    )

    model, processor = load_dinov3("facebook/dinov3-test", num_classes=4)

    assert isinstance(model, DINOv3Segmenter)
    assert isinstance(processor, DINOv3SegmentationProcessor)
    assert processor.image_processor is image_processor


def test_query_segmenter_is_an_explicit_placeholder():
    model = QuerySegmenter(QueryModel())

    with pytest.raises(NotImplementedError, match="query-based"):
        model(torch.randn(2, 3, 16, 16))


def test_student_loader_builds_mobilenet_segmenter():
    model = load_student(
        "mobilenet_v3_small",
        num_classes=4,
        pretrained=False,
    )
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(1, 3, 32, 32))

    assert logits.shape == (1, 4, 32, 32)


def test_student_loader_builds_fast_scnn_with_full_resolution_logits():
    model = load_student("fast_scnn", num_classes=4, pretrained=False)
    inputs = torch.randn(2, 3, 64, 96)
    targets = torch.randint(0, 4, (2, 64, 96))

    logits = model(inputs)
    loss = nn.functional.cross_entropy(logits, targets)
    loss.backward()

    assert isinstance(model, FastSCNN)
    assert logits.shape == (2, 4, 64, 96)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_fast_scnn_rejects_unavailable_pretrained_weights():
    with pytest.raises(ValueError, match="pretrained=false"):
        load_student("fast_scnn", num_classes=4, pretrained=True)


def test_alternative_teacher_loaders_are_placeholders():
    with pytest.raises(NotImplementedError):
        load_sam3("model-id", ["background", "rock"])
