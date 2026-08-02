import pytest
import torch
from torch import nn

from src.models import load_dinov3, load_sam3, load_student
from src.models.segmentation import (
    BackboneSegmenter,
    QuerySegmenter,
)


class QueryModel(nn.Module):
    pass


def test_backbone_segmenter_returns_full_resolution_logits():
    model = BackboneSegmenter(
        backbone=nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
        in_channels=8,
        num_classes=4,
    )

    logits = model(torch.randn(2, 3, 16, 16))

    assert logits.shape == (2, 4, 16, 16)


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


def test_alternative_teacher_loaders_are_placeholders():
    with pytest.raises(NotImplementedError):
        load_dinov3("model-id", num_classes=4)

    with pytest.raises(NotImplementedError):
        load_sam3("model-id", ["background", "rock"])
