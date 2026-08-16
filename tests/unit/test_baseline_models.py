from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torchvision import models

from src.models.fast_scnn import DepthwiseSeparableConv
from src.models.loaders import BASELINE_MODEL_NAMES, load_model
from src.models.resnet_unet import ResNet34UNet
from src.models.segmentation import (
    Mask2FormerSegmenter,
    SegFormerSegmenter,
    SemanticSegmenter,
)


CLASS_NAMES = ("background", "sky", "small_rock", "large_rock")


class ScalarCriterion(nn.Module):
    def forward(self, scores, labels):
        return nn.functional.cross_entropy(scores, labels)


class TinyDenseSegmenter(SemanticSegmenter):
    def __init__(self):
        super().__init__(CLASS_NAMES)
        self.classifier = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, images):
        return self.classifier(images)


class TinySegFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.segformer = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.decode_head = nn.Conv2d(8, 4, kernel_size=1)

    def forward(self, pixel_values):
        return SimpleNamespace(logits=self.decode_head(self.segformer(pixel_values)))


class TinyMask2Former(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.pixel_level_module = nn.Module()
        self.model.pixel_level_module.encoder = nn.Conv2d(3, 4, kernel_size=1)
        self.class_queries = nn.Parameter(torch.randn(3, 5))
        self.mask_queries = nn.Conv2d(3, 3, kernel_size=1)
        self.received_targets = None

    def forward(
        self,
        pixel_values,
        mask_labels=None,
        class_labels=None,
    ):
        batch_size = pixel_values.shape[0]
        class_queries_logits = self.class_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        masks_queries_logits = self.mask_queries(pixel_values)
        loss = None
        if mask_labels is not None:
            self.received_targets = mask_labels, class_labels
            loss = class_queries_logits.square().mean()
            loss = loss + masks_queries_logits.square().mean()
        return SimpleNamespace(
            loss=loss,
            class_queries_logits=class_queries_logits,
            masks_queries_logits=masks_queries_logits,
        )


def test_baseline_dispatch_is_limited_to_the_five_reviewed_models():
    assert BASELINE_MODEL_NAMES == (
        "fast_scnn",
        "resnet34_unet",
        "segformer_b0",
        "dinov3_vitl16",
        "mask2former_swinl",
    )
    with pytest.raises(ValueError, match="unsupported baseline"):
        load_model("mobilenet", class_names=CLASS_NAMES)


def test_fast_scnn_uses_activated_depthwise_and_pointwise_convolutions():
    block = DepthwiseSeparableConv(8, 12)

    assert isinstance(block.depthwise[-1], nn.ReLU)
    assert isinstance(block.pointwise[-1], nn.ReLU)


def test_dense_baseline_contract_returns_loss_and_scores():
    model = TinyDenseSegmenter()
    images = torch.randn(2, 3, 8, 12)
    labels = torch.randint(0, 4, (2, 8, 12))

    loss, scores = model.loss_and_scores(images, labels, ScalarCriterion())
    loss.backward()

    assert scores.shape == (2, 4, 8, 12)
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert list(model.encoder_parameters()) == []


def test_resnet34_unet_has_real_skips_and_full_resolution_output():
    model = ResNet34UNet(models.resnet34(weights=None), CLASS_NAMES)
    images = torch.randn(1, 3, 32, 32)
    model.eval()

    with torch.no_grad():
        scores = model(images)

    assert scores.shape == (1, 4, 32, 32)
    assert model.decoder4.convolutions[0].in_channels == 512 + 256
    assert model.decoder3.convolutions[0].in_channels == 256 + 128
    assert model.decoder2.convolutions[0].in_channels == 128 + 64
    assert model.decoder1.convolutions[0].in_channels == 64 + 64


def test_segformer_wrapper_interpolates_official_decoder_scores():
    model = SegFormerSegmenter(TinySegFormer(), CLASS_NAMES)
    images = torch.randn(2, 3, 31, 47)

    scores = model(images)

    assert scores.shape == (2, 4, 31, 47)
    assert set(model.encoder_parameters()) == set(model.model.segformer.parameters())


def test_mask2former_converts_dense_targets_and_uses_native_loss_once():
    core = TinyMask2Former()
    model = Mask2FormerSegmenter(core, CLASS_NAMES)
    images = torch.randn(2, 3, 8, 12)
    labels = torch.zeros(2, 8, 12, dtype=torch.long)
    labels[0, :, 6:] = 3
    labels[0, 0, 0] = 255
    labels[1, 2:4, 3:7] = 2

    loss, scores = model.loss_and_scores(images, labels, ScalarCriterion())
    mask_labels, class_labels = core.received_targets
    loss.backward()

    assert scores.shape == (2, 4, 8, 12)
    assert class_labels[0].tolist() == [0, 3]
    assert class_labels[1].tolist() == [0, 2]
    assert mask_labels[0].shape == (2, 8, 12)
    assert not mask_labels[0][:, 0, 0].any()
    assert core.class_queries.grad is not None


def test_mask2former_rejects_an_all_ignore_training_target():
    model = Mask2FormerSegmenter(TinyMask2Former(), CLASS_NAMES)
    labels = torch.full((1, 8, 12), 255)

    with pytest.raises(ValueError, match="all-ignore"):
        model.dense_targets(labels)
