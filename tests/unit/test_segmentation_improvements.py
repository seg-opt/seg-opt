import random

import numpy as np
import pytest
import torch
from torch import nn

from src.data.transforms import EvalSegmentationTransform, TrainSegmentationTransform
from src.models.inference import sliding_window_logits
from src.models.segmentation import make_segmentation_head
from src.train.losses import SegmentationLoss
from src.train.trainer import TrainerModule
from src.utils.utils import load_config


def test_train_transform_keeps_image_mask_alignment_at_full_frame_size():
    random.seed(7)
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    mask = np.zeros((12, 18), dtype=np.int64)
    image[3:9, 10:16, 0] = 255
    mask[3:9, 10:16] = 3
    transform = TrainSegmentationTransform(
        flip_probability=0,
        brightness=0,
        contrast=0,
        gamma_min=1,
        gamma_max=1,
    )

    transformed_image, transformed_mask = transform(image, mask)

    assert transformed_image.shape == (512, 768, 3)
    assert transformed_mask.shape == (512, 768)
    assert np.count_nonzero(transformed_mask == 3) > 0
    assert transformed_image[..., 0][transformed_mask == 3].mean() > 200
    assert transformed_image[..., 0][transformed_mask == 0].mean() < 30


def test_eval_transform_is_deterministic_and_uses_full_frame_size():
    image = np.zeros((10, 15, 3), dtype=np.uint8)
    mask = np.zeros((10, 15), dtype=np.int64)

    transform = EvalSegmentationTransform()
    transformed_image, transformed_mask = transform(image, mask)
    repeated_image, repeated_mask = transform(image, mask)

    assert transformed_image.shape == (512, 768, 3)
    assert transformed_mask.shape == (512, 768)
    np.testing.assert_array_equal(transformed_image, repeated_image)
    np.testing.assert_array_equal(transformed_mask, repeated_mask)


def test_segmentation_loss_rewards_correct_rare_class_predictions_and_ignores_padding():
    targets = torch.tensor([[[0, 2], [3, 255]]])
    correct = torch.full((1, 4, 2, 2), -5.0)
    wrong = torch.zeros_like(correct)
    for row, column, class_id in ((0, 0, 0), (0, 1, 2), (1, 0, 3)):
        correct[0, class_id, row, column] = 5
    correct[0, :, 1, 1] = torch.tensor([100.0, -100.0, -100.0, -100.0])
    loss = SegmentationLoss(
        num_classes=4,
        class_weights=(0.25, 0.5, 1, 2),
        dice_weight=0.5,
    )

    correct_total = loss(correct, targets)
    changed_padding = correct.clone()
    changed_padding[:, :, 1, 1] = torch.tensor([-100.0, 100.0, 50.0, -50.0])

    assert correct_total < loss(wrong, targets)
    torch.testing.assert_close(correct_total, loss(changed_padding, targets))
    assert torch.isfinite(correct_total)


def test_trainer_logs_losses_and_segmentation_metrics_for_wandb():
    module = TrainerModule(
        teacher=None,
        student=nn.Conv2d(3, 4, kernel_size=1),
        task_loss=SegmentationLoss(4, dice_weight=0.5),
        distillation_loss=None,
        distillation_weight=0,
        class_names=["background", "sky", "small_rock", "large_rock"],
    )
    logged = {}

    def record(name, value, **_):
        logged[name] = value

    def record_dict(metrics, **_):
        logged.update(metrics)

    module.log = record
    module.log_dict = record_dict
    batch = {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "labels": torch.randint(0, 4, (2, 8, 8)),
    }

    module.on_train_epoch_start()
    module.training_step(batch, batch_idx=0)
    module.on_train_epoch_end()
    module.on_validation_epoch_start()
    module.validation_step(batch, batch_idx=0)
    module.on_validation_epoch_end()

    expected = {
        "train/loss",
        "train/task_loss",
        "train/kd_loss",
        "train/miou",
        "train/macro_f1",
        "train/pixel_accuracy",
        "train/iou_small_rock",
        "train/precision_large_rock",
        "train/recall_large_rock",
        "val/loss",
        "val/miou",
        "val/pixel_accuracy",
    }
    assert expected.issubset(logged)


def test_sliding_window_matches_pointwise_full_image_inference():
    model = nn.Conv2d(3, 4, kernel_size=1)
    pixels = torch.randn(2, 3, 9, 13)

    expected = model(pixels)
    actual = sliding_window_logits(model, pixels, crop_size=6, stride=4)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("head_type", ["linear", "lightweight"])
def test_configurable_segmentation_heads_preserve_spatial_shape(head_type):
    head = make_segmentation_head(
        in_channels=12,
        num_classes=4,
        head_type=head_type,
        hidden_channels=18,
    )

    assert head(torch.randn(2, 12, 8, 9)).shape == (2, 4, 8, 9)


def test_canonical_dinov3_config_is_a_full_frame_frozen_linear_probe():
    config = load_config("experiments/baselines/dinov3_vitl16.yaml")

    assert config.model.name == "dinov3_vitl16"
    assert config.model.model_id == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert config.model.freeze_backbone
    assert (config.data.image_height, config.data.image_width) == (512, 768)
    assert config.loss.dice_weight == pytest.approx(0.5)
    assert config.training.batch_size * config.training.accumulate_grad_batches == 16
