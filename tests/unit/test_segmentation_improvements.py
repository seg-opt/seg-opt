import pytest
import torch

from src.train.losses import SegmentationLoss
from src.utils.utils import load_config


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


def test_segmentation_loss_uses_weighted_macro_dice():
    logits = torch.log(
        torch.tensor([[[[0.8, 0.3, 0.99]], [[0.2, 0.7, 0.01]]]])
    )
    targets = torch.tensor([[[0, 1, 255]]])
    loss = SegmentationLoss(
        num_classes=2,
        class_weights=(0.25, 0.75),
        cross_entropy_weight=0,
        dice_weight=1,
    )

    actual = loss(logits, targets)

    expected = 0.25 * (1 - 1.6 / 2.1) + 0.75 * (1 - 1.4 / 1.9)
    torch.testing.assert_close(actual, torch.tensor(expected))

    changed_ignored_logits = logits.clone()
    changed_ignored_logits[:, :, :, 2] = torch.tensor([[[-100.0], [100.0]]])
    torch.testing.assert_close(actual, loss(changed_ignored_logits, targets))


def test_benchmark_dinov3_config_is_a_full_frame_frozen_linear_probe():
    config = load_config("experiments/baselines_benchmark/dinov3_vitl16.yaml")

    assert config.model.name == "dinov3_vitl16"
    assert config.model.model_id == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert config.model.freeze_backbone
    assert (config.data.image_height, config.data.image_width) == (512, 768)
    assert config.loss.dice_weight == pytest.approx(0.5)
    assert config.training.batch_size * config.training.accumulate_grad_batches == 16
