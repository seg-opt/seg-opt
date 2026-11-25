import pytest
import torch

from src.models.segmentation import make_segmentation_head
from src.train.losses import SegmentationLoss
from tests.unit.test_submit_hydra import load_hydra_config


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


@pytest.mark.parametrize("dice_eps", [None, 0.25])
def test_segmentation_loss_uses_dice_eps(dice_eps):
    kwargs = {} if dice_eps is None else {"dice_eps": dice_eps}
    loss = SegmentationLoss(2, cross_entropy_weight=0, dice_weight=1, **kwargs)
    eps = 1e-8 if dice_eps is None else dice_eps

    actual = loss(torch.zeros(1, 2, 1, 1), torch.zeros(1, 1, 1, dtype=torch.long))

    assert loss.dice_eps == eps
    torch.testing.assert_close(actual, torch.tensor(1 - 0.5 / (1.5 + eps)))


@pytest.mark.parametrize("head_type", ["linear", "lightweight"])
def test_configurable_segmentation_heads_preserve_spatial_shape(head_type):
    head = make_segmentation_head(
        in_channels=12,
        num_classes=4,
        head_type=head_type,
        hidden_channels=18,
    )

    assert head(torch.randn(2, 12, 8, 9)).shape == (2, 4, 8, 9)


def test_benchmark_dinov3_config_is_a_full_frame_frozen_linear_probe(tmp_path):
    config = load_hydra_config(
        tmp_path,
        'dataset=lunar',
        'method=supervised',
        'teacher=none',
        'student=dinov3_vitl16',
        'training.output_dir=results/baselines_benchmark/dinov3_vitl16/seed42',
        'training.batch_size=2',
        'training.accumulate_grad_batches=8',
        'training.learning_rate=0.001',
        'training.backbone_learning_rate=null',
    )

    assert config.model.name == "dinov3_vitl16"
    assert config.model.model_id == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert config.model.freeze_backbone
    assert (config.data.image_height, config.data.image_width) == (512, 768)
    assert config.loss.dice_weight == pytest.approx(0.5)
    assert config.training.batch_size * config.training.accumulate_grad_batches == 16
