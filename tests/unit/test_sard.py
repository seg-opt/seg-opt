from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.distillation.sard import (
    SARDLoss,
    capture_features,
    structure_cues,
    structure_weights,
)
from src.models.fast_scnn import FastSCNN
from src.models.segmentation import FeatureTap
from src.train.losses import SegmentationLoss
from src.train.trainer import BaselineModule, DistillationModule, create_trainer
from src.utils.utils import load_config

CLASS_NAMES = ["background", "sky", "small_rock", "large_rock"]


def _step_features() -> torch.Tensor:
    """A vertical step edge in every channel: left half 0, right half 1."""
    features = torch.zeros(1, 4, 16, 16)
    features[..., 8:] = 1.0
    return features


def _corner_features() -> torch.Tensor:
    """A bright square whose corners mix horizontal and vertical gradients."""
    features = torch.zeros(1, 4, 16, 16)
    features[..., 4:12, 4:12] = 1.0
    return features


def test_edge_cue_peaks_on_the_step_and_vanishes_on_flat_regions():
    cues = structure_cues(_step_features(), density_radius=1)

    assert cues.edge[0, :, 7:9].min() > 0
    assert cues.edge[0, :, :6].abs().max() == 0
    assert cues.edge[0, :, 10:].abs().max() == 0
    # A straight edge has a single gradient direction, so no corner response.
    assert cues.corner.abs().max() < 1e-6


def test_corner_cue_fires_at_square_corners_not_along_its_sides():
    cues = structure_cues(_corner_features(), density_radius=1)

    corner = cues.corner[0, 3:5, 3:5].max()
    side = cues.corner[0, 7:9, 3:5].max()
    assert corner > 0
    assert corner > 10 * side


def test_density_cue_measures_local_feature_spread():
    cues = structure_cues(_step_features(), density_radius=1)

    assert cues.density[0, :, 7:9].min() > 0
    assert cues.density[0, :, :6].abs().max() < 1e-6


def test_structure_weights_are_normalized_and_exclude_ignored_cells():
    valid = torch.ones(1, 16, 16)
    valid[:, :2] = 0

    weights = structure_weights(_step_features(), valid, density_radius=1)

    assert torch.allclose(weights.sum(dim=(-2, -1)), torch.ones(1))
    assert weights[:, :2].abs().max() == 0
    assert weights[0, 4, 7] > 10 * weights[0, 4, 0]


def test_zero_structure_weights_give_the_uniform_ablation():
    valid = torch.ones(2, 6, 6)
    valid[1, 0] = 0

    weights = structure_weights(
        torch.randn(2, 3, 6, 6),
        valid,
        edge_weight=0.0,
        corner_weight=0.0,
        density_weight=0.0,
    )

    assert torch.allclose(weights[0], torch.full((6, 6), 1 / 36))
    assert torch.allclose(weights[1][valid[1].bool()], torch.full((30,), 1 / 30))


def test_sard_loss_trains_student_and_heads_but_not_the_teacher_projection():
    loss = SARDLoss(
        student_channels=5,
        teacher_channels=7,
        projection_channels=8,
        denoiser_hidden_channels=6,
        spatial_stride=2,
    )
    student = torch.randn(2, 5, 4, 4, requires_grad=True)
    teacher = torch.randn(2, 7, 8, 8, requires_grad=True)
    labels = torch.randint(0, 4, (2, 16, 16))
    labels[0, :4] = 255

    result = loss(student, teacher, labels)
    result.total.backward()

    assert result.total.ndim == 0 and torch.isfinite(result.total)
    assert set(result.components) == {
        "raw_feature",
        "raw_distribution",
        "weighted_feature",
        "weighted_distribution",
    }
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None
    assert loss.student_projection.weight.grad is not None
    assert loss.denoiser[0].weight.grad is not None
    assert loss.teacher_projection.weight.grad is None
    assert not loss.teacher_projection.weight.requires_grad


def test_distribution_term_is_learnable_from_matching_student_features():
    torch.manual_seed(0)
    loss = SARDLoss(
        student_channels=4,
        teacher_channels=4,
        projection_channels=4,
        denoiser_hidden_channels=16,
        feature_weight=0.0,
        distribution_weight=1.0,
        spatial_stride=1,
    )
    optimizer = torch.optim.Adam(loss.denoiser.parameters(), lr=1e-2)
    labels = torch.zeros(4, 8, 8, dtype=torch.long)
    first = last = None
    for _ in range(300):
        features = torch.randn(4, 4, 8, 8)
        # A student equal to the teacher lets the head recover the noise.
        with torch.no_grad():
            loss.student_projection.weight.copy_(loss.teacher_projection.weight)
            loss.student_projection.bias.zero_()
        value = loss(features, features, labels).components["raw_distribution"]
        optimizer.zero_grad()
        value.backward()
        optimizer.step()
        first = value.item() if first is None else first
        last = value.item()

    # Predicting zero would score ~1 (unit-variance noise).
    assert first > 0.8
    assert last < 0.3


class TinyFeatureModel(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(channels, 4, kernel_size=1)
        self.forward_calls = 0
        self.channels = channels

    def forward(self, inputs):
        self.forward_calls += 1
        return self.classifier(torch.relu(self.body(inputs)))

    def loss_and_scores(self, inputs, labels, criterion):
        scores = self(inputs)
        return criterion(scores, labels), scores

    def encoder_parameters(self):
        return iter(())

    def distillation_feature_tap(self):
        return FeatureTap(self.classifier, self.channels, use_input=True)


def _batch():
    return {
        "pixel_values": torch.randn(2, 3, 16, 16),
        "labels": torch.randint(0, 4, (2, 16, 16)),
    }


def _sard_module(student, teacher, *, total_steps=4):
    return DistillationModule(
        student,
        SegmentationLoss(4),
        teacher=teacher,
        distillation_loss=SARDLoss(
            student_channels=student.channels,
            teacher_channels=teacher.channels,
            projection_channels=8,
            denoiser_hidden_channels=8,
            spatial_stride=2,
        ),
        head_learning_rate=1e-3,
        class_names=CLASS_NAMES,
        total_steps=total_steps,
    )


def test_capture_features_records_classifier_input_and_removes_its_hook():
    model = TinyFeatureModel(6)
    with capture_features(model) as captured:
        model(torch.randn(1, 3, 8, 8))

    assert captured.value.shape == (1, 6, 8, 8)
    assert not model.classifier._forward_hooks


def test_sard_training_step_updates_student_and_heads_only():
    teacher = TinyFeatureModel(10)
    student = TinyFeatureModel(6)
    module = _sard_module(student, teacher)
    module.log = lambda *args, **kwargs: None
    module.train()

    loss = module.training_step(_batch(), batch_idx=0)
    loss.backward()

    assert student.body.weight.grad is not None
    assert teacher.body.weight.grad is None
    assert module.distillation_loss.student_projection.weight.grad is not None
    assert teacher.forward_calls == 1
    assert not teacher.training

    groups = module._optimizer_groups()
    distillation = [group for group in groups if group["name"] == "distillation"]
    assert len(distillation) == 1
    names = {id(p) for p in distillation[0]["params"]}
    assert id(module.distillation_loss.student_projection.weight) in names
    assert id(module.distillation_loss.teacher_projection.weight) not in names


def test_sard_checkpoint_contains_only_student_weights(tmp_path):
    module = _sard_module(TinyFeatureModel(6), TinyFeatureModel(10), total_steps=1)
    trainer = create_trainer(
        tmp_path,
        max_epochs=1,
        precision="32-true",
        devices=1,
        fast_dev_run=True,
        enable_wandb=False,
    )
    loader = DataLoader([_batch()], batch_size=None)

    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    state = torch.load(
        tmp_path / "checkpoints" / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )["state_dict"]
    assert not any(key.startswith(("teacher.", "distillation_loss.")) for key in state)
    baseline = BaselineModule(
        TinyFeatureModel(6),
        SegmentationLoss(4),
        head_learning_rate=1e-3,
        class_names=CLASS_NAMES,
        total_steps=1,
    )
    baseline.load_state_dict(state, strict=True)


def test_fast_scnn_tap_matches_the_declared_channels_at_stride_8():
    model = FastSCNN(4, class_names=CLASS_NAMES).eval()
    tap = model.distillation_feature_tap()
    with torch.no_grad(), capture_features(model) as captured:
        model(torch.randn(1, 3, 64, 96))

    assert captured.value.shape == (1, tap.channels, 8, 12)


@pytest.mark.parametrize("variant", ["full", "uniform"])
def test_checked_in_sard_configs(variant):
    config = load_config(
        f"experiments/sard/fast_scnn_sard_{variant}_from_segformer_b4.yaml"
    )

    assert config.model.name == "fast_scnn"
    assert config.data.mask_variant == "clean"
    assert config.training.batch_size * config.training.accumulate_grad_batches == 16
    assert Path(config.training.output_dir).parts[-4:] == (
        "sard",
        "fast_scnn_from_segformer_b4",
        variant,
        "seed42",
    )
    distillation = config.distillation
    assert distillation is not None and distillation.method == "sard"
    assert distillation.sard_feature_weight == 1.0
    assert distillation.sard_distribution_weight == 0.5
    assert distillation.sard_noise_alpha == 0.5
    assert distillation.sard_density_radius == 7
    betas = (
        distillation.sard_edge_weight,
        distillation.sard_corner_weight,
        distillation.sard_density_weight,
    )
    assert betas == ((2.0, 1.0, 1.0) if variant == "full" else (0.0, 0.0, 0.0))
    assert Path(distillation.teacher.config_path).parts[-2:] == (
        "baselines_benchmark",
        "segformer_b4.yaml",
    )
