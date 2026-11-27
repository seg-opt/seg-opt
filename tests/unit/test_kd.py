import pytest
import torch
from torch.nn import functional as F

from src.distillation.kd import (
    PixelWiseKDLoss,
    boundary_masks,
    trimap_mask,
)


def _labels() -> torch.Tensor:
    labels = torch.zeros(1, 5, 5, dtype=torch.long)
    labels[:, 2, 2] = 1
    return labels


def test_boundary_masks_are_class_aware_soft_and_border_safe():
    masks = boundary_masks(
        _labels(),
        num_classes=2,
        output_size=(5, 5),
        edge_kernel_size=3,
    )

    assert masks.edge.shape == (1, 2, 5, 5)
    assert torch.all((0 <= masks.edge) & (masks.edge <= 1))
    # Both classes participate in the 3x3 semantic transition around the
    # isolated pixel, rather than creating an artificial image-frame edge.
    assert masks.edge[0, 0].sum().item() == 9
    assert masks.edge[0, 1].sum().item() == 9

    uniform = boundary_masks(
        torch.zeros(1, 5, 5, dtype=torch.long),
        num_classes=2,
        output_size=(5, 5),
        edge_kernel_size=3,
    )
    assert uniform.edge.sum().item() == 0
    assert trimap_mask(
        torch.zeros(1, 5, 5, dtype=torch.long),
        num_classes=2,
        edge_kernel_size=3,
    ).sum().item() == 0


def test_boundary_masks_average_pool_and_exclude_ignore_neighbourhoods():
    labels = _labels()
    labels[:, 0, 0] = 255
    masks = boundary_masks(
        labels,
        num_classes=2,
        output_size=(2, 2),
        edge_kernel_size=3,
    )

    assert masks.edge.shape == (1, 2, 2, 2)
    assert masks.valid[0, 0, 0, 0].item() == 0
    assert masks.edge[:, :, 0, 0].sum().item() == 0
    assert torch.all((0 <= masks.edge) & (masks.edge <= 1))


def test_boundary_masks_reject_invalid_kernel_and_labels():
    with pytest.raises(ValueError, match="odd"):
        boundary_masks(
            _labels(),
            num_classes=2,
            output_size=(5, 5),
            edge_kernel_size=2,
        )
    with pytest.raises(ValueError, match="outside"):
        boundary_masks(
            torch.full((1, 5, 5), 2),
            num_classes=2,
            output_size=(5, 5),
            edge_kernel_size=3,
        )


@pytest.mark.parametrize("weight", [0, 1])
def test_pixel_kd_enabled_tracks_weight(weight):
    loss = PixelWiseKDLoss(num_classes=2, weight=weight)

    assert loss.enabled is bool(weight)
    loss.weight = 1 - weight
    assert loss.enabled is bool(1 - weight)


def test_zero_weight_kd_returns_a_differentiable_zero_without_resize_work():
    student = torch.randn(1, 2, 5, 5, requires_grad=True)
    teacher = torch.randn(1, 2, 5, 5)
    labels = _labels()

    result = PixelWiseKDLoss(num_classes=2, weight=0)(student, teacher, labels)
    result.total.backward()

    assert result.total.item() == 0
    assert student.grad is not None


def test_pixel_kd_ignores_void_pixels_and_uses_the_same_result_contract():
    student = torch.zeros(1, 2, 3, 3, requires_grad=True)
    teacher = student.detach().clone()
    student.data[:, :, 0, 0] = torch.tensor([5.0, -5.0]).view(1, 2)
    teacher[:, :, 0, 0] = torch.tensor([-5.0, 5.0]).view(1, 2)
    labels = torch.zeros(1, 3, 3, dtype=torch.long)
    labels[:, 0, 0] = 255

    result = PixelWiseKDLoss(num_classes=2, spatial_stride=1)(
        student,
        teacher,
        labels,
    )
    result.total.backward()

    assert result.total.item() == pytest.approx(0.0, abs=1e-6)
    assert set(result.components) == {"raw_vanilla", "weighted_vanilla"}
    assert result.components["raw_vanilla"].item() == pytest.approx(0.0, abs=1e-6)
    assert student.grad is not None


def test_pixel_kd_matches_valid_pixel_reduction_and_weight():
    student = torch.tensor(
        [[[[0.0, 1.0], [0.5, -0.2]], [[0.5, -0.5], [0.4, 0.1]]]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[[[0.1, -0.4], [0.2, 0.9]], [[-0.2, 0.6], [0.7, -0.1]]]]
    )
    labels = torch.tensor([[[0, 255], [1, 0]]])
    result = PixelWiseKDLoss(
        num_classes=2,
        weight=1.5,
        spatial_stride=1,
    )(student, teacher, labels)

    teacher_log_probabilities = F.log_softmax(teacher, dim=1)
    pixel_kl = (
        teacher_log_probabilities.exp()
        * (teacher_log_probabilities - F.log_softmax(student, dim=1))
    ).sum(dim=1)
    valid = labels != 255
    expected_raw = pixel_kl[valid].mean()

    torch.testing.assert_close(result.components["raw_vanilla"], expected_raw)
    torch.testing.assert_close(result.total, expected_raw * 1.5)
