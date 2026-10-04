import pytest
import torch
from torch.nn import functional as F

from src.distillation.bpkd import (
    BPKDLoss,
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


def test_bpkd_identical_logits_are_zero_and_student_remains_differentiable():
    logits = torch.randn(2, 3, 8, 12, requires_grad=True)
    labels = torch.randint(0, 3, (2, 8, 12))
    result = BPKDLoss(num_classes=3, spatial_stride=1)(logits, logits, labels)

    result.total.backward()

    assert result.total.item() == pytest.approx(0.0, abs=1e-6)
    assert logits.grad is not None
    assert result.components["weighted_body"].item() == pytest.approx(0.0, abs=1e-6)
    assert result.components["weighted_edge"].item() == pytest.approx(0.0, abs=1e-6)


def test_bpkd_pom_matches_the_paper_reduction():
    student = torch.tensor(
        [[[[0.2, 1.1, -0.3], [0.0, 0.8, -0.2], [0.4, -0.5, 0.7]],
          [[-0.6, 0.3, 0.9], [0.4, -0.7, 0.1], [0.2, 0.5, -0.4]]]],
        requires_grad=True,
    )
    teacher = student.detach() + torch.tensor([0.0, 0.25]).view(1, 2, 1, 1)
    labels = _labels()[:, 1:4, 1:4]
    loss = BPKDLoss(
        num_classes=2,
        body_weight=0,
        edge_weight=1,
        edge_alpha=[2, 3],
        edge_kernel_size=3,
        spatial_stride=1,
        prm=False,
        pom=True,
    )

    result = loss(student, teacher, labels)
    masks = boundary_masks(
        labels,
        num_classes=2,
        output_size=(3, 3),
        edge_kernel_size=3,
    )
    teacher_probs = F.softmax(teacher, dim=1)
    pixel_kl = (
        teacher_probs
        * (F.log_softmax(teacher, dim=1) - F.log_softmax(student, dim=1))
    ).sum(dim=1)
    counts = (masks.edge > 0).sum(dim=(-2, -1)).clamp_min(1)
    expected = (
        torch.tensor([2.0, 3.0])
        * (pixel_kl.unsqueeze(1) * masks.edge).sum(dim=(-2, -1))
        / counts
    ).sum(dim=1).mean()

    torch.testing.assert_close(result.total, expected)
    result.total.backward()
    assert student.grad is not None


def test_bpkd_body_matches_channel_wise_kl_and_weighting():
    student = torch.tensor(
        [[[[0.2, -0.4], [0.8, 0.1]], [[-0.3, 0.9], [0.0, -0.6]]]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[[[0.5, -0.1], [0.2, -0.2]], [[-0.5, 0.4], [0.3, -0.8]]]]
    )
    labels = torch.zeros(1, 2, 2, dtype=torch.long)
    result = BPKDLoss(
        num_classes=2,
        body_weight=2.5,
        edge_weight=0,
        edge_kernel_size=3,
        spatial_stride=1,
    )(student, teacher, labels)

    student_log_probabilities = F.log_softmax(student.reshape(2, 4), dim=1)
    teacher_log_probabilities = F.log_softmax(teacher.reshape(2, 4), dim=1)
    expected_raw = (
        teacher_log_probabilities.exp()
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=1).mean()

    torch.testing.assert_close(result.components["raw_body"], expected_raw)
    torch.testing.assert_close(result.components["weighted_body"], expected_raw * 2.5)
    torch.testing.assert_close(result.total, expected_raw * 2.5)


def test_bpkd_prm_masks_logits_before_valid_pixel_reduction():
    torch.manual_seed(7)
    student = torch.randn(1, 2, 5, 5, requires_grad=True)
    teacher = torch.randn(1, 2, 5, 5)
    labels = _labels()
    result = BPKDLoss(
        num_classes=2,
        body_weight=0,
        edge_weight=3,
        edge_kernel_size=3,
        spatial_stride=1,
        prm=True,
        pom=False,
    )(student, teacher, labels)

    masks = boundary_masks(
        labels,
        num_classes=2,
        output_size=(5, 5),
        edge_kernel_size=3,
    )
    masked_student = student * masks.edge
    masked_teacher = teacher * masks.edge
    teacher_log_probabilities = F.log_softmax(masked_teacher, dim=1)
    pixel_kl = (
        teacher_log_probabilities.exp()
        * (teacher_log_probabilities - F.log_softmax(masked_student, dim=1))
    ).sum(dim=1)
    expected_raw = (pixel_kl * masks.valid[:, 0]).sum() / masks.valid.sum()

    torch.testing.assert_close(result.components["raw_edge"], expected_raw)
    torch.testing.assert_close(result.total, expected_raw * 3)


def test_bpkd_empty_edge_is_a_finite_differentiable_zero():
    student = torch.randn(1, 2, 8, 8, requires_grad=True)
    teacher = torch.randn(1, 2, 8, 8)
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    result = BPKDLoss(
        num_classes=2,
        body_weight=0,
        edge_weight=50,
        spatial_stride=1,
    )(student, teacher, labels)

    result.total.backward()

    assert result.total.item() == pytest.approx(0.0, abs=1e-6)
    assert student.grad is not None


@pytest.mark.parametrize(
    ("prm", "pom"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_bpkd_prm_and_pom_are_independently_switchable(prm, pom):
    student = torch.randn(1, 2, 5, 5, requires_grad=True)
    teacher = torch.randn(1, 2, 5, 5)
    result = BPKDLoss(
        num_classes=2,
        body_weight=0,
        edge_weight=1,
        edge_kernel_size=3,
        spatial_stride=1,
        prm=prm,
        pom=pom,
    )(student, teacher, _labels())

    result.total.backward()

    assert torch.isfinite(result.total)
    assert student.grad is not None


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
    assert result.components["raw_body"].item() == 0
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


def test_bpkd_uses_float32_kl_under_bfloat16_autocast_inputs():
    student = torch.randn(1, 2, 8, 8, dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn(1, 2, 8, 8, dtype=torch.bfloat16)
    labels = _labels().repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    result = BPKDLoss(num_classes=2, spatial_stride=1)(student, teacher, labels)

    result.total.backward()

    assert result.total.dtype == torch.float32
    assert torch.isfinite(result.total)
    assert student.grad is not None


def test_bpkd_rejects_logits_with_wrong_class_dimension():
    with pytest.raises(ValueError, match="channels"):
        BPKDLoss(num_classes=2)(
            torch.randn(1, 2, 5, 5),
            torch.randn(1, 3, 5, 5),
            _labels(),
        )
