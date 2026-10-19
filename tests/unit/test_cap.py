import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.distillation.cap import AttentionProjector, CAPLoss, CrossAttention


def test_attention_projector_supports_the_segformer_b0_b4_contract():
    projector = AttentionProjector(
        student_dims=256,
        teacher_dims=512,
        hw_dims=(16, 24),
        num_heads=8,
        softmax_scale=5.0,
    ).eval()
    student_features = torch.randn(2, 256, 16, 24)

    with torch.no_grad():
        projected = projector(student_features)

    assert projected.shape == (2, 512, 16, 24)
    assert torch.isfinite(projected).all()


def test_attention_projector_backpropagates_through_every_component():
    projector = AttentionProjector(
        student_dims=4,
        teacher_dims=8,
        hw_dims=(2, 3),
        num_heads=2,
    )
    student_features = torch.randn(2, 4, 2, 3, requires_grad=True)

    projector(student_features).square().mean().backward()

    parameters = (
        projector.pos_embed,
        projector.query.weight,
        projector.proj_student[0].weight,
        projector.proj_pos[0].weight,
        projector.pos_attention.q.weight,
        projector.pos_attention.k.weight,
        projector.pos_attention.v.weight,
        projector.pos_attention.proj.weight,
        projector.ffn.fc1.weight,
        projector.ffn.fc2.weight,
    )
    assert student_features.grad is not None
    assert all(parameter.grad is not None for parameter in parameters)


def test_global_cross_attention_matches_manual_scaled_attention():
    attention = CrossAttention(
        student_dims=2,
        teacher_dims=2,
        num_heads=1,
        softmax_scale=2.0,
    )
    with torch.no_grad():
        for projection in (attention.q, attention.k, attention.v, attention.proj):
            projection.weight.copy_(torch.eye(2))
            projection.bias.zero_()

    student_tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    query_tokens = torch.tensor([[[1.0, 1.0], [1.0, -1.0]]])
    expected_weights = torch.softmax(
        (query_tokens @ student_tokens.transpose(-2, -1))
        * (2.0 / math.sqrt(2.0)),
        dim=-1,
    )
    expected = expected_weights @ student_tokens

    torch.testing.assert_close(
        attention(student_tokens, query_tokens),
        expected,
    )


def test_batch_expansion_does_not_replace_or_duplicate_learned_parameters():
    projector = AttentionProjector(4, 8, (2, 3), num_heads=2).eval()
    query_pointer = projector.query.weight.data_ptr()
    query_elements = projector.query.weight.numel()
    position_pointer = projector.pos_embed.data_ptr()
    position_elements = projector.pos_embed.numel()

    with torch.no_grad():
        assert projector(torch.randn(1, 4, 2, 3)).shape == (1, 8, 2, 3)
        assert projector(torch.randn(4, 4, 2, 3)).shape == (4, 8, 2, 3)

    assert projector.query.weight.data_ptr() == query_pointer
    assert projector.query.weight.numel() == query_elements
    assert projector.pos_embed.data_ptr() == position_pointer
    assert projector.pos_embed.numel() == position_elements


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"teacher_dims": 7, "num_heads": 2}, "divisible"),
        ({"softmax_scale": 0}, "softmax_scale"),
        ({"softmax_scale": float("inf")}, "softmax_scale"),
    ],
)
def test_attention_projector_rejects_invalid_configuration(kwargs, message):
    defaults = {
        "student_dims": 4,
        "teacher_dims": 8,
        "hw_dims": (2, 3),
        "num_heads": 2,
        "softmax_scale": 5.0,
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        AttentionProjector(**defaults)


@pytest.mark.parametrize(
    "inputs, message",
    [
        (torch.randn(2, 4, 6), "shape"),
        (torch.randn(2, 5, 2, 3), "channels"),
        (torch.randn(2, 4, 3, 2), "spatial"),
        (torch.randn(0, 4, 2, 3), "batch"),
    ],
)
def test_attention_projector_rejects_invalid_input_shape(inputs, message):
    projector = AttentionProjector(4, 8, (2, 3), num_heads=2)

    with pytest.raises(ValueError, match=message):
        projector(inputs)


def test_attention_projector_rejects_non_tensor_input():
    projector = AttentionProjector(4, 8, (2, 3), num_heads=2)

    with pytest.raises(TypeError, match="torch.Tensor"):
        projector([[1.0]])


def test_projector_is_composed_only_of_torch_modules():
    projector = AttentionProjector(4, 8, (2, 3), num_heads=2)

    assert all(isinstance(module, nn.Module) for module in projector.modules())


def test_cap_loss_matches_scaledkd_normalization_and_reduction():
    loss = CAPLoss(2, 2, (1, 2), weight=2.5, num_heads=1)
    loss.projector = nn.Identity()
    student = torch.tensor(
        [
            [[[1.0, 2.0]], [[2.0, 0.0]]],
            [[[0.0, 3.0]], [[4.0, 4.0]]],
        ],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [
            [[[2.0, 0.0]], [[0.0, 1.0]]],
            [[[3.0, 1.0]], [[4.0, 2.0]]],
        ],
        requires_grad=True,
    )

    result = loss(student, teacher)
    expected = F.mse_loss(
        F.normalize(student, dim=1),
        F.normalize(teacher, dim=1),
        reduction="sum",
    ) / student.shape[0]

    torch.testing.assert_close(result.components["raw_cap"], expected)
    torch.testing.assert_close(result.components["weighted_cap"], expected * 2.5)
    torch.testing.assert_close(result.total, expected * 2.5)

    result.total.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_cap_loss_backpropagates_through_projector_but_not_teacher():
    loss = CAPLoss(4, 8, (2, 3), num_heads=2)
    student = torch.randn(2, 4, 2, 3, requires_grad=True)
    teacher = torch.randn(2, 8, 2, 3, requires_grad=True)

    result = loss(student, teacher)
    result.total.backward()

    assert result.total.ndim == 0
    assert result.total.dtype == torch.float32
    assert student.grad is not None
    assert teacher.grad is None
    assert all(
        parameter.grad is not None
        for parameter in loss.projector.parameters()
        if parameter.requires_grad
    )


def test_cap_loss_rejects_mismatched_teacher_features():
    loss = CAPLoss(4, 8, (2, 3), num_heads=2)

    with pytest.raises(ValueError, match="teacher feature shape"):
        loss(torch.randn(2, 4, 2, 3), torch.randn(2, 7, 2, 3))
