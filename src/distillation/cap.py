"""Cross-attention feature projector from ScaleKD."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.distillation.bpkd import DistillationLossResult


class CrossAttention(nn.Module):
    """Multi-head attention with teacher-space queries and student K/V tokens."""

    def __init__(
        self,
        *,
        student_dims: int,
        teacher_dims: int,
        num_heads: int,
        softmax_scale: float,
    ) -> None:
        super().__init__()
        self.student_dims = student_dims
        self.teacher_dims = teacher_dims
        self.num_heads = num_heads
        if self.teacher_dims % self.num_heads:
            raise ValueError("teacher_dims must be divisible by num_heads")
        if (
            isinstance(softmax_scale, bool)
            or not isinstance(softmax_scale, (int, float))
            or not math.isfinite(softmax_scale)
            or softmax_scale <= 0
        ):
            raise ValueError("softmax_scale must be a positive finite number")

        self.head_dims = self.teacher_dims // self.num_heads
        self.scale = self.head_dims**-0.5 * float(softmax_scale)
        self.q = nn.Linear(self.teacher_dims, self.teacher_dims)
        self.k = nn.Linear(self.student_dims, self.teacher_dims)
        self.v = nn.Linear(self.student_dims, self.teacher_dims)
        self.proj = nn.Linear(self.teacher_dims, self.teacher_dims)

    def forward(
        self,
        student_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if student_tokens.ndim != 3:
            raise ValueError("student_tokens must have shape [B, N, student_dims]")
        if query_tokens.ndim != 3:
            raise ValueError("query_tokens must have shape [B, N, teacher_dims]")
        if student_tokens.shape[0] != query_tokens.shape[0]:
            raise ValueError("student and query tokens must have matching batch size")
        if student_tokens.shape[-1] != self.student_dims:
            raise ValueError("student token channels do not match student_dims")
        if query_tokens.shape[-1] != self.teacher_dims:
            raise ValueError("query token channels do not match teacher_dims")

        batch_size, student_token_count, _ = student_tokens.shape
        query_token_count = query_tokens.shape[1]

        query = self.q(query_tokens).reshape(
            batch_size,
            query_token_count,
            self.num_heads,
            self.head_dims,
        )
        key = self.k(student_tokens).reshape(
            batch_size,
            student_token_count,
            self.num_heads,
            self.head_dims,
        )
        value = self.v(student_tokens).reshape(
            batch_size,
            student_token_count,
            self.num_heads,
            self.head_dims,
        )

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        output = attention @ value
        output = output.transpose(1, 2).reshape(
            batch_size,
            query_token_count,
            self.teacher_dims,
        )
        return self.proj(output)


class ResidualFFN(nn.Module):
    """ScaleKD's transformer FFN with its residual connection."""

    def __init__(self, embed_dims: int) -> None:
        super().__init__()
        hidden_dims = embed_dims * 4
        self.fc1 = nn.Linear(embed_dims, hidden_dims)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dims, embed_dims)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.fc2(self.activation(self.fc1(inputs)))


class AttentionProjector(nn.Module):
    """Align a fixed-size student feature map with a teacher feature space."""

    def __init__(
        self,
        student_dims: int,
        teacher_dims: int,
        hw_dims: tuple[int, int],
        *,
        num_heads: int = 8,
        softmax_scale: float = 5.0,
    ) -> None:
        super().__init__()
        self.student_dims = student_dims
        self.teacher_dims = teacher_dims
        self.hw_dims = hw_dims

        height, width = self.hw_dims
        self.proj_student = nn.Sequential(
            nn.Conv2d(
                self.student_dims,
                self.student_dims,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(self.student_dims),
            nn.ReLU(),
        )
        self.pos_embed = nn.Parameter(
            torch.empty(1, self.student_dims, height, width)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.query = nn.Embedding(height * width, self.teacher_dims)
        self.proj_pos = nn.Sequential(
            nn.Conv2d(self.teacher_dims, self.teacher_dims, kernel_size=1),
            nn.ReLU(),
        )
        self.pos_attention = CrossAttention(
            student_dims=self.student_dims,
            teacher_dims=self.teacher_dims,
            num_heads=num_heads,
            softmax_scale=softmax_scale,
        )
        self.norm = nn.LayerNorm(self.teacher_dims)
        self.ffn = ResidualFFN(self.teacher_dims)

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape [B, student_dims, H, W]")
        if inputs.shape[0] <= 0:
            raise ValueError("inputs must have a positive batch size")
        if inputs.shape[1] != self.student_dims:
            raise ValueError(
                "input channels do not match student_dims: "
                f"expected {self.student_dims}, got {inputs.shape[1]}"
            )
        if tuple(inputs.shape[-2:]) != self.hw_dims:
            raise ValueError(
                "input spatial shape does not match hw_dims: "
                f"expected {self.hw_dims}, got {tuple(inputs.shape[-2:])}"
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(inputs)
        batch_size = inputs.shape[0]
        height, width = self.hw_dims

        student_features = self.proj_student(inputs) + self.pos_embed
        student_tokens = student_features.flatten(2).transpose(1, 2)

        query_features = self.query.weight.reshape(
            height,
            width,
            self.teacher_dims,
        )
        query_features = query_features.permute(2, 0, 1).unsqueeze(0)
        query_features = query_features.expand(batch_size, -1, -1, -1)
        query_tokens = self.proj_pos(query_features).flatten(2).transpose(1, 2)

        projected_tokens = self.pos_attention(student_tokens, query_tokens)
        projected_tokens = self.ffn(self.norm(projected_tokens))
        return projected_tokens.transpose(1, 2).reshape(
            batch_size,
            self.teacher_dims,
            height,
            width,
        ).contiguous()


class CAPLoss(nn.Module):
    """ScaleKD feature matching through a cross-attention projector."""

    def __init__(
        self,
        student_dims: int,
        teacher_dims: int,
        hw_dims: tuple[int, int],
        *,
        weight: float = 1.0,
        num_heads: int = 8,
        softmax_scale: float = 5.0,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.projector = AttentionProjector(
            student_dims=student_dims,
            teacher_dims=teacher_dims,
            hw_dims=hw_dims,
            num_heads=num_heads,
            softmax_scale=softmax_scale,
        )

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> DistillationLossResult:
        projected = self.projector(student_features)
        if teacher_features.shape != projected.shape:
            raise ValueError(
                "teacher feature shape must match projected student feature shape: "
                f"expected {tuple(projected.shape)}, got {tuple(teacher_features.shape)}"
            )

        projected = F.normalize(projected.float(), dim=1)
        teacher = F.normalize(teacher_features.detach().float(), dim=1)
        raw_cap = F.mse_loss(projected, teacher, reduction="sum") / projected.shape[0]
        weighted_cap = raw_cap * self.weight
        return DistillationLossResult(
            total=weighted_cap,
            components={
                "raw_cap": raw_cap,
                "weighted_cap": weighted_cap,
            },
        )
