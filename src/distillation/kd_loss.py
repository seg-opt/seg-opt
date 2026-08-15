import torch
from torch import nn
from torch.nn import functional as F


class LogitDistillationLoss(nn.Module):
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        return self.kl_div(student_probs, teacher_probs) * self.temperature**2
