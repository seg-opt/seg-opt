import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitDistillationLoss(nn.Module):
    def __init__(self, temperature, alfa):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        self.alfa = alfa


    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor):
        student_probab = F.log_softmax(student_logits)
        teacher_probab = F.softmax(teacher_logits)

        kd_loss = self.kl_div(student_probab, teacher_probab) * (self.temperature ** 2)
        task_loss = self.ce_loss(labels, student_logits)

        total_loss = self.alfa * kd_loss + (1.0 - self.alfa) * task_loss

        return total_loss