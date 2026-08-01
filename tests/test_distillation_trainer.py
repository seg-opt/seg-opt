import pytest
import torch
from torch import nn

from src.distillation.kd_loss import LogitDistillationLoss
from src.train.trainer import TrainerModule


class TinySegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.projection = nn.Conv2d(3, num_classes, kernel_size=1)
        self.forward_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        return self.projection(inputs)


def make_module():
    teacher = TinySegmentationModel(4)
    student = TinySegmentationModel(4)
    module = TrainerModule(
        teacher=teacher,
        student=student,
        task_loss=nn.CrossEntropyLoss(),
        distillation_loss=LogitDistillationLoss(temperature=4.0),
        task_weight=1.0,
        distillation_weight=0.5,
        learning_rate=1e-3,
        weight_decay=0.01,
    )
    return module, teacher, student


def make_batch():
    return {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "labels": torch.randint(0, 4, (2, 8, 8)),
    }


def test_kd_loss_is_zero_for_identical_logits():
    logits = torch.randn(2, 4, 8, 8)
    loss = LogitDistillationLoss(temperature=3.0)(logits, logits)

    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_shared_step_combines_losses_and_only_trains_student():
    module, teacher, student = make_module()
    module.train()
    module.on_train_epoch_start()

    loss, metrics = module.shared_step(make_batch())
    loss.backward()

    expected = metrics["task_loss"] + 0.5 * metrics["kd_loss"]
    assert metrics["loss"].item() == pytest.approx(expected.item())
    assert student.projection.weight.grad is not None
    assert teacher.projection.weight.grad is None
    assert not teacher.training


def test_validation_uses_student_without_calling_teacher():
    module, teacher, _ = make_module()
    module.log = lambda *args, **kwargs: None

    loss = module.validation_step(make_batch(), batch_idx=0)

    assert loss.ndim == 0
    assert teacher.forward_calls == 0


def test_optimizer_uses_configured_hyperparameters():
    module, _, student = make_module()

    optimizer = module.configure_optimizers()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)
    assert set(optimizer.param_groups[0]["params"]) == set(student.parameters())
