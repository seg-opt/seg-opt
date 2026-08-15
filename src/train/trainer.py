from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch import nn

from src.train.profiling import GpuStatsCallback
from src.utils.logging import make_wandb_logger


class TrainerModule(L.LightningModule):
    """Train a student with hard labels and logits from a frozen teacher."""

    def __init__(
        self,
        teacher: nn.Module | None,
        student: nn.Module,
        task_loss: nn.Module,
        distillation_loss: nn.Module | None,
        task_weight: float = 1.0,
        distillation_weight: float = 1.0,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()

        self.teacher = teacher
        self.student = student
        self.task_loss = task_loss
        self.distillation_loss = distillation_loss
        self.task_weight = task_weight
        self.distillation_weight = distillation_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if self.teacher is not None:
            self.teacher.requires_grad_(False)
        if self.distillation_weight and self.teacher is None:
            raise ValueError("a teacher is required when distillation_weight is non-zero")

        self.save_hyperparameters(
            ignore=[
                "teacher",
                "student",
                "task_loss",
                "distillation_loss",
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.student(inputs)

    def on_train_epoch_start(self) -> None:
        if self.teacher is not None:
            self.teacher.eval()

    def shared_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        inputs = batch["pixel_values"]
        targets = batch["labels"]
        student_logits = self.student(inputs)

        # todo make abstraction on this loss so it can be easily changed
        task_loss = self.task_loss(student_logits, targets)
        if self.distillation_weight:
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
            kd_loss = self.distillation_loss(student_logits, teacher_logits)
        else:
            kd_loss = task_loss.new_zeros(())
        loss = self.task_weight * task_loss + self.distillation_weight * kd_loss

        metrics = {
            "loss": loss.detach(),
            "task_loss": task_loss.detach(),
            "kd_loss": kd_loss.detach(),
        }
        return loss, metrics

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, metrics = self.shared_step(batch)

        self.log(
            "train/loss",
            metrics["loss"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch["pixel_values"].shape[0],
        )
        self.log_dict(
            {
                f"train/{name}": value
                for name, value in metrics.items()
                if name != "loss"
            },
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=batch["pixel_values"].shape[0],
        )

        return loss

    def _evaluation_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        inputs = batch["pixel_values"]
        targets = batch["labels"]
        student_logits = self.student(inputs)
        loss = self.task_loss(student_logits, targets)

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=inputs.shape[0],
        )
        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._evaluation_step(batch, stage="val")

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._evaluation_step(batch, stage="test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.student.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


def create_trainer(
    output_dir: str | Path,
    max_epochs: int,
    precision: str = "32-true",
    devices: int | list[int] | str = "auto",
    log_every_n_steps: int = 50,
    profile: bool = False,
    profiler=None,
    fast_dev_run: bool = False,
) -> L.Trainer:
    output_dir = Path(output_dir)
    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            monitor="val/loss",
            mode="min",
            save_top_k=2,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if profile:
        callbacks.append(GpuStatsCallback(every_n_steps=log_every_n_steps))

    return L.Trainer(
        default_root_dir=output_dir,
        accelerator="auto",
        devices=devices,
        max_epochs=max_epochs,
        precision=precision,
        gradient_clip_val=1.0,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
        logger=make_wandb_logger(output_dir),
        profiler=profiler,
        fast_dev_run=fast_dev_run,
    )
