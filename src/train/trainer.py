import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys

from src.distillation.bpkd import DistillationLossResult
from src.train.metrics import SegmentationMetrics
from src.train.profiling import GpuStatsCallback, NvtxRangesCallback
from src.utils.logging import make_wandb_logger


def cosine_with_warmup_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
) -> float:
    """Learning-rate multiplier for linear warmup followed by cosine decay."""
    step = min(step, total_steps - 1)
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps

    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return 1.0
    progress = (step - warmup_steps) / (decay_steps - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _unique_trainable(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    unique: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if parameter.requires_grad and id(parameter) not in seen:
            unique.append(parameter)
            seen.add(id(parameter))
    return unique


def parameter_groups(
    model: nn.Module,
    *,
    head_learning_rate: float,
    backbone_learning_rate: float | None,
) -> list[dict[str, Any]]:
    """Split trainable model parameters into disjoint head/backbone groups."""

    trainable = _unique_trainable(model.parameters())
    if not trainable:
        raise ValueError("model has no trainable parameters")

    encoder_getter = getattr(model, "encoder_parameters", None)
    encoder = _unique_trainable(encoder_getter()) if callable(encoder_getter) else []
    trainable_ids = {id(parameter) for parameter in trainable}
    if any(id(parameter) not in trainable_ids for parameter in encoder):
        raise ValueError("encoder_parameters returned a parameter outside the model")

    encoder_ids = {id(parameter) for parameter in encoder}
    head = [parameter for parameter in trainable if id(parameter) not in encoder_ids]
    groups: list[dict[str, Any]] = []
    if head:
        groups.append(
            {
                "name": "head",
                "params": head,
                "lr": head_learning_rate,
            }
        )
    if encoder:
        if backbone_learning_rate is None or backbone_learning_rate <= 0:
            raise ValueError(
                "a positive backbone_learning_rate is required when the model "
                "exposes trainable encoder parameters"
            )
        groups.append(
            {
                "name": "backbone",
                "params": encoder,
                "lr": backbone_learning_rate,
            }
        )
    return groups


class BaselineModule(L.LightningModule):
    """Supervised Lightning module shared by all segmentation baselines."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        *,
        head_learning_rate: float,
        backbone_learning_rate: float | None = None,
        weight_decay: float = 0.0,
        warmup_fraction: float = 0.05,
        class_names: Sequence[str],
        ignore_index: int = 255,
        trimap_kernel_size: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        super().__init__()
        if not 0 <= warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if total_steps is not None and total_steps <= 0:
            raise ValueError("total_steps must be positive when provided")

        self.model = model
        self.criterion = criterion
        self.head_learning_rate = head_learning_rate
        self.backbone_learning_rate = backbone_learning_rate
        self.weight_decay = weight_decay
        self.warmup_fraction = warmup_fraction
        self.total_steps = total_steps
        self.metrics = SegmentationMetrics(
            class_names,
            ignore_index,
            trimap_kernel_size=trimap_kernel_size,
        )
        self._latest_metrics: dict[str, dict[str, Any]] = {}
        # A subclass (DistillationModule) may add nn.Module constructor
        # arguments.  Ignore their conventional names here too because this
        # call inspects the active subclass frame under Lightning.
        self.save_hyperparameters(
            ignore=["model", "criterion", "teacher", "distillation_loss"]
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def _validate_scores(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        expected = (
            labels.shape[0],
            len(self.metrics.class_names),
            *labels.shape[-2:],
        )
        if tuple(scores.shape) != expected:
            raise ValueError(
                "model scores must have shape [B, classes, H, W]; "
                f"expected {expected}, got {tuple(scores.shape)}"
            )

    def _loss_and_scores(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss_and_scores = getattr(self.model, "loss_and_scores", None)
        if not callable(loss_and_scores):
            raise TypeError("baseline model must implement loss_and_scores")

        result = loss_and_scores(images, labels, self.criterion)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("loss_and_scores must return (loss, scores)")
        loss, scores = result
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise TypeError("loss_and_scores must return a scalar tensor loss")
        if not isinstance(scores, torch.Tensor):
            raise TypeError("loss_and_scores must return tensor scores")
        self._validate_scores(scores, labels)
        return loss, scores

    def _log_loss(
        self,
        stage: str,
        loss: torch.Tensor,
        *,
        batch_size: int,
        on_step: bool,
    ) -> None:
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=on_step,
            on_epoch=True,
            sync_dist=not on_step,
            batch_size=batch_size,
        )

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        images = batch["pixel_values"]
        labels = batch["labels"]
        loss, scores = self._loss_and_scores(images, labels)
        self.metrics.update("train", scores.detach().argmax(dim=1), labels)
        self._log_loss(
            "train",
            loss,
            batch_size=images.shape[0],
            on_step=True,
        )
        return loss

    def _evaluation_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        images = batch["pixel_values"]
        labels = batch["labels"]
        loss, scores = self._loss_and_scores(images, labels)
        self.metrics.update(stage, scores.argmax(dim=1), labels)
        self._log_loss(
            stage,
            loss,
            batch_size=images.shape[0],
            on_step=False,
        )
        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._evaluation_step(batch, "val")

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._evaluation_step(batch, "test")

    def _reset_metrics(self, stage: str) -> None:
        self.metrics.reset(stage)

    def _log_metrics(self, stage: str) -> None:
        scalar_metrics, confusion = self.metrics.snapshot(stage)
        self._latest_metrics[stage] = {
            **{
                name: float(value.detach().cpu())
                for name, value in scalar_metrics.items()
            },
            "confusion_matrix": confusion.detach().cpu().tolist(),
        }
        for name, value in scalar_metrics.items():
            self.log(
                f"{stage}/{name}",
                value,
                prog_bar=name == "miou",
                on_step=False,
                on_epoch=True,
            )

    def latest_metrics(self, stage: str) -> dict[str, Any]:
        """Return a copy of the latest JSON-safe epoch metrics."""
        if stage not in self._latest_metrics:
            raise RuntimeError(f"no completed {stage!r} epoch is available")
        metrics = self._latest_metrics[stage]
        return {
            **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            "confusion_matrix": [row.copy() for row in metrics["confusion_matrix"]],
        }

    def on_train_epoch_start(self) -> None:
        self._reset_metrics("train")

    def on_train_epoch_end(self) -> None:
        self._log_metrics("train")

    def on_validation_epoch_start(self) -> None:
        self._reset_metrics("val")

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val")

    def on_test_epoch_start(self) -> None:
        self._reset_metrics("test")

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test")

    def _estimated_total_steps(self) -> int:
        if self.total_steps is not None:
            return self.total_steps
        total_steps = int(self.trainer.estimated_stepping_batches)
        if total_steps <= 0:
            raise RuntimeError("trainer estimated no optimizer steps")
        return total_steps

    def configure_optimizers(self) -> dict[str, Any]:
        groups = parameter_groups(
            self.model,
            head_learning_rate=self.head_learning_rate,
            backbone_learning_rate=self.backbone_learning_rate,
        )
        optimizer = torch.optim.AdamW(groups, weight_decay=self.weight_decay)

        total_steps = self._estimated_total_steps()
        warmup_steps = min(total_steps, math.ceil(total_steps * self.warmup_fraction))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_with_warmup_multiplier(
                step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "cosine_with_warmup",
            },
        }


class DistillationModule(BaselineModule):
    """Canonical baseline trainer with a frozen teacher used only for training."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        *,
        teacher: nn.Module,
        distillation_loss: nn.Module,
        head_learning_rate: float,
        backbone_learning_rate: float | None = None,
        weight_decay: float = 0.0,
        warmup_fraction: float = 0.05,
        class_names: Sequence[str],
        ignore_index: int = 255,
        trimap_kernel_size: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        super().__init__(
            model,
            criterion,
            head_learning_rate=head_learning_rate,
            backbone_learning_rate=backbone_learning_rate,
            weight_decay=weight_decay,
            warmup_fraction=warmup_fraction,
            class_names=class_names,
            ignore_index=ignore_index,
            trimap_kernel_size=trimap_kernel_size,
            total_steps=total_steps,
        )
        self.teacher = teacher
        self.distillation_loss = distillation_loss
        self.teacher_metrics = SegmentationMetrics(
            class_names,
            ignore_index,
            trimap_kernel_size=trimap_kernel_size,
        )
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.save_hyperparameters(
            ignore=["model", "criterion", "teacher", "distillation_loss"]
        )

    @staticmethod
    def _distillation_logits(model: nn.Module, scores: torch.Tensor) -> torch.Tensor:
        converter = getattr(model, "to_distillation_logits", None)
        logits = converter(scores) if callable(converter) else scores
        if not isinstance(logits, torch.Tensor):
            raise TypeError("to_distillation_logits must return a tensor")
        return logits

    def train(self, mode: bool = True) -> "DistillationModule":
        super().train(mode)
        self.teacher.eval()
        return self

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.teacher_metrics.reset("train")
        self.teacher.eval()

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        teacher_metrics, _ = self.teacher_metrics.snapshot("train")
        diagnostic_names = ["miou", "iou_small_rock"]
        if self.teacher_metrics.trimap_kernel_size is not None:
            diagnostic_names.append("trimap_iou_small_rock")
        for name in diagnostic_names:
            self.log(
                f"train/teacher_{name}",
                teacher_metrics[name],
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def _log_training_component(
        self,
        name: str,
        value: torch.Tensor,
        *,
        batch_size: int,
    ) -> None:
        self.log(
            f"train/{name}",
            value.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=batch_size,
        )

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        images = batch["pixel_values"]
        labels = batch["labels"]
        task_loss, scores = self._loss_and_scores(images, labels)
        self.teacher.eval()
        with torch.no_grad():
            teacher_scores = self.teacher(images)
        result = self.distillation_loss(
            self._distillation_logits(self.model, scores),
            self._distillation_logits(self.teacher, teacher_scores),
            labels,
        )
        if not isinstance(result, DistillationLossResult):
            raise TypeError("distillation_loss must return DistillationLossResult")
        if result.total.ndim != 0:
            raise TypeError("distillation loss must return a scalar tensor")

        loss = task_loss + result.total
        self.metrics.update("train", scores.detach().argmax(dim=1), labels)
        teacher_predictions = teacher_scores.argmax(dim=1)
        self.teacher_metrics.update("train", teacher_predictions, labels)
        valid = labels != self.metrics.ignore_index
        valid_count = valid.sum()
        agreement = torch.where(
            valid_count > 0,
            (scores.detach().argmax(dim=1).eq(teacher_predictions) & valid)
            .sum()
            .to(dtype=scores.dtype)
            / valid_count.clamp_min(1).to(dtype=scores.dtype),
            scores.sum() * 0,
        )
        kd_to_task_ratio = result.total.detach() / task_loss.detach().clamp_min(1e-8)
        self._log_loss("train", loss, batch_size=images.shape[0], on_step=True)
        self._log_training_component(
            "task_loss",
            task_loss,
            batch_size=images.shape[0],
        )
        self._log_training_component(
            "distillation_loss",
            result.total,
            batch_size=images.shape[0],
        )
        self._log_training_component(
            "teacher_student_agreement",
            agreement,
            batch_size=images.shape[0],
        )
        self._log_training_component(
            "kd_to_task_ratio",
            kd_to_task_ratio,
            batch_size=images.shape[0],
        )
        for name, value in result.components.items():
            self._log_training_component(
                f"distillation_{name}",
                value,
                batch_size=images.shape[0],
            )
        return loss

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Exclude frozen teacher weights from student deployment checkpoints."""
        state = super().state_dict(*args, **kwargs)
        for key in [key for key in state if key.startswith("teacher.")]:
            state.pop(key)
        return state

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> _IncompatibleKeys:
        incompatible = super().load_state_dict(
            state_dict,
            strict=False,
            assign=assign,
        )
        missing = [
            key for key in incompatible.missing_keys if not key.startswith("teacher.")
        ]
        unexpected = list(incompatible.unexpected_keys)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "distillation checkpoint state does not match the student: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return _IncompatibleKeys(missing, unexpected)


def create_trainer(
    output_dir: str | Path,
    max_epochs: int,
    *,
    precision: str = "bf16-mixed",
    devices: int | list[int] | str = "auto",
    accumulate_grad_batches: int = 1,
    log_every_n_steps: int = 50,
    max_steps: int | None = None,
    enable_nvtx_ranges: bool = False,
    profile: bool = False,
    profiler: Any = None,
    fast_dev_run: bool = False,
    enable_wandb: bool = True,
) -> L.Trainer:
    """Create the canonical trainer.

    ``fast_dev_run`` is a checkpoint-preserving one-batch smoke mode rather
    than Lightning's built-in mode, which disables checkpoint callbacks.
    """
    if accumulate_grad_batches <= 0:
        raise ValueError("accumulate_grad_batches must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when specified")

    output_dir = Path(output_dir)
    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best",
            auto_insert_metric_name=False,
            monitor="val/miou",
            mode="max",
            save_top_k=1,
            save_last=False,
            save_weights_only=True,
        )
    ]
    if enable_wandb:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if profile:
        callbacks.append(GpuStatsCallback(every_n_steps=log_every_n_steps))
    if enable_nvtx_ranges:
        callbacks.append(NvtxRangesCallback())

    return L.Trainer(
        default_root_dir=output_dir,
        accelerator="auto",
        devices=devices,
        max_epochs=1 if fast_dev_run else max_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
        logger=make_wandb_logger(output_dir) if enable_wandb else False,
        profiler=profiler,
        fast_dev_run=False,
        limit_train_batches=1 if fast_dev_run else 1.0,
        limit_val_batches=1 if fast_dev_run else 1.0,
        limit_test_batches=1 if fast_dev_run else 1.0,
        num_sanity_val_steps=0 if fast_dev_run else 2,
        deterministic="warn",
    )
