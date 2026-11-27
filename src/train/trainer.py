import colorsys
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from PIL import Image, ImageDraw
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys

from src.data.dataset import _decode_lusnar_pair
from src.data.processors import SegmentationImageProcessor
from src.data.transforms import EvalSegmentationTransform
from src.distillation.bpkd import DistillationLossResult
from src.distillation.sard import capture_features
from src.train.metrics import SegmentationMetrics
from src.train.profiling import GpuStatsCallback, NvtxRangesCallback
from src.utils.logging import make_wandb_logger

_TRAINING_ONLY_PREFIXES = ("teacher.", "distillation_loss.")


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

    def _loss_and_scores(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.loss_and_scores(images, labels, self.criterion)

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
        self.metrics.reset("train")

    def on_train_epoch_end(self) -> None:
        self._log_metrics("train")

    def on_validation_epoch_start(self) -> None:
        self.metrics.reset("val")

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val")

    def on_test_epoch_start(self) -> None:
        self.metrics.reset("test")

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test")

    def _estimated_total_steps(self) -> int:
        if self.total_steps is not None:
            return self.total_steps
        total_steps = int(self.trainer.estimated_stepping_batches)
        if total_steps <= 0:
            raise RuntimeError("trainer estimated no optimizer steps")
        return total_steps

    def _optimizer_groups(self) -> list[dict[str, Any]]:
        return parameter_groups(
            self.model,
            head_learning_rate=self.head_learning_rate,
            backbone_learning_rate=self.backbone_learning_rate,
        )

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self._optimizer_groups(),
            weight_decay=self.weight_decay,
        )

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
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.save_hyperparameters(
            ignore=["model", "criterion", "teacher", "distillation_loss"]
        )

    @staticmethod
    def _distillation_logits(model: nn.Module, scores: torch.Tensor) -> torch.Tensor:
        converter = getattr(model, "to_distillation_logits", None)
        logits = converter(scores) if callable(converter) else scores
        return logits

    def train(self, mode: bool = True) -> "DistillationModule":
        super().train(mode)
        self.teacher.eval()
        return self

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.teacher.eval()

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
        if getattr(self.distillation_loss, "requires_features", False):
            task_loss, scores, result = self._feature_distillation(images, labels)
        else:
            task_loss, scores = self._loss_and_scores(images, labels)
            # Lightning switches all child modules with the parent.  Reassert
            # this immediately before every teacher forward so the teacher
            # cannot ever update BatchNorm/dropout state during a student step.
            self.teacher.eval()
            with torch.no_grad():
                teacher_scores = self.teacher(images)
            result = self.distillation_loss(
                self._distillation_logits(self.model, scores),
                self._distillation_logits(self.teacher, teacher_scores),
                labels,
            )
        loss = task_loss + result.total
        self.metrics.update("train", scores.detach().argmax(dim=1), labels)
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
        for name, value in result.components.items():
            self._log_training_component(
                f"distillation_{name}",
                value,
                batch_size=images.shape[0],
            )
        return loss

    def _feature_distillation(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DistillationLossResult]:
        with capture_features(self.model) as student_features:
            task_loss, scores = self._loss_and_scores(images, labels)
        self.teacher.eval()
        with torch.no_grad(), capture_features(self.teacher) as teacher_features:
            self.teacher(images)
        if student_features.value is None or teacher_features.value is None:
            raise RuntimeError("feature tap did not fire during the forward pass")
        result = self.distillation_loss(
            student_features.value,
            teacher_features.value,
            labels,
        )
        return task_loss, scores, result

    def _optimizer_groups(self) -> list[dict[str, Any]]:
        """Also optimize trainable distillation heads, such as SARD projections."""
        groups = super()._optimizer_groups()
        auxiliary = _unique_trainable(self.distillation_loss.parameters())
        if auxiliary:
            groups.append(
                {
                    "name": "distillation",
                    "params": auxiliary,
                    "lr": self.head_learning_rate,
                }
            )
        return groups

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Keep only student weights in deployment checkpoints.

        The frozen teacher and any training-only distillation heads are
        excluded so a distilled checkpoint loads into a plain BaselineModule.
        """
        state = super().state_dict(*args, **kwargs)
        for key in [key for key in state if key.startswith(_TRAINING_ONLY_PREFIXES)]:
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
            key
            for key in incompatible.missing_keys
            if not key.startswith(_TRAINING_ONLY_PREFIXES)
        ]
        unexpected = list(incompatible.unexpected_keys)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "distillation checkpoint state does not match the student: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return _IncompatibleKeys(missing, unexpected)


class LusnarExamplesCallback(Callback):
    def __init__(
        self, examples_dir, output_dir, class_names, height, width, *,
        validation_dataset=None, checkpoint_sha256=None, ignore_index=255,
    ):
        self.examples_dir = Path(examples_dir) if examples_dir is not None else None
        self.validation_dataset = validation_dataset
        self.checkpoint_sha256 = checkpoint_sha256
        self.ignore_index = ignore_index
        self.output_dir = Path(output_dir) / "segmentation_samples"
        self.class_names = tuple(class_names)
        self.transform = EvalSegmentationTransform(height=height, width=width)
        self.processor = SegmentationImageProcessor()

    def _samples(self, manifest, validation_indices):
        if self.examples_dir is not None:
            for sample in manifest["samples"]:
                files = {Path(item["path"]).name: self.examples_dir / item["path"]
                         for item in sample["files"]}
                image, labels = _decode_lusnar_pair((files["rgb.png"], files["label.png"]))
                image, labels = self.transform(image, labels)
                pixels = self.processor(images=[image], return_tensors="pt")["pixel_values"]
                yield image, labels, pixels
            return

        dataset = self.validation_dataset
        for index in validation_indices:
            image, labels = dataset.decode_pair(dataset.pairs[index])
            image, labels = dataset.transform(image, labels)
            pixels = dataset.processor(
                images=[image], return_tensors="pt", **dataset.processor_kwargs,
            )["pixel_values"]
            yield image, labels, pixels

    def on_test_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        validation_indices = []
        if self.examples_dir is not None:
            source = "lusnar/examples"
            manifest_bytes = (self.examples_dir / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            filenames = [
                f"{sample['scene']}_{sample['timestamp']}.png" for sample in manifest["samples"]
            ]
        else:
            source = "validation"
            frame_ids = self.validation_dataset.frame_ids
            validation_indices = sorted(range(len(frame_ids)), key=frame_ids.__getitem__)[:8]
            filenames = [f"sample-{number:02d}.png" for number in range(len(validation_indices))]
            manifest = {"source": source, "samples": [
                {"frame_id": frame_ids[index], "file": filename}
                for index, filename in zip(validation_indices, filenames)
            ]}
        self.output_dir.mkdir(parents=True, exist_ok=False)
        if self.checkpoint_sha256 is not None:
            manifest["checkpoint_sha256"] = self.checkpoint_sha256
        if self.examples_dir is None or self.checkpoint_sha256 is not None:
            manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
        (self.output_dir / "manifest.json").write_bytes(manifest_bytes)
        palette = np.array([
            [round(channel * 255) for channel in colorsys.hsv_to_rgb(
                index / len(self.class_names), 0.75, 0.95
            )]
            for index in range(len(self.class_names))
        ], dtype=np.uint8)
        with torch.inference_mode(), trainer.precision_plugin.forward_context():
            for filename, (image, labels, pixels) in zip(filenames, self._samples(manifest, validation_indices), strict=True):
                predictions = pl_module(pixels.to(pl_module.device)).argmax(dim=1)[0]
                prediction = predictions.cpu().numpy()
                ignored = labels == self.ignore_index
                label_rgb = palette[np.where(ignored, 0, labels)]
                label_rgb[ignored] = (128, 128, 128)
                height, width = labels.shape
                legend = list(zip(self.class_names, palette))
                if ignored.any():
                    legend.append(("Ignored", (128, 128, 128)))
                canvas = Image.new(
                    "RGB", (max(3 * width, 300), height + 30 + 18 * len(legend)), "white"
                )
                draw = ImageDraw.Draw(canvas)
                for column, (title, panel) in enumerate((
                    ("Input", image), ("Ground truth", label_rgb),
                    ("Prediction", palette[prediction]),
                )):
                    canvas.paste(Image.fromarray(panel), (column * width, 24))
                    draw.text((column * width + 2, 4), title, fill="black")
                for index, (name, rgb) in enumerate(legend):
                    y = height + 30 + index * 18
                    color = tuple(int(value) for value in rgb)
                    draw.rectangle((4, y, 16, y + 12), fill=color)
                    draw.text((22, y), name, fill="black")
                canvas.save(self.output_dir / filename)
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb

                run = logger.experiment
                artifact_prefix = "lusnar-examples" if self.examples_dir else "segmentation-samples"
                artifact = wandb.Artifact(
                    name=f"{artifact_prefix}-{run.id}", type="evaluation",
                    metadata={"source": source, "sample_count": len(filenames)},
                )
                artifact.add_dir(str(self.output_dir))
                run.log_artifact(artifact)
                if filenames:
                    logger.log_image(
                        key="segmentation_samples",
                        images=[str(self.output_dir / filename) for filename in filenames],
                        caption=[
                            sample.get("frame_id", filename)
                            for sample, filename in zip(manifest["samples"], filenames)
                        ],
                    )


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
    wandb_run_name: str | None = None,
    wandb_config: dict | None = None,
) -> L.Trainer:
    """``fast_dev_run`` is a checkpoint-preserving one-batch smoke mode rather
    than Lightning's built-in mode, which disables checkpoint callbacks.
    """
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
        ),
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="epoch-{epoch:03d}",
            auto_insert_metric_name=False,
            every_n_epochs=5,
            save_on_train_epoch_end=True,
            save_top_k=-1,
            save_weights_only=False,
        ),
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
        logger=(
            make_wandb_logger(output_dir, run_name=wandb_run_name, run_config=wandb_config)
            if enable_wandb
            else False
        ),
        profiler=profiler,
        fast_dev_run=False,
        limit_train_batches=1 if fast_dev_run else 1.0,
        limit_val_batches=1 if fast_dev_run else 1.0,
        limit_test_batches=1 if fast_dev_run else 1.0,
        num_sanity_val_steps=0 if fast_dev_run else 2,
        deterministic="warn",
    )
