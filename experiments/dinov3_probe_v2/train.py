import argparse
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import lightning as L

from src.data.dataloaders import make_dataloaders
from src.data.dataset import load_datasets
from src.data.transforms import EvalSegmentationTransform, TrainSegmentationTransform
from src.models.loaders import load_dinov3
from src.train.losses import SegmentationLoss
from src.train.profiling import make_profiler
from src.train.trainer import TrainerModule, create_trainer
from src.utils.logging import configure_logging
from src.utils.utils import load_config


PRECISION = {"fp32": "32-true", "fp16": "16-mixed", "bf16": "bf16-mixed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a supervised DINOv3 lunar segmentation probe"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--devices", default="auto", help="Lightning device count/list")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profiler-dir", type=Path)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--offline", action="store_true", help="disable Weights & Biases")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    teacher = config.model.teacher
    if teacher is None or teacher.name != "dinov3":
        raise ValueError("train_dinov3 requires model.teacher.name=dinov3")
    if len(config.data.class_names) != config.model.num_classes:
        raise ValueError("class_names must contain one name per model class")

    output_dir = Path(config.training.output_dir)
    log = configure_logging(output_dir, args.verbose)
    L.seed_everything(config.training.seed, workers=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")

    log.info("Loading DINOv3 backbone %s", teacher.model_id)
    model, processor = load_dinov3(
        teacher.model_id,
        num_classes=config.model.num_classes,
        freeze_backbone=teacher.freeze,
        head_type=teacher.head_type,
        head_batch_norm=teacher.head_batch_norm,
        head_hidden_channels=teacher.head_hidden_channels,
        head_dropout=teacher.head_dropout,
        unfreeze_last_blocks=teacher.unfreeze_last_blocks,
    )

    augmentation_kwargs = asdict(config.data.augmentation)
    train_transform = TrainSegmentationTransform(
        crop_size=config.data.image_size,
        ignore_index=config.loss.ignore_index,
        **augmentation_kwargs,
    )
    eval_transform = EvalSegmentationTransform(
        short_side=config.data.eval_short_side,
        size_divisor=config.data.size_divisor,
        ignore_index=config.loss.ignore_index,
    )
    datasets = load_datasets(
        processor=processor,
        root=config.data.dataset_root,
        excluded_id_files=config.data.excluded_id_files,
        train_frac=config.data.train_frac,
        val_frac=config.data.val_frac,
        test_frac=config.data.test_frac,
        seed=config.training.seed,
        train_transform=train_transform,
        eval_transform=eval_transform,
        processor_kwargs={"do_resize": False},
        max_train_samples=config.data.max_train_samples,
        max_val_samples=config.data.max_val_samples,
        max_test_samples=config.data.max_test_samples,
    )
    loaders = make_dataloaders(
        *datasets,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.training.seed,
    )
    log.info("Dataset sizes: train=%d val=%d test=%d", *(len(x) for x in datasets))

    task_loss = SegmentationLoss(
        num_classes=config.model.num_classes,
        class_weights=config.loss.class_weights,
        cross_entropy_weight=config.loss.cross_entropy_weight,
        dice_weight=config.loss.dice_weight,
        ignore_index=config.loss.ignore_index,
    )
    module = TrainerModule(
        teacher=None,
        student=model,
        task_loss=task_loss,
        distillation_loss=None,
        task_weight=1.0,
        distillation_weight=0.0,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        backbone_learning_rate=config.training.backbone_learning_rate,
        class_names=config.data.class_names,
        ignore_index=config.loss.ignore_index,
        eval_crop_size=config.evaluation.crop_size,
        eval_stride=config.evaluation.stride,
    )

    profiler_dir = args.profiler_dir
    if profiler_dir is None:
        profiler_dir = Path(
            os.getenv("TMPDIR", tempfile.gettempdir())
        ) / "seg-opt-dinov3-profiler"
    profiler = make_profiler(profiler_dir) if args.profile else None
    trainer = create_trainer(
        output_dir=output_dir,
        max_epochs=config.training.epochs,
        precision=PRECISION[config.training.precision],
        devices=args.devices,
        log_every_n_steps=config.training.logging_steps,
        profile=args.profile,
        profiler=profiler,
        fast_dev_run=args.fast_dev_run,
        enable_wandb=not args.offline,
        checkpoint_monitor="val/miou",
        checkpoint_mode="max",
    )
    if trainer.logger is not None:
        trainer.logger.log_hyperparams(asdict(config))
    log.info("Starting DINOv3 probe training")
    trainer.fit(module, train_dataloaders=loaders[0], val_dataloaders=loaders[1])
    trainer.test(
        module,
        dataloaders=loaders[2],
        ckpt_path=None if args.fast_dev_run else "best",
    )
    log.info("Training complete; artifacts written to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
