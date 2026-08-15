from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import lightning as L
from torch import nn

from src.data.dataloaders import make_dataloaders
from src.data.dataset import load_datasets
from src.data.processors import SegmentationImageProcessor
from src.distillation.kd_loss import LogitDistillationLoss
from src.models.loaders import load_student, load_teacher
from src.train.profiling import make_profiler
from src.train.trainer import TrainerModule, create_trainer
from src.utils.logging import configure_logging
from src.utils.utils import load_config


PRECISION = {"fp32": "32-true", "fp16": "16-mixed", "bf16": "bf16-mixed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a distilled segmentation model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--devices", default="auto", help="Lightning device count/list")
    parser.add_argument("--profile", action="store_true", help="record GPU metrics and a PyTorch trace")
    parser.add_argument(
        "--profiler-dir",
        type=Path,
        help="trace directory (defaults to $TMPDIR/seg-opt-profiler)",
    )
    parser.add_argument("--fast-dev-run", action="store_true", help="run one batch through each loop")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(str(args.config))
    output_dir = Path(config.training.output_dir)
    log = configure_logging(output_dir, args.verbose)
    L.seed_everything(config.training.seed, workers=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")
    if config.distillation.weight:
        log.info("Loading teacher %s", config.model.teacher.model_id)
        teacher, processor = load_teacher(
            config.model.teacher.name,
            config.model.teacher.model_id,
            config.model.num_classes,
            config.data.class_names,
        )
    else:
        log.info("Distillation disabled; running a supervised student baseline")
        teacher = None
        processor = SegmentationImageProcessor(config.data.image_size)
    log.info("Loading student %s", config.model.student.backbone_id)
    student = load_student(
        config.model.student.backbone_id,
        config.model.num_classes,
        config.model.student.pretrained,
    )

    datasets = load_datasets(
        processor=processor,
        root=config.data.dataset_root,
        excluded_id_files=config.data.excluded_id_files,
        train_frac=config.data.train_frac,
        val_frac=config.data.val_frac,
        test_frac=config.data.test_frac,
        seed=config.training.seed,
    )
    loaders = make_dataloaders(
        *datasets,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.training.seed,
    )
    log.info("Dataset sizes: train=%d val=%d test=%d", *(len(x) for x in datasets))

    module = TrainerModule(
        teacher=teacher,
        student=student,
        task_loss=nn.CrossEntropyLoss(),
        distillation_loss=(
            LogitDistillationLoss(config.distillation.temperature)
            if config.distillation.weight
            else None
        ),
        task_weight=config.distillation.hard_label_weight,
        distillation_weight=config.distillation.weight,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    profiler_dir = args.profiler_dir
    if profiler_dir is None:
        profiler_dir = Path(
            os.getenv("TMPDIR", tempfile.gettempdir())
        ) / "seg-opt-profiler"
    profiler = make_profiler(profiler_dir) if args.profile else None
    if args.profile:
        log.info("Profiler artifacts: %s", profiler_dir.resolve())

    trainer = create_trainer(
        output_dir=output_dir,
        max_epochs=config.training.epochs,
        precision=PRECISION[config.training.precision],
        devices=args.devices,
        log_every_n_steps=config.training.logging_steps,
        profile=args.profile,
        profiler=profiler,
        fast_dev_run=args.fast_dev_run,
    )
    log.info("Starting training%s", " with profiling" if args.profile else "")
    trainer.fit(module, train_dataloaders=loaders[0], val_dataloaders=loaders[1])
    trainer.test(
        module,
        dataloaders=loaders[2],
        ckpt_path=None if args.fast_dev_run else "best",
    )
    log.info("Training complete; artifacts written to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
