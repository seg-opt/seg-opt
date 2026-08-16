import argparse
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from scripts.evaluate import (
    PRECISION,
    evaluate_checkpoint,
    prepare_artifact_directory,
)
from src.baselines import build_baseline
from src.train.trainer import create_trainer
from src.utils.logging import configure_logging
from src.utils.utils import load_config, resolved_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one canonical supervised segmentation baseline"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--devices", default="auto")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="run one batch per stage while retaining and restoring best.ckpt",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--offline", action="store_true", help="disable W&B")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _output_directory(args: argparse.Namespace, model_name: str, configured: str) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if not args.fast_dev_run:
        return Path(configured)
    run_id = os.getenv("SLURM_JOB_ID", f"local-{os.getpid()}")
    restart = int(os.getenv("SLURM_RESTART_COUNT", "0"))
    if restart:
        run_id = f"{run_id}-retry{restart}"
    return Path("results") / "smoke" / run_id / model_name


def _checkpoint_callback(trainer: L.Trainer) -> ModelCheckpoint:
    callbacks = [
        callback
        for callback in trainer.callbacks
        if isinstance(callback, ModelCheckpoint)
    ]
    if len(callbacks) != 1:
        raise RuntimeError("canonical trainer must contain exactly one checkpoint callback")
    return callbacks[0]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = _output_directory(
        args,
        config.model.name,
        config.training.output_dir,
    )
    config = prepare_artifact_directory(config, output_dir)
    log = configure_logging(output_dir, args.verbose)
    L.seed_everything(config.training.seed, workers=True)

    log.info("Loading canonical baseline %s", config.model.name)
    components = build_baseline(config)
    log.info(
        "Dataset sizes: train=%d validation=%d test=%d; split=%s",
        len(components.train_dataset),
        len(components.validation_dataset),
        len(components.test_dataset),
        components.split_manifest.sha256,
    )
    trainer = create_trainer(
        output_dir=output_dir,
        max_epochs=config.training.epochs,
        precision=PRECISION[config.training.precision],
        devices=args.devices,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        log_every_n_steps=config.training.logging_steps,
        fast_dev_run=args.fast_dev_run,
        enable_wandb=not args.offline,
    )
    if trainer.logger is not None:
        trainer.logger.log_hyperparams(resolved_config(config))

    log.info("Starting %s run", "smoke" if args.fast_dev_run else "full")
    trainer.fit(
        components.module,
        train_dataloaders=components.train_loader,
        val_dataloaders=components.validation_loader,
    )
    checkpoint_path = Path(_checkpoint_callback(trainer).best_model_path)
    if not checkpoint_path.is_file():
        raise RuntimeError("training completed without checkpoints/best.ckpt")

    artifact = evaluate_checkpoint(
        config,
        components,
        checkpoint_path,
        output_dir,
        devices=args.devices,
        run_kind="smoke" if args.fast_dev_run else "full",
        trainer=trainer,
    )
    log.info(
        "Run complete: val mIoU=%.4f, test mIoU=%.4f, artifacts=%s",
        artifact["validation"]["miou"],
        artifact["test"]["miou"],
        output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
