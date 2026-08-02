from __future__ import annotations

import logging
import os
from pathlib import Path

from lightning.pytorch.loggers import WandbLogger


def configure_logging(output_dir: str | Path, verbose: bool = False) -> logging.Logger:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("seg_opt")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def make_wandb_logger(output_dir: str | Path) -> WandbLogger:
    output_dir = Path(output_dir)
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    job_id = os.getenv("SLURM_JOB_ID")
    run_name = output_dir.name if job_id is None else f"{output_dir.name}-{job_id}"
    return WandbLogger(
        project=os.getenv("WANDB_PROJECT", "seg-opt"),
        name=run_name,
        save_dir=wandb_dir,
        log_model=False,
    )
