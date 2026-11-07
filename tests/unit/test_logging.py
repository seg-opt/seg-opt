from pathlib import Path
from unittest.mock import Mock

from scripts.train import _wandb_run_name
from src.utils import logging as logging_utils


def test_wandb_run_name_describes_experiment_config_and_seed():
    config_path = Path(
        "experiments/strong_teacher_distillation/fast_scnn_sard_uniform_b4_80e.yaml"
    )

    assert _wandb_run_name(config_path, 42) == (
        "strong_teacher_distillation-fast_scnn_sard_uniform_b4_80e-seed42"
    )


def test_wandb_logger_appends_slurm_job_id(monkeypatch, tmp_path):
    logger = Mock()
    wandb_logger = Mock(return_value=logger)
    monkeypatch.setattr(logging_utils, "WandbLogger", wandb_logger)
    monkeypatch.setenv("SLURM_JOB_ID", "9445033")

    result = logging_utils.make_wandb_logger(
        tmp_path,
        run_name="baselines_benchmark-pidnet_s_imagenet_80e-seed42",
    )

    assert result is logger
    wandb_logger.assert_called_once_with(
        project="seg-opt",
        name="baselines_benchmark-pidnet_s_imagenet_80e-seed42-9445033",
        save_dir=tmp_path / "wandb",
        log_model=False,
    )


def test_wandb_logger_appends_local_process_id(monkeypatch, tmp_path):
    wandb_logger = Mock()
    monkeypatch.setattr(logging_utils, "WandbLogger", wandb_logger)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(logging_utils.os, "getpid", lambda: 1234)

    logging_utils.make_wandb_logger(tmp_path, run_name="local-run")

    assert wandb_logger.call_args.kwargs["name"] == "local-run-local-1234"
