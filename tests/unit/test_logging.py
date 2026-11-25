from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.train import _wandb_run_config, _wandb_run_name
from src.utils import logging as logging_utils


def test_wandb_logger_appends_slurm_job_id(monkeypatch, tmp_path):
    logger = Mock()
    wandb_logger = Mock(return_value=logger)
    monkeypatch.setattr(logging_utils, "WandbLogger", wandb_logger)
    monkeypatch.setenv("SLURM_JOB_ID", "9445033")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)

    result = logging_utils.make_wandb_logger(
        tmp_path,
        run_name=_wandb_run_name({"teacher": "segformer_b4", "student": "segformer_b2", "seed": 42}),
    )

    assert result is logger
    wandb_logger.assert_called_once_with(
        project="seg-opt",
        name="teacher-segformer_b4-student-segformer_b2-seed42-9445033",
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


@pytest.mark.parametrize("method", ["supervised", "sard"])
def test_wandb_routing_and_tags(monkeypatch, tmp_path, method):
    distillation = None
    if method == "sard":
        distillation = SimpleNamespace(
            method=method, teacher=SimpleNamespace(config_path="teachers/lunar_b4.yaml")
        )
    config = SimpleNamespace(
        data=SimpleNamespace(dataset="lunar"),
        model=SimpleNamespace(name="segformer_b2"),
        training=SimpleNamespace(seed=42),
        distillation=distillation,
    )
    monkeypatch.setattr("scripts.train.load_config", lambda _: SimpleNamespace(
        model=SimpleNamespace(name="segformer_b4")
    ))
    run_config = _wandb_run_config(config, "full")
    assert run_config["seed"] == 42
    assert run_config["teacher"] == ("segformer_b4" if distillation else None)
    expected_teacher = "segformer_b4" if distillation else "none"
    assert _wandb_run_name(run_config) == f"teacher-{expected_teacher}-student-segformer_b2-seed42"

    logger = Mock()
    monkeypatch.setattr(logging_utils, "WandbLogger", logger)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_RUN_GROUP", raising=False)
    logging_utils.make_wandb_logger(tmp_path, run_config=run_config)
    options = logger.call_args.kwargs
    assert options["project"] == "seg-opt-lunar"
    assert options["group"] == f"lunar_{method}"
    tags = {"dataset:lunar", f"method:{method}", "student:segformer_b2", "run_kind:full"}
    if distillation:
        tags.add("teacher:segformer_b4")
    assert set(options["tags"]) == tags

    monkeypatch.setenv("WANDB_PROJECT", "custom-project")
    monkeypatch.setenv("WANDB_RUN_GROUP", "custom-group")
    logging_utils.make_wandb_logger(tmp_path, run_config=run_config)
    assert logger.call_args.kwargs["project"] == "custom-project"
    assert logger.call_args.kwargs["group"] == "custom-group"
    assert set(logger.call_args.kwargs["tags"]) == tags
