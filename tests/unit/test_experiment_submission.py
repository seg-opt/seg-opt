import json

import pytest

from src.utils.utils import load_config
from tests.unit.test_submit_hydra import launcher, run_job, submit_command


SUPERVISED_FAST_SCNN = ("dataset=lunar", "method=supervised", "teacher=none", "student=fast_scnn")


def test_default_output_is_checkout_results(launcher):
    repo, _, _, env = launcher
    env.pop("OUTPUT_PATH")
    result = submit_command(launcher, "--smoke-only", *SUPERVISED_FAST_SCNN)
    assert result.returncode == 0, result.stderr
    run, = (repo / "results/experiments").iterdir()
    assert f"Results: {run}" in result.stdout


def test_external_dependency_gates_smoke_and_training(launcher):
    result = submit_command(launcher, "--after-job", "19", *SUPERVISED_FAST_SCNN)
    assert result.returncode == 0, result.stderr
    trace = launcher[2]
    assert "--dependency=afterok:19" in json.loads((trace / "101.json").read_text())
    assert "--dependency=afterok:101" in json.loads((trace / "102.json").read_text())


@pytest.mark.parametrize(
    ("option", "module", "flag", "dataset"),
    [
        ("--evaluate", "scripts.evaluate", "--checkpoint", "lusnar"),
        ("--init-checkpoint", "scripts.train", "--init-checkpoint", "lunar"),
    ],
)
def test_submit_selects_explicit_checkpoint_operation(launcher, option, module, flag, dataset):
    checkpoint = "/output/teacher.ckpt"
    result = submit_command(
        launcher, option, checkpoint, f"dataset={dataset}", "method=supervised",
        "teacher=none", "student=dinov3_vitl16_mask2former",
    )
    assert result.returncode == 0, result.stderr
    jobs = sorted(launcher[2].glob("*.sbatch"))
    assert len(jobs) == (1 if module == "scripts.evaluate" else 2)
    for job in jobs:
        arguments = run_job(launcher, int(job.stem))["args"]
        assert arguments[:2] == ["-m", module]
        assert arguments[arguments.index(flag) + 1] == checkpoint


def test_bpkd_control_and_distillation_use_separate_overrides(launcher):
    control = submit_command(launcher, "--full-only", *SUPERVISED_FAST_SCNN)
    assert control.returncode == 0, control.stderr
    checkpoint = "/output/new-teacher/checkpoints/best.ckpt"
    distilled = submit_command(
        launcher, "--full-only", "dataset=lunar", "method=bpkd", "teacher=bpkd_mask2former_swinl", "student=fast_scnn",
        f"distillation.teacher.checkpoint_path={checkpoint}",
    )
    assert distilled.returncode == 0, distilled.stderr
    assert load_config(run_job(launcher, 101)["args"][3]).distillation is None
    distillation = load_config(run_job(launcher, 102)["args"][3]).distillation
    assert distillation.method == "bpkd"
    assert distillation.teacher.checkpoint_path == checkpoint
