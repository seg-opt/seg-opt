from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

from scripts.compose_experiment import compose_experiment, write_configs
from src.utils.utils import load_config


ROOT = Path(__file__).resolve().parents[2]
OVERRIDES = ("dataset=lunar", "method=vanilla", "teacher=lunar_segformer_b4")


def load_hydra_config(tmp_path, *overrides):
    path = Path(tempfile.mkdtemp(dir=tmp_path)) / "config.yaml"
    write_configs(*compose_experiment(overrides), path)
    return load_config(path)


@pytest.fixture
def launcher(tmp_path):
    repo = tmp_path / "checkout with spaces"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("submit", "compose_experiment.py"):
        shutil.copy(ROOT / "scripts" / name, scripts)
    shutil.copytree(ROOT / "experiments/hydra", repo / "experiments/hydra")
    (scripts / "train.sbatch").write_text(
        'set -eu\nexec "$@"\n'
    )
    commands = tmp_path / "commands"
    commands.mkdir()
    (commands / "uv").write_text(
        '#!/usr/bin/env bash\n'
        'set -eu\n'
        '[[ "$1 $2 $3 $4" == "run --only-group submit python" ]]\n'
        'shift 4\n'
        'printf "compose\\n" >> "$TRACE/uv.log"\n'
        'exec "$TEST_PYTHON" "$@"\n'
    )
    (commands / "sbatch").write_text(
        '#!/usr/bin/env python3\n'
        'import json, os, sys\n'
        'from pathlib import Path\n'
        'trace = Path(os.environ["TRACE"])\n'
        'job_id = 101 + len(list(trace.glob("*.sbatch")))\n'
        '(trace / f"{job_id}.sbatch").write_text(sys.stdin.read())\n'
        '(trace / f"{job_id}.json").write_text(json.dumps(sys.argv[1:]))\n'
        'print(f"{job_id};eagle")\n'
    )
    (commands / "python").write_text(
        '#!/usr/bin/env python3\n'
        'import json, os, sys\n'
        'from pathlib import Path\n'
        'if sys.argv[1:3] == ["-m", "scripts.compose_experiment"]:\n'
        '    os.execv(os.environ["TEST_PYTHON"], [os.environ["TEST_PYTHON"], *sys.argv[1:]])\n'
        'config = Path(sys.argv[sys.argv.index("--config") + 1])\n'
        'print(json.dumps({"args": sys.argv[1:], "yaml": config.read_text(), "env": {\n'
        '    key: os.environ.get(key) for key in ("WANDB_MODE", "SINGULARITYENV_WANDB_MODE",\n'
        '        "APPTAINERENV_WANDB_MODE", "WANDB_PROJECT", "WANDB_RUN_GROUP")}}))\n'
    )
    for command in commands.iterdir():
        command.chmod(0o755)
    image = tmp_path / "image.sif"
    image.touch()
    output = tmp_path / "run outputs"
    job_tmp = tmp_path / "job tmp"
    job_tmp.mkdir()
    trace = tmp_path / "trace"
    trace.mkdir()
    env = {
        **os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "TRACE": str(trace), "OUTPUT_PATH": str(output), "IMAGE_PATH": str(image),
        "DATA_PATH": str(tmp_path), "TEST_PYTHON": sys.executable, "TMPDIR": str(job_tmp),
    }
    for option in ("WANDB_MODE", "WANDB_PROJECT", "WANDB_RUN_GROUP"):
        env.pop(option, None)
    return repo, output, trace, env


def submit(launcher, *overrides):
    return submit_command(launcher, *OVERRIDES, *overrides)


def submit_command(launcher, *args):
    repo, _, trace, env = launcher
    return subprocess.run(
        [str(repo / "scripts/submit"), *args],
        cwd=trace, env=env, text=True, capture_output=True, timeout=20,
    )


def run_job(launcher, job_id):
    repo, _, trace, env = launcher
    result = subprocess.run(
        ["bash", str(trace / f"{job_id}.sbatch")], cwd=repo, env=env,
        check=True, text=True, capture_output=True, timeout=20,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("student", ["b0", "b1", "b2"])
def test_lunar_distillation_defaults(launcher, student):
    submission = submit(launcher, f"student=segformer_{student}")
    assert submission.returncode == 0, submission.stderr
    assert not list(Path(launcher[3]["TMPDIR"]).iterdir())
    config = load_config(Path(run_job(launcher, 101)["args"][3]))

    assert config.model.name == f"segformer_{student}"
    assert config.model.model_id == f"nvidia/mit-{student}"
    assert config.model.pretrained and not config.model.freeze_backbone
    assert Path(config.data.dataset_root) == (ROOT / "../../datasets/artificial_lunar_landscape").resolve()
    assert Path(config.data.split_manifest) == (ROOT / "experiments/baselines_benchmark/split_seed42_aligned.json")
    assert config.data.num_workers == 8
    assert config.data.class_names == ("background", "sky", "small_rock", "large_rock")
    assert (config.data.image_height, config.data.image_width) == (512, 768)
    assert config.data.mask_variant == "clean"
    assert asdict(config.data.augmentation) == {
        "flip_probability": 0.5, "brightness": 0.15, "contrast": 0.15,
        "gamma_min": 0.85, "gamma_max": 1.15,
    }
    assert asdict(config.loss) == {
        "cross_entropy_weight": 1.0, "dice_weight": 0.5, "ignore_index": 255,
    }
    assert config.metrics.trimap_kernel_size == 7
    training = asdict(config.training)
    training.pop("output_dir")
    assert training == {
        "epochs": 80, "batch_size": 8, "accumulate_grad_batches": 2,
        "learning_rate": 6.0e-4, "backbone_learning_rate": 6.0e-5,
        "weight_decay": 0.05, "warmup_fraction": 0.05, "precision": "bf16",
        "logging_steps": 20, "seed": 42,
    }
    kd = config.distillation
    assert kd.method == "vanilla"
    assert kd.spatial_stride == 8
    assert kd.vanilla_weight == kd.vanilla_temperature == 1.0
    assert load_config(kd.teacher.config_path).model.name == "segformer_b4"
    assert Path(kd.teacher.checkpoint_path) == ROOT / "results/sard/teacher_segformer_b4/checkpoints/best.ckpt"


def test_submit_composes_once_and_uses_slurm_dependency(launcher):
    repo, output, trace, env = launcher
    submission = submit(
        launcher, "student=segformer_b2", "training.seed=43", "training.learning_rate=0.001",
    )
    assert submission.returncode == 0, submission.stderr
    assert (trace / "uv.log").read_text() == "compose\n"
    assert len(list(trace.glob("*.sbatch"))) == 2
    assert not list(Path(env["TMPDIR"]).iterdir())
    run, = (output / "experiments").iterdir()
    assert not list(run.iterdir())
    assert submission.stdout.splitlines() == ["Smoke job: 101", f"Results: {run}", "Training job: 102"]

    smoke_options = json.loads((trace / "101.json").read_text())
    train_options = json.loads((trace / "102.json").read_text())
    assert f"--chdir={repo}" in smoke_options and f"--chdir={repo}" in train_options
    assert "--time=00:30:00" in smoke_options and "--mem=64G" in smoke_options
    assert not any(option.startswith("--dependency") for option in smoke_options)
    assert "--time=3-00:00:00" in train_options and "--mem=128G" in train_options
    assert "--dependency=afterok:101" in train_options
    smoke = run_job(launcher, 101)
    train = run_job(launcher, 102)

    smoke_config = load_config(Path(smoke["args"][3]))
    config = load_config(Path(train["args"][3]))

    smoke_values, train_values = asdict(smoke_config), asdict(config)
    for values in (smoke_values, train_values):
        values["distillation"]["teacher"].pop("config_path")
    assert smoke_values == train_values
    assert Path(smoke_config.distillation.teacher.config_path).read_bytes() == Path(config.distillation.teacher.config_path).read_bytes()
    assert config.training.seed == 43
    assert config.training.learning_rate == 0.001
    for job, mode, stage in ((smoke, "offline", "smoke"), (train, "online", "train")):
        assert {value for key, value in job["env"].items() if key.endswith("WANDB_MODE")} == {mode}
        assert job["args"][:3] == ["-m", "scripts.train", "--config"]
        assert Path(job["args"][3]).is_relative_to(env["TMPDIR"])
        assert job["args"][4:8] == ["--devices", "1", "--output-dir", f"/output/experiments/{run.name}/{stage}"]
    assert smoke["args"][8:] == ["--fast-dev-run"]
    assert train["args"][8:] == []


@pytest.mark.parametrize("overrides", [(), ("student=unknown",)])
def test_invalid_composition_never_submits(launcher, overrides):
    _, output, trace, _ = launcher
    submission = submit(launcher, *overrides)
    assert submission.returncode != 0
    assert not list(trace.glob("*.sbatch"))
    assert not output.exists()
