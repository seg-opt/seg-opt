import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


DISTILLATION = "experiments/logit_distillation"
REPO = Path(__file__).resolve().parents[2]
GUARD_SHA256 = "f633c97c9a90b06cbeeac2769eca1aa1472c6abdde6bec1bc83fbe9a79a62073"


@pytest.fixture(params=range(3))
def launcher(tmp_path, request):
    root = tmp_path / "source with spaces"
    for student in range(3):
        campaign = f"b4_to_b{student}_lunar_80e"
        shutil.copytree(REPO / DISTILLATION / campaign, root / DISTILLATION / campaign, dirs_exist_ok=True)
    for name in ("b4_lunar_80e.sh", "check_wandb.py"):
        shared = REPO / DISTILLATION / name
        if shared.exists():
            shutil.copy2(shared, root / DISTILLATION)
    (root / "scripts").mkdir()
    record = 'import json, os, sys; print(json.dumps([sys.argv[1:], dict(os.environ)]))'
    (root / "scripts/train.sbatch").write_text(f"'{sys.executable}' -c '{record}' \"$@\"\n")
    (root / "bin").mkdir()
    sbatch = root / "bin/sbatch"
    sbatch.write_text(f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
                      "p = Path('calls'); rows = p.read_text().splitlines() if p.exists() else []\n"
                      "p.write_text('\\n'.join([*rows, json.dumps(sys.argv[1:])]) + '\\n')\nprint(101 + len(rows))\n")
    sbatch.chmod(0o755)
    (root / "image.sif").touch()
    teacher = root / "results/sard/teacher_segformer_b4/checkpoints/best.ckpt"
    teacher.parent.mkdir(parents=True)
    teacher.touch()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("WANDB_", "SLURM_", "SBATCH_"))}
    env.update(PATH=f"{root / 'bin'}:{os.environ['PATH']}", REPO_PATH=str(root), CAMPAIGN_ID="camp1",
               IMAGE_PATH=str(root / "image.sif"), DATA_PATH=str(root), OUTPUT_PATH=str(root / "results"),
               WANDB_API_KEY="test-key", WANDB_ENTITY="ogrodowskijedrzej-put")
    return root, f"b4_to_b{request.param}_lunar_80e", request.param, env

def invoke(launcher, script, **env):
    root, campaign, _, defaults = launcher
    return subprocess.run(["bash", f"{DISTILLATION}/{campaign}/{script}"], cwd=root,
                          env={**defaults, **env}, text=True, capture_output=True, timeout=10)

def test_slurm_headers(launcher):
    root, campaign, student, _ = launcher
    text = (root / DISTILLATION / campaign / "train.sbatch").read_text()
    assert [line for line in text.splitlines() if line.startswith("#SBATCH")] == [
        f"#SBATCH --{value}" for value in (f"job-name=b4-b{student}-lunar-80e", "account=pl1200-01",
        "partition=proxima", "constraint=h100", "gpus-per-node=1", "cpus-per-task=8", "mem=128G", "time=24:00:00")]
    text += "".join(p.read_text() for p in (root / DISTILLATION).glob("b4_lunar_80e.sh"))
    assert "experiment.py" not in text
    assert "scripts/train.sbatch" in text


def test_smoke_precedes_array_and_campaign_cannot_be_reused(launcher):
    root, campaign, student, _ = launcher
    result = invoke(launcher, "submit.sh")
    assert result.returncode == 0, result.stderr
    directory = root / "results/logit_distillation" / campaign / "camp1"
    target = f"{DISTILLATION}/{campaign}/train.sbatch"
    assert [json.loads(line) for line in (root / "calls").read_text().splitlines()] == [
        ["--parsable", "--export=ALL,CAMPAIGN_SMOKE=1", "--time=00:30:00", f"--job-name=b4-b{student}-online-smoke",
         f"--output={directory}/smoke-%j.out", f"--error={directory}/smoke-%j.err", target],
        ["--parsable", "--export=ALL,CAMPAIGN_SMOKE=0", "--array=0-2%3", "--dependency=afterok:101",
         f"--output={directory}/train-%A_%a.out", f"--error={directory}/train-%A_%a.err", target]]
    assert result.stdout == f"Campaign: camp1\nSmoke: 101\nFull array: 102\nResults: {directory}\n"
    assert (directory / "submission.txt").read_text() == "campaign_id=camp1\nsmoke_job=101\narray_job=102\n"
    assert invoke(launcher, "submit.sh").returncode != 0

def test_invalid_campaign_id(launcher):
    result = invoke(launcher, "submit.sh", CAMPAIGN_ID="bad/id")
    assert result.returncode == 2
    assert result.stderr == "CAMPAIGN_ID must contain only letters, digits, underscores, or hyphens\n"


@pytest.mark.parametrize("index,smoke", [("0", False), ("1", False), ("2", False), (None, True)])
def test_training_arguments_and_container_environment(launcher, index, smoke):
    root, campaign, student, _ = launcher
    result = invoke(launcher, "train.sbatch", **({"CAMPAIGN_SMOKE": "1"} if smoke else {"SLURM_ARRAY_TASK_ID": index}))
    assert result.returncode == 0, result.stderr
    args, env = json.loads(result.stdout)
    guard = f"{DISTILLATION}/{campaign}/check_wandb.py"
    seed = 42 + int(index or 0)
    assert args == ["bash", "-c", f'python {guard} && exec python -m scripts.train "$@"', "_",
                    "--config", f"{DISTILLATION}/{campaign}/seed{seed}.yaml", "--output-dir",
                    f"results/logit_distillation/{campaign}/camp1/" + ("smoke" if smoke else f"seed{seed}"),
                    "--devices", "1", *(["--fast-dev-run"] if smoke else [])]
    assert hashlib.sha256((root / guard).read_bytes()).hexdigest() == GUARD_SHA256
    for name, value in {"WANDB_MODE": "online", "WANDB_PROJECT": "seg-opt", "WANDB_ENTITY": "ogrodowskijedrzej-put",
                        "WANDB_RUN_GROUP": f"b4-to-b{student}-lunar-80e-camp1" + ("-smoke" if smoke else "")}.items():
        assert all(env[prefix + name] == value for prefix in ("", "SINGULARITYENV_", "APPTAINERENV_"))

def test_unexpected_array_index(launcher):
    result = invoke(launcher, "train.sbatch", SLURM_ARRAY_TASK_ID="3")
    assert result.returncode == 2
    assert result.stderr == "Unexpected array index\n"

def test_wandb_guard_rejects_offline():
    for guard in (REPO / DISTILLATION).rglob("check_wandb.py"):
        result = subprocess.run([sys.executable, str(guard)], env={**os.environ, "WANDB_MODE": "offline"},
                                text=True, capture_output=True, timeout=30)
        assert result.returncode != 0
        assert "This campaign requires WANDB_MODE=online" in result.stderr
