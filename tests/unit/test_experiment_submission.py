import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
DISTILLATION = "experiments/logit_distillation"
BASELINES = "experiments/baselines_benchmark"
DEFAULT_CONFIG = f"{DISTILLATION}/fast_scnn_from_segformer_b4.yaml"
B0_CONFIG = f"{DISTILLATION}/segformer_b0_from_segformer_b4.yaml"
BPKD = "experiments/bpkd"
BASELINE_CONFIGS = [f"{BASELINES}/{name}.yaml" for name in (
    "fast_scnn", "resnet34_unet", "segformer_b0", "segformer_b2",
    "segformer_b4", "dinov3_vitl16", "mask2former_swinl",
)]
BPKD_CONFIGS = [f"{BPKD}/fast_scnn_{name}_clean.yaml" for name in (
    "task_only", "vanilla", "body_only", "edge_only", "bpkd_w3",
    "bpkd_w5", "bpkd_w7", "bpkd_w9",
)]
WRAPPERS = {
    f"{DISTILLATION}/submit.sh": (["submit"], [DEFAULT_CONFIG]),
    f"{DISTILLATION}/segformer_b0_from_segformer_b4_submit.sh": (["submit"], [B0_CONFIG]),
    f"{DISTILLATION}/train.sbatch": (["launch"], [DEFAULT_CONFIG]),
    f"{DISTILLATION}/smoke.sbatch": (["launch", "--smoke"], [DEFAULT_CONFIG]),
    f"{DISTILLATION}/segformer_b0_from_segformer_b4_train.sbatch": (["launch"], [B0_CONFIG]),
    f"{DISTILLATION}/segformer_b0_from_segformer_b4_smoke.sbatch": (
        ["launch", "--smoke"], [B0_CONFIG]),
    f"{BASELINES}/train.sbatch": (["launch"], BASELINE_CONFIGS),
    f"{BASELINES}/smoke.sbatch": (["launch", "--smoke"], [f"{BASELINES}/fast_scnn.yaml"]),
    f"{BPKD}/train.sbatch": (["launch"], BPKD_CONFIGS),
    f"{BPKD}/train_mask2former_screen.sbatch": (["launch"], [
        f"{BPKD}/mask2former_vanilla_clean.yaml", f"{BPKD}/mask2former_bpkd_w7_clean.yaml",
    ]),
}


class ExperimentSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "source with spaces"
        self.root.mkdir()
        for relative in WRAPPERS:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPOSITORY / relative, target)
        self.configs = ["experiments/first.yaml", "experiments/second model.yaml"]
        for relative in set(self.configs + [
            config for _, configs in WRAPPERS.values() for config in configs
        ]):
            (self.root / relative).write_text("model: {}\n", encoding="utf-8")
        (self.root / "scripts").mkdir()
        (self.root / "scripts/train.sbatch").write_text(
            'printf "Legacy wrapper bypassed shared CLI\\n" >&2\nexit 97\n', encoding="utf-8"
        )
        (self.root / "scripts/experiment.py").write_text(
            "import json, os, sys\n"
            "print(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), "
            "'repo': os.environ.get('REPO_PATH'), "
            "'mode': os.environ.get('WANDB_MODE'), "
            "'project': os.environ.get('WANDB_PROJECT'), "
            "'group': os.environ.get('WANDB_RUN_GROUP'), "
            "'index': os.environ.get('SLURM_ARRAY_TASK_ID'), 'no_site': sys.flags.no_site}))\n"
            "if sys.argv[1] == 'submit':\n"
            "    print('Run: /unique/run\\nSmoke job: 101\\nFull job: 202')\n"
            "sys.exit(int(os.environ.get('SHARED_EXIT', '0')))\n",
            encoding="utf-8",
        )
        binary_directory = self.root / "bin"
        binary_directory.mkdir()
        for command in ("sbatch", "scancel"):
            binary = binary_directory / command
            binary.write_text(
                '#!/usr/bin/env bash\nprintf "Unexpected scheduler call\\n" >&2\nexit 98\n',
                encoding="utf-8",
            )
            binary.chmod(0o755)
        python = binary_directory / "python3"
        python.write_text(
            '#!/usr/bin/env bash\nexec ' + shlex.quote(sys.executable) + ' "$@"\n',
            encoding="utf-8",
        )
        python.chmod(0o755)
        self.environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SLURM_", "SBATCH_", "WANDB_"))
            and key not in {"REPO_PATH", "FAST_DEV_RUN", "IMAGE_PATH", "DATA_PATH"}
        }
        self.environment["PATH"] = f"{binary_directory}:{os.environ['PATH']}"

    def run_script(self, relative, *arguments, **environment):
        return subprocess.run(
            ["bash", str(self.root / relative), *arguments],
            cwd=self.root.parent,
            env={**self.environment, **environment},
            text=True,
            capture_output=True,
            timeout=10,
        )

    def payload(self, relative, *arguments, **environment):
        result = self.run_script(relative, *arguments, **environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout.splitlines()[0])
        self.assertEqual(payload["cwd"], str(self.root))
        self.assertEqual(payload["repo"], str(self.root))
        self.assertEqual(payload["no_site"], 1)
        return payload

    def test_defaults_and_explicit_configs_are_forwarded_exactly(self):
        for relative, (command, configs) in WRAPPERS.items():
            for arguments in ([], self.configs, [str(self.root / self.configs[1])]):
                with self.subTest(script=relative, arguments=arguments):
                    payload = self.payload(relative, *arguments)
                    self.assertEqual(payload["args"], command + (arguments or configs))

    def test_array_selection_and_validation_are_delegated(self):
        for relative, (command, configs) in WRAPPERS.items():
            if command[0] != "launch":
                continue
            for index in ("1", "-1", "2", "nonsense", "01"):
                with self.subTest(script=relative, index=index):
                    payload = self.payload(relative, SLURM_ARRAY_TASK_ID=index)
                    self.assertEqual(payload["args"], command + configs)
                    self.assertEqual(payload["index"], index)

    def test_all_spooled_sbatch_wrappers_resolve_real_repository(self):
        spool = self.root.parent / "spool"
        spool.mkdir()
        for relative, (command, _) in WRAPPERS.items():
            if not relative.endswith(".sbatch"):
                continue
            spooled_script = spool / "slurm_script"
            shutil.copy2(self.root / relative, spooled_script)
            for variable in ("REPO_PATH", "SLURM_SUBMIT_DIR"):
                with self.subTest(script=relative, variable=variable):
                    payload = self.payload(
                        str(spooled_script), *self.configs, **{variable: str(self.root)}
                    )
                    self.assertEqual(payload["args"], command + self.configs)

    def test_explicit_repository_takes_precedence(self):
        for relative, (command, _) in WRAPPERS.items():
            with self.subTest(script=relative):
                payload = self.payload(
                    relative, *self.configs, REPO_PATH=str(self.root),
                    SLURM_SUBMIT_DIR=str(self.root.parent / "not the repository"),
                )
                self.assertEqual(payload["args"], command + self.configs)

    def test_explicit_wandb_overrides_are_preserved(self):
        for relative in WRAPPERS:
            for mode in ("offline", "online", "disabled"):
                with self.subTest(script=relative, mode=mode):
                    payload = self.payload(
                        relative, WANDB_MODE=mode, WANDB_PROJECT="existing-project",
                        WANDB_RUN_GROUP="existing-group",
                    )
                    self.assertEqual(payload["mode"], mode)
                    self.assertEqual(payload["project"], "existing-project")
                    self.assertEqual(payload["group"], "existing-group")
                    self.assertNotIn("--offline", payload["args"])

    def test_unset_mode_delegates_to_shared_offline_default(self):
        for relative in WRAPPERS:
            with self.subTest(script=relative):
                payload = self.payload(relative)
                self.assertIsNone(payload["mode"])
                self.assertNotIn("--offline", payload["args"])
                self.assertNotIn("--wandb-mode", payload["args"])

    def test_existing_suite_project_and_group_defaults_are_retained(self):
        defaults = {
            f"{DISTILLATION}/train.sbatch": (
                "seg-opt-logit-distillation", "segformer-b4-to-fastscnn-clean-seed42"),
            f"{BASELINES}/train.sbatch": ("seg-opt-clean-benchmarks", "clean-aligned-seed42"),
            f"{BPKD}/train.sbatch": ("seg-opt-bpkd", "mask2former-to-fastscnn-clean-seed42"),
            f"{BPKD}/train_mask2former_screen.sbatch": (
                "seg-opt-bpkd", "mask2former-born-again-clean-seed42"),
            f"{DISTILLATION}/segformer_b0_from_segformer_b4_train.sbatch": (
                "seg-opt-logit-distillation", "segformer-b4-to-segformer-b0-clean-seed42"),
        }
        for relative, (project, group) in defaults.items():
            with self.subTest(script=relative):
                payload = self.payload(relative)
                self.assertEqual(payload["project"], project)
                self.assertEqual(payload["group"], group)

    def test_submitters_delegate_without_scheduler_or_required_path_env(self):
        for relative in WRAPPERS:
            if not relative.endswith("submit.sh"):
                continue
            with self.subTest(script=relative):
                result = self.run_script(relative, *self.configs, "--profile", "cluster profile.json")
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout.splitlines()[0])
                self.assertEqual(payload["args"], [
                    "submit", *self.configs, "--profile", "cluster profile.json",
                ])
                self.assertIn("Smoke job: 101", result.stdout)
                self.assertIn("Full job: 202", result.stdout)

    def test_shared_exit_status_is_propagated(self):
        for relative in WRAPPERS:
            with self.subTest(script=relative):
                self.assertEqual(self.run_script(relative, SHARED_EXIT="23").returncode, 23)

    def test_sbatch_resource_headers_remain_available(self):
        for relative in WRAPPERS:
            if relative.endswith(".sbatch"):
                with self.subTest(script=relative):
                    text = (self.root / relative).read_text(encoding="utf-8")
                    self.assertIn("#SBATCH --account=pl1200-01", text)
                    self.assertIn("#SBATCH --partition=proxima", text)
                    self.assertIn("#SBATCH --gpus-per-node=1", text)
        for relative, array in (
            (f"{BASELINES}/train.sbatch", "0-6%1"),
            (f"{BPKD}/train.sbatch", "0-7%1"),
            (f"{BPKD}/train_mask2former_screen.sbatch", "0-1%1"),
        ):
            with self.subTest(script=relative):
                self.assertIn(
                    f"#SBATCH --array={array}",
                    (self.root / relative).read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()