from pathlib import Path
import re
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
    f"{DISTILLATION}/submit.sh": [DEFAULT_CONFIG],
    f"{DISTILLATION}/segformer_b0_from_segformer_b4_submit.sh": [B0_CONFIG],
    f"{DISTILLATION}/train.sbatch": [DEFAULT_CONFIG],
    f"{DISTILLATION}/smoke.sbatch": [DEFAULT_CONFIG],
    f"{DISTILLATION}/segformer_b0_from_segformer_b4_train.sbatch": [B0_CONFIG],
    f"{DISTILLATION}/segformer_b0_from_segformer_b4_smoke.sbatch": [B0_CONFIG],
    f"{BASELINES}/train.sbatch": BASELINE_CONFIGS,
    f"{BASELINES}/smoke.sbatch": [f"{BASELINES}/fast_scnn.yaml"],
    f"{BPKD}/train.sbatch": BPKD_CONFIGS,
    f"{BPKD}/train_mask2former_screen.sbatch": [
        f"{BPKD}/mask2former_vanilla_clean.yaml", f"{BPKD}/mask2former_bpkd_w7_clean.yaml",
    ],
}


class ExperimentSubmissionTests(unittest.TestCase):
    def test_family_launchers_keep_configs_and_wandb_defaults(self):
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
        for relative, configs in WRAPPERS.items():
            with self.subTest(script=relative):
                text = (REPOSITORY / relative).read_text(encoding="utf-8")
                self.assertNotIn("experiment.py", text)
                if relative.endswith("submit.sh"):
                    scripts = re.findall(r'"\$experiment_dir/([^\"]+\.sbatch)"', text)
                    prefix = Path(relative).name.removesuffix("submit.sh")
                    self.assertEqual(scripts, [prefix + "smoke.sbatch", prefix + "train.sbatch"])
                    self.assertEqual(text.count("sbatch --parsable"), 2)
                    self.assertIn('--dependency="afterok:${smoke_job}"', text)
                    text = "\n".join((REPOSITORY / relative).with_name(name).read_text()
                                     for name in scripts)
                else:
                    self.assertIn("exec bash scripts/train.sbatch", text)
                    self.assertIn("#SBATCH --account=pl1200-01", text)
                    self.assertIn("#SBATCH --partition=proxima", text)
                    self.assertIn("#SBATCH --gpus-per-node=1", text)
                    if len(configs) > 1:
                        self.assertIn(f"#SBATCH --array=0-{len(configs) - 1}%1", text)
                    mode = "offline" if relative.endswith("smoke.sbatch") else "online"
                    self.assertIn(f'export WANDB_MODE="{mode}"', text)
                    self.assertEqual("--fast-dev-run" in text, mode == "offline")
                self.assertEqual(list(dict.fromkeys(re.findall(r"experiments/[\w/]+\.yaml", text))),
                                 configs)
                expected = defaults.get(relative.replace("submit.sh", "train.sbatch"))
                for name, value in zip(("PROJECT", "RUN_GROUP"), expected or (None, None), strict=True):
                    matches = re.findall(rf'export WANDB_{name}="\$\{{WANDB_{name}:-([^}}]+)\}}"', text)
                    self.assertEqual(matches, [value] if value else [])


if __name__ == "__main__":
    unittest.main()
