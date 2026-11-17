from pathlib import Path

import pytest


def test_historical_miou_deltas_document_unrounded_receipts():
    readme = Path("experiments/README.md").read_text(encoding="utf-8")
    historical_results = readme.split("## Completed Controlled CNN Results", 1)[1]

    assert "mIoU deltas use the unrounded receipt values" in historical_results


def test_baseline_teacher_candidates_refer_to_the_experiment_map():
    readme = Path("experiments/baselines_benchmark/README.md").read_text(
        encoding="utf-8"
    )
    introduction = readme.split("| Config |", 1)[0]

    assert "[experiment map](../README.md#clean-baseline-benchmark)" in introduction


def test_adopted_teacher_instructions_use_one_completion_barrier_job():
    readme = Path("experiments/README.md").read_text(encoding="utf-8")
    instructions = " ".join(readme.split("For further approved studies", 1)[1].split())

    assert "`pipeline --after-job JOB_ID`" in instructions
    assert "one successful completion barrier" in instructions
    assert "both the teacher and paired supervised baseline" in instructions


@pytest.mark.parametrize(
    ("relative", "reference"),
    [
        ("experiments/README.md", "results/canonical/"),
        ("experiments/baselines_benchmark/README.md", "--results-root results/canonical/baselines_benchmark"),
        ("experiments/bpkd/README.md", "results/canonical/bpkd/teachers/mask2former_swinl_clean/seed42/checkpoints/best.ckpt"),
        ("experiments/strong_teacher_distillation/README.md", "results/canonical/strong_teacher_distillation/teacher_dinov3_mask2former/seed42/checkpoints/best.ckpt"),
        ("experiments/sard/smoke.sbatch", "results/canonical/baselines_benchmark/segformer_b4/seed42/checkpoints/best.ckpt"),
        ("experiments/sard/submit.sh", "results/canonical/baselines_benchmark/segformer_b4/seed42"),
    ],
)
def test_future_run_guidance_uses_canonical_output_paths(relative, reference):
    assert reference in Path(relative).read_text(encoding="utf-8")
