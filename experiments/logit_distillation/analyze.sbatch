#!/usr/bin/env bash
# Deterministic paired teacher/control/KD error analysis on validation and test.
# Submit with: sbatch --export=ALL,IMAGE_PATH,DATA_PATH experiments/logit_distillation/analyze.sbatch
#SBATCH --job-name=segformer-b4-kd-analysis
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

export REPO_PATH="${SLURM_SUBMIT_DIR:?Slurm did not provide the submission directory}"
analysis_dir="results/logit_distillation/analysis/segformer_b4_fast_scnn_seed42"

exec bash scripts/train.sbatch \
    python -m scripts.analyze_distillation \
    --teacher-config experiments/baselines_benchmark/segformer_b4.yaml \
    --teacher-checkpoint results/baselines_benchmark/segformer_b4/seed42/checkpoints/best.ckpt \
    --control-config experiments/logit_distillation/fast_scnn_batch8_control.yaml \
    --control-checkpoint results/logit_distillation/fast_scnn_batch8_control/seed42/checkpoints/best.ckpt \
    --kd-config experiments/logit_distillation/fast_scnn_from_segformer_b4.yaml \
    --kd-checkpoint results/logit_distillation/fast_scnn_from_segformer_b4/seed42/checkpoints/best.ckpt \
    --output-dir "$analysis_dir" \
    --device cuda \
    --precision bf16
