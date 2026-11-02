#!/usr/bin/env bash
# One-batch checks for the strong-teacher comparison.
#SBATCH --job-name=strong-teacher-smoke
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --array=0-1%1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-%x-%A_%a.out
#SBATCH --error=results/slurm-%x-%A_%a.err

set -euo pipefail

case "$SLURM_ARRAY_TASK_ID" in
    0) CONFIG="experiments/strong_teacher_distillation/segformer_b0_logit_kd.yaml"; NAME="segformer_b0_logit_kd" ;;
    1) CONFIG="experiments/strong_teacher_distillation/fast_scnn_logit_kd.yaml"; NAME="fast_scnn_logit_kd" ;;
    *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

export WANDB_MODE="offline"
exec bash scripts/train.sbatch \
    python -m scripts.train \
    --config "$CONFIG" \
    --output-dir "results/strong_teacher_distillation/smoke/${SLURM_ARRAY_JOB_ID}/${NAME}" \
    --devices 1 \
    --fast-dev-run \
    --offline
