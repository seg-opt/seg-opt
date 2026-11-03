#!/usr/bin/env bash
# Controlled comparison against the existing B4-teacher runs.
#SBATCH --job-name=strong-teacher-kd
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --array=0-1%1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=results/slurm-%x-%A_%a.out
#SBATCH --error=results/slurm-%x-%A_%a.err

set -euo pipefail

case "$SLURM_ARRAY_TASK_ID" in
    0) CONFIG="experiments/strong_teacher_distillation/segformer_b0_logit_kd.yaml" ;;
    1) CONFIG="experiments/strong_teacher_distillation/fast_scnn_logit_kd.yaml" ;;
    *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

export WANDB_MODE="online"
export WANDB_PROJECT="${WANDB_PROJECT:-seg-opt-strong-teacher}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-dinov3-mask2former-kd-seed42}"
exec bash scripts/train.sbatch \
    python -m scripts.train \
    --config "$CONFIG" \
    --devices 1
