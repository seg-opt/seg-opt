#!/usr/bin/env bash
# SARD comparison on the SegFormer-B4 teacher: task-only control, vanilla
# logit KD, uniform SARD (W = 1) and full structure-aware SARD.
#SBATCH --job-name=sard-fastscnn
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --array=0-3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm-%x-%A_%a.out
#SBATCH --error=results/slurm-%x-%A_%a.err

set -euo pipefail

case "$SLURM_ARRAY_TASK_ID" in
    0) CONFIG="experiments/baselines_benchmark/fast_scnn.yaml" ;;
    1) CONFIG="experiments/logit_distillation/fast_scnn_from_segformer_b4.yaml" ;;
    2) CONFIG="experiments/sard/fast_scnn_sard_uniform_from_segformer_b4.yaml" ;;
    3) CONFIG="experiments/sard/fast_scnn_sard_full_from_segformer_b4.yaml" ;;
    *)
        echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2
        exit 1
        ;;
esac

export WANDB_MODE="offline"

exec bash scripts/train.sbatch \
    python -m scripts.train \
    --config "$CONFIG" \
    --devices 1 \
    --offline
