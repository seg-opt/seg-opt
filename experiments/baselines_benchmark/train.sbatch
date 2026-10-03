#!/usr/bin/env bash
# Seven clean/aligned supervised baselines: Fast-SCNN plus six teacher candidates.
#SBATCH --job-name=seg-clean-bench
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --array=0-6%1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=results/slurm-%x-%A_%a.out
#SBATCH --error=results/slurm-%x-%A_%a.err

set -euo pipefail

case "$SLURM_ARRAY_TASK_ID" in
    0) CONFIG="experiments/baselines_benchmark/fast_scnn.yaml" ;;
    1) CONFIG="experiments/baselines_benchmark/resnet34_unet.yaml" ;;
    2) CONFIG="experiments/baselines_benchmark/segformer_b0.yaml" ;;
    3) CONFIG="experiments/baselines_benchmark/segformer_b2.yaml" ;;
    4) CONFIG="experiments/baselines_benchmark/segformer_b4.yaml" ;;
    5) CONFIG="experiments/baselines_benchmark/dinov3_vitl16.yaml" ;;
    6) CONFIG="experiments/baselines_benchmark/mask2former_swinl.yaml" ;;
    *)
        echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2
        exit 1
        ;;
esac

export WANDB_MODE="online"
export WANDB_PROJECT="${WANDB_PROJECT:-seg-opt-clean-benchmarks}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-clean-aligned-seed42}"

exec bash scripts/train.sbatch \
    python -m scripts.train \
    --config "$CONFIG" \
    --devices 1
