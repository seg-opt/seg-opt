#!/usr/bin/env bash
# Four-run vanilla KD sweep: spatial stride 4, weight {1, 4}, temperature {1, 2}.
# Submit through segformer_b0_from_segformer_b4_s4_sweep_submit.sh.
#SBATCH --job-name=seg-b4-b0-kd-s4
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --array=0-3%1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=results/slurm-%x-%A_%a.out
#SBATCH --error=results/slurm-%x-%A_%a.err

set -euo pipefail

case "$SLURM_ARRAY_TASK_ID" in
    0) CONFIG="experiments/logit_distillation/segformer_b0_from_segformer_b4_s4_w1_t1.yaml" ;;
    1) CONFIG="experiments/logit_distillation/segformer_b0_from_segformer_b4_s4_w1_t2.yaml" ;;
    2) CONFIG="experiments/logit_distillation/segformer_b0_from_segformer_b4_s4_w4_t1.yaml" ;;
    3) CONFIG="experiments/logit_distillation/segformer_b0_from_segformer_b4_s4_w4_t2.yaml" ;;
    *)
        echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2
        exit 1
        ;;
esac

export WANDB_MODE="online"
export WANDB_PROJECT="${WANDB_PROJECT:-seg-opt-logit-distillation}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-segformer-b4-to-segformer-b0-s4-vanilla-sweep-seed42}"

exec bash scripts/train.sbatch \
    python -m scripts.train \
    --config "$CONFIG" \
    --devices 1
