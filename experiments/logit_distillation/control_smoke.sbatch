#!/usr/bin/env bash
# One-batch pipeline check for the batch-8 Fast-SCNN task-only control.
#SBATCH --job-name=fast-scnn-b8-control-smoke
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

export WANDB_MODE="offline"

exec bash scripts/train.sbatch \
    python -m scripts.train \
    --config experiments/logit_distillation/fast_scnn_batch8_control.yaml \
    --output-dir results/logit_distillation/smoke/fast_scnn_batch8_control/seed42 \
    --devices 1 \
    --fast-dev-run \
    --offline
