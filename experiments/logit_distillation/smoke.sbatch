#!/usr/bin/env bash
# One-batch, H100 dependency and pipeline check for SegFormer-B4 logit KD.
# The full training submission is gated on this job with afterok.
#SBATCH --job-name=segformer-b4-kd-smoke
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
    --config experiments/logit_distillation/fast_scnn_from_segformer_b4.yaml \
    --output-dir results/logit_distillation/smoke/segformer_b4_to_fast_scnn/seed42 \
    --devices 1 \
    --fast-dev-run \
    --offline
