#!/usr/bin/env bash
# Train one config: sbatch experiments/sard/run.sbatch <config.yaml>
#SBATCH --job-name=sard-run
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

CONFIG=${1:?usage: sbatch run.sbatch <config.yaml>}

export WANDB_MODE="offline"

exec bash scripts/train.sbatch \
    python -m scripts.train \
    --config "$CONFIG" \
    --devices 1 \
    --offline
