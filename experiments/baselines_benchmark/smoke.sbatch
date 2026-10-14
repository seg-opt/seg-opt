#!/usr/bin/env bash
# One-batch dependency and pipeline check for the active benchmark.
#SBATCH --job-name=seg-clean-bench-smoke
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
    --config experiments/baselines_benchmark/fast_scnn.yaml \
    --devices 1 \
    --fast-dev-run \
    --offline
