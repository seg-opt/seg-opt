#!/usr/bin/env bash
# Full SegFormer-B4 teacher on clean masks; identical to the benchmark run so
# SARD and the logit-KD experiment share one teacher checkpoint.
#SBATCH --job-name=sard-teacher-segformer-b4
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

export WANDB_MODE="offline"

exec bash scripts/train.sbatch \
    python -m scripts.train \
    --config experiments/baselines_benchmark/segformer_b4.yaml \
    --devices 1 \
    --offline
