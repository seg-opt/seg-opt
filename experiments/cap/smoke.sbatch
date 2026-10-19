#!/usr/bin/env bash
# One-batch checks for both B4-to-B0 control and CAP training paths.
# The two full jobs are gated on this job with afterok.
#SBATCH --job-name=segformer-cap-smoke
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

export WANDB_MODE="offline"

bash scripts/train.sbatch \
    python -m scripts.train \
    --config experiments/cap/segformer_b0_from_segformer_b4_no_cap.yaml \
    --output-dir "results/cap/smoke/${SLURM_JOB_ID}/no_cap" \
    --devices 1 \
    --fast-dev-run \
    --offline

exec bash scripts/train.sbatch \
    python -m scripts.train \
    --config experiments/cap/segformer_b0_from_segformer_b4_cap.yaml \
    --output-dir "results/cap/smoke/${SLURM_JOB_ID}/cap" \
    --devices 1 \
    --fast-dev-run \
    --offline
