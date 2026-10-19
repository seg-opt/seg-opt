#!/usr/bin/env bash
# One-batch dependency and pipeline check for the SegFormer-B4 to B0 KD sweep.
# The full array is submitted with an afterok dependency on this job.
#SBATCH --job-name=seg-b4-b0-kd-s4-smoke
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
    --config experiments/logit_distillation/segformer_b0_from_segformer_b4_s4_w1_t1.yaml \
    --output-dir results/logit_distillation/smoke/segformer_b4_to_segformer_b0_s4/seed42 \
    --devices 1 \
    --fast-dev-run \
    --offline
