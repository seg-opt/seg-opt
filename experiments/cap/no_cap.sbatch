#!/usr/bin/env bash
# B4-to-B0 vanilla logit-KD control without CAP.
# Submit through submit.sh so this starts only after the smoke job succeeds.
#SBATCH --job-name=segformer-b4-b0-no-cap
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

export WANDB_MODE="online"
export WANDB_PROJECT="${WANDB_PROJECT:-seg-opt-cap}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-segformer-b4-to-b0-cap-ablation-seed42}"

exec bash scripts/train.sbatch \
    python -m scripts.train \
    --config experiments/cap/segformer_b0_from_segformer_b4_no_cap.yaml \
    --devices 1
