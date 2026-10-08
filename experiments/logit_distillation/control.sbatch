#!/usr/bin/env bash
# Full task-only control matched to the physical batch used for vanilla KD.
# Submit through submit_control.sh to gate this run on the smoke job.
#SBATCH --job-name=fast-scnn-b8-control
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
export WANDB_PROJECT="${WANDB_PROJECT:-seg-opt-logit-distillation}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-fast-scnn-batch8-control-clean-seed42}"

exec bash scripts/train.sbatch \
    python -m scripts.train \
    --config experiments/logit_distillation/fast_scnn_batch8_control.yaml \
    --devices 1
