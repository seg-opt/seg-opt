#!/usr/bin/env bash
#SBATCH --job-name=dinov3-probe-v2
#SBATCH --account=pl1200-01
#SBATCH --partition=tesla
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

GRANT_DIR="$HOME/pl1200-01/project_data"
PROJECT_DIR="$GRANT_DIR/$USER/seg-opt"
CACHE_DIR="$PROJECT_DIR/.cache"
JOB_TMP="$HOME/grant_${SLURM_JOB_ACCOUNT}/scratch/$USER/$SLURM_JOB_ID"
CONFIG="experiments/dinov3_probe_v2/config.yaml"

mkdir -p \
    "$CACHE_DIR/huggingface" \
    "$CACHE_DIR/wandb" \
    "$PROJECT_DIR/results" \
    "$JOB_TMP"
export HF_HOME="$CACHE_DIR/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface/transformers"
export WANDB_CACHE_DIR="$CACHE_DIR/wandb/cache"
export WANDB_CONFIG_DIR="$CACHE_DIR/wandb/config"
export WANDB_DATA_DIR="$CACHE_DIR/wandb/data"
export WANDB_PROJECT="seg-opt"
export TMPDIR="$JOB_TMP"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$PROJECT_DIR"
source .venv/bin/activate

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun python -m experiments.dinov3_probe_v2.train \
    --config "$CONFIG" \
    --devices 1

echo "Finished: $(date --iso-8601=seconds)"
