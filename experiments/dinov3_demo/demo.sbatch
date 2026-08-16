#!/usr/bin/env bash
#SBATCH --job-name=dinov3-demo
#SBATCH --account=pl1200-01
#SBATCH --partition=tesla
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

GRANT_DIR="$HOME/pl1200-01/project_data"
PROJECT_DIR="$GRANT_DIR/$USER/seg-opt"
CACHE_DIR="$PROJECT_DIR/.cache"
JOB_TMP="$HOME/grant_${SLURM_JOB_ACCOUNT}/scratch/$USER/$SLURM_JOB_ID"

mkdir -p "$CACHE_DIR/huggingface" "$PROJECT_DIR/results" "$JOB_TMP"
export HF_HOME="$CACHE_DIR/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface/transformers"
export TMPDIR="$JOB_TMP"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$PROJECT_DIR"
source .venv/bin/activate

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun python -m experiments.dinov3_demo.demo \
    --device cuda \
    --train-images 128 \
    --eval-images 32 \
    --epochs 100 \
    --batch-size 16

echo "Finished: $(date --iso-8601=seconds)"
