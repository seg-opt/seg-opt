#!/usr/bin/env bash
#SBATCH --job-name=seg-opt-profile
#SBATCH --account=pl1200-01
#SBATCH --partition=tesla
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

# Keep code, artifacts, and caches out of the small login-node home filesystem.
GRANT_DIR="$HOME/pl1200-01/project_data"
PROJECT_DIR="$GRANT_DIR/$USER/seg-opt"
CACHE_DIR="$PROJECT_DIR/.cache"
JOB_TMP="$HOME/grant_${SLURM_JOB_ACCOUNT}/scratch/$USER/$SLURM_JOB_ID"

CONFIG="experiments/baseline_resnet34/config.yaml"
PROFILER_DIR="$JOB_TMP/profiler"

mkdir -p \
    "$CACHE_DIR/huggingface" \
    "$CACHE_DIR/torch" \
    "$CACHE_DIR/wandb" \
    "$CACHE_DIR/xdg" \
    "$PROJECT_DIR/results" \
    "$PROFILER_DIR" \
    "$JOB_TMP"

# Keep framework and model downloads on grant storage.
export HF_HOME="$CACHE_DIR/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface/transformers"
export TORCH_HOME="$CACHE_DIR/torch"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export WANDB_CACHE_DIR="$CACHE_DIR/wandb/cache"
export WANDB_CONFIG_DIR="$CACHE_DIR/wandb/config"
export WANDB_DATA_DIR="$CACHE_DIR/wandb/data"
export WANDB_PROJECT="seg-opt"
export TMPDIR="$JOB_TMP"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "Project directory not found: $PROJECT_DIR" >&2
    exit 1
fi
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    echo "Virtual environment not found: $PROJECT_DIR/.venv" >&2
    echo "Run 'uv sync --frozen' in the project directory first." >&2
    exit 1
fi
if [[ ! -d "$GRANT_DIR/datasets/artificial_lunar_landscape" ]]; then
    echo "Dataset not found under $GRANT_DIR/datasets" >&2
    exit 1
fi

cd "$PROJECT_DIR"
source .venv/bin/activate

echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Project:   $PROJECT_DIR"
echo "Python:    $(command -v python)"
echo "Cache:     $CACHE_DIR"
echo "Profiler:  $PROFILER_DIR"
echo "Started:   $(date --iso-8601=seconds)"
nvidia-smi

srun python -m scripts.train \
    --config "$CONFIG" \
    --devices 1 \
    --profile \
    --profiler-dir "$PROFILER_DIR"

echo "Finished: $(date --iso-8601=seconds)"
