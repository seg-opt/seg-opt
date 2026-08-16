#!/usr/bin/env bash
#SBATCH --job-name=fast-scnn
#SBATCH --account=pl1200-01
#SBATCH --partition=tesla
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

GRANT_DIR="$HOME/pl1200-01/project_data"
PROJECT_DIR="$GRANT_DIR/$USER/seg-opt"
CACHE_DIR="$PROJECT_DIR/.cache"
RUN_ATTEMPT="${SLURM_RESTART_COUNT:-0}"
JOB_TMP="$HOME/grant_${SLURM_JOB_ACCOUNT}/scratch/$USER/${SLURM_JOB_ID}-${RUN_ATTEMPT}"
SOURCE_ARCHIVE="$PROJECT_DIR/results/source_snapshots/fast_scnn_equal_recipe_v1_retry1.tar.gz"
SOURCE_SHA256="047e9a9b37d5ce2fc3c5fe42a13ec5ac3031e78ab7bb17660a73d1dd426b97bb"
RUN_ROOT="$JOB_TMP/project_data/$USER/seg-opt"
CONFIG="experiments/baseline_fast_scnn/config.yaml"

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
if [[ ! -f "$SOURCE_ARCHIVE" ]]; then
    echo "Source snapshot not found: $SOURCE_ARCHIVE" >&2
    exit 1
fi
if ! echo "$SOURCE_SHA256  $SOURCE_ARCHIVE" | sha256sum --check --status; then
    echo "Source snapshot checksum mismatch: $SOURCE_ARCHIVE" >&2
    exit 1
fi

LOCAL_TMP="$(mktemp -d -- "/tmp/seg-opt-${SLURM_JOB_ID}-${RUN_ATTEMPT}.XXXXXX")"
chmod 700 -- "$LOCAL_TMP"
mkdir -p \
    "$CACHE_DIR/torch" \
    "$CACHE_DIR/wandb" \
    "$CACHE_DIR/xdg" \
    "$PROJECT_DIR/results" \
    "$RUN_ROOT"
tar -xzf "$SOURCE_ARCHIVE" -C "$RUN_ROOT"
ln -s "$PROJECT_DIR/results" "$RUN_ROOT/results"
ln -s "$GRANT_DIR/datasets" "$JOB_TMP/project_data/datasets"

export TORCH_HOME="$CACHE_DIR/torch"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export WANDB_CACHE_DIR="$CACHE_DIR/wandb/cache"
export WANDB_CONFIG_DIR="$CACHE_DIR/wandb/config"
export WANDB_DATA_DIR="$CACHE_DIR/wandb/data"
export WANDB_PROJECT="seg-opt"
# No W&B credentials are configured on this account. Keep a local run history
# that can be uploaded later with `wandb sync` after authentication.
export WANDB_MODE="${WANDB_MODE:-offline}"
# Keep Python multiprocessing sockets below Linux's AF_UNIX path-length limit.
export TMPDIR="$LOCAL_TMP"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

cd "$RUN_ROOT"
source "$PROJECT_DIR/.venv/bin/activate"

echo "Job:      $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Source:   $SOURCE_ARCHIVE"
echo "Run root: $RUN_ROOT"
echo "Temp:     $TMPDIR"
echo "Python:   $(command -v python)"
echo "Started:  $(date --iso-8601=seconds)"
nvidia-smi

srun python -m scripts.train \
    --config "$CONFIG" \
    --devices 1

echo "Finished: $(date --iso-8601=seconds)"
