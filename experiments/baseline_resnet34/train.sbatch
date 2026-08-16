#!/usr/bin/env bash
# Submit with: sbatch --account=<grant-id> --export=ALL,IMAGE_PATH,DATA_PATH scripts/train.sbatch
#SBATCH --job-name=seg-opt-train
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=seg-opt-train-%j.out
#SBATCH --error=seg-opt-train-%j.err

set -euo pipefail

: "${IMAGE_PATH:?Set IMAGE_PATH to an immutable development SIF path}"
: "${DATA_PATH:?Set DATA_PATH to the grant-backed datasets directory}"

runtime=$(command -v singularity || command -v apptainer || true)
[[ -n $runtime ]] || { echo "error: Singularity/Apptainer is unavailable" >&2; exit 1; }

repo_path=${REPO_PATH:-$PWD}
cache_path=${CACHE_PATH:-$HOME/$SLURM_JOB_ACCOUNT/project_data/containers/seg-opt/development/cache/$USER}
output_path=${OUTPUT_PATH:-$repo_path/results}
job_tmp="${SLURM_TMPDIR:-/tmp}/seg-opt-$SLURM_JOB_ID"

[[ -d $repo_path ]] || { echo "error: repository not found: $repo_path" >&2; exit 1; }
[[ -d $DATA_PATH ]] || { echo "error: dataset directory not found: $DATA_PATH" >&2; exit 1; }
mkdir -p "$cache_path" "$output_path" "$job_tmp"
trap 'rm -rf "$job_tmp"' EXIT

if (($# == 0)); then
    set -- python -m scripts.train \
        --config experiments/baseline_resnet34/config.yaml \
        --devices 1
fi

echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Image:     $IMAGE_PATH"
echo "Project:   $repo_path"
echo "Dataset:   $DATA_PATH"
echo "Cache:     $cache_path"
echo "Output:    $output_path"
echo "Started:   $(date --iso-8601=seconds)"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
container_tmp="/tmp"
srun env \
    SINGULARITYENV_TMPDIR="$container_tmp" \
    APPTAINERENV_TMPDIR="$container_tmp" \
    "$runtime" exec --nv --pwd /workspace \
    --bind "$repo_path:/workspace:ro" \
    --bind "$DATA_PATH:/data:ro" \
    --bind "$DATA_PATH:/datasets:ro" \
    --bind "$cache_path:/cache" \
    --bind "$job_tmp:/tmp" \
    --bind "$output_path:/output" \
    --bind "$output_path:/workspace/results" \
    "$IMAGE_PATH" "$@"

echo "Finished: $(date --iso-8601=seconds)"
