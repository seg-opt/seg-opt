#!/usr/bin/env bash
# Submit with either exported variables or explicit paths, for example:
# sbatch --account=<grant-id> scripts/train.sbatch \
#   --image-path /path/to/development.sif --data-path /path/to/datasets \
#   python -m scripts.train --config experiments/baselines_benchmark/fast_scnn.yaml --devices 1
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

image_path=${IMAGE_PATH:-}
data_path=${DATA_PATH:-}
container_python_path=
while (($#)); do
    case "$1" in
        --image-path)
            if (($# < 2)); then
                echo "error: --image-path requires a SIF path" >&2
                exit 2
            fi
            image_path=$2
            shift 2
            ;;
        --data-path)
            if (($# < 2)); then
                echo "error: --data-path requires a datasets directory" >&2
                exit 2
            fi
            data_path=$2
            shift 2
            ;;
        --python-path)
            if (($# < 2)); then
                echo "error: --python-path requires a container-visible directory" >&2
                exit 2
            fi
            container_python_path=$2
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

[[ -n $image_path ]] || {
    echo "error: provide --image-path or set IMAGE_PATH" >&2
    exit 2
}
[[ -n $data_path ]] || {
    echo "error: provide --data-path or set DATA_PATH" >&2
    exit 2
}
[[ -f $image_path ]] || { echo "error: image not found: $image_path" >&2; exit 1; }
[[ -d $data_path ]] || { echo "error: dataset directory not found: $data_path" >&2; exit 1; }

runtime=$(command -v singularity || command -v apptainer || true)
[[ -n $runtime ]] || { echo "error: Singularity/Apptainer is unavailable" >&2; exit 1; }

repo_path=${REPO_PATH:-$PWD}
cache_path=${CACHE_PATH:-$HOME/$SLURM_JOB_ACCOUNT/project_data/containers/seg-opt/development/cache/$USER}
output_path=${OUTPUT_PATH:-$repo_path/results}
job_tmp="${SLURM_TMPDIR:-/tmp}/seg-opt-$SLURM_JOB_ID"

[[ -d $repo_path ]] || { echo "error: repository not found: $repo_path" >&2; exit 1; }
mkdir -p "$cache_path" "$output_path" "$job_tmp"
trap 'rm -rf "$job_tmp"' EXIT

if (($# == 0)); then
    set -- python -m scripts.train \
        --config experiments/baselines_benchmark/fast_scnn.yaml \
        --devices 1
fi

echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Image:     $image_path"
echo "Project:   $repo_path"
echo "Dataset:   $data_path"
echo "Cache:     $cache_path"
echo "Output:    $output_path"
echo "Started:   $(date --iso-8601=seconds)"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
container_tmp="/tmp"
container_environment=(
    "SINGULARITYENV_TMPDIR=$container_tmp"
    "APPTAINERENV_TMPDIR=$container_tmp"
)
if [[ -n $container_python_path ]]; then
    container_environment+=("SINGULARITYENV_PYTHONPATH=$container_python_path")
fi
srun /usr/bin/env \
    "${container_environment[@]}" \
    "$runtime" exec --nv --pwd /workspace \
    --bind "$repo_path:/workspace:ro" \
    --bind "$data_path:/data:ro" \
    --bind "$data_path:/datasets:ro" \
    --bind "$cache_path:/cache" \
    --bind "$job_tmp:/tmp" \
    --bind "$output_path:/output" \
    --bind "$output_path:/workspace/results" \
    "$image_path" "$@"

echo "Finished: $(date --iso-8601=seconds)"
