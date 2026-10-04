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
nsys_profile=${NSYS_PROFILE:-0}
nsys_delay_seconds=${NSYS_DELAY_SECONDS:-60}
nsys_duration_seconds=${NSYS_DURATION_SECONDS:-45}
ncu_profile=${NCU_PROFILE:-0}
ncu_launch_skip_before_match=${NCU_LAUNCH_SKIP_BEFORE_MATCH:-200}
ncu_launch_count=${NCU_LAUNCH_COUNT:-1}

[[ -d $repo_path ]] || { echo "error: repository not found: $repo_path" >&2; exit 1; }
mkdir -p "$cache_path" "$output_path" "$job_tmp"
trap 'rm -rf "$job_tmp"' EXIT

if [[ $nsys_profile != 0 && $nsys_profile != 1 ]]; then
    echo "error: NSYS_PROFILE must be 0 or 1" >&2
    exit 2
fi
if [[ $ncu_profile != 0 && $ncu_profile != 1 ]]; then
    echo "error: NCU_PROFILE must be 0 or 1" >&2
    exit 2
fi
if [[ $nsys_profile == 1 && $ncu_profile == 1 ]]; then
    echo "error: enable either NSYS_PROFILE or NCU_PROFILE, not both" >&2
    exit 2
fi
if [[ ! $nsys_delay_seconds =~ ^[0-9]+$ || ! $nsys_duration_seconds =~ ^[1-9][0-9]*$ ]]; then
    echo "error: NSYS_DELAY_SECONDS must be non-negative and NSYS_DURATION_SECONDS positive integers" >&2
    exit 2
fi
if [[ ! $ncu_launch_skip_before_match =~ ^[0-9]+$ || ! $ncu_launch_count =~ ^[1-9][0-9]*$ ]]; then
    echo "error: NCU_LAUNCH_SKIP_BEFORE_MATCH must be non-negative and NCU_LAUNCH_COUNT positive integers" >&2
    exit 2
fi

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
if [[ $nsys_profile == 1 ]]; then
    container_environment+=(
        "SINGULARITYENV_SEG_OPT_NVTX_RANGES=1"
        "APPTAINERENV_SEG_OPT_NVTX_RANGES=1"
    )
fi

container_binds=(
    --bind "$repo_path:/workspace:ro"
    --bind "$data_path:/data:ro"
    --bind "$data_path:/datasets:ro"
    --bind "$cache_path:/cache"
    --bind "$job_tmp:/tmp"
    --bind "$output_path:/output"
    --bind "$output_path:/workspace/results"
)

if [[ $nsys_profile == 1 ]]; then
    command -v nsys >/dev/null || {
        echo "error: nsys is unavailable on $(hostname); load the Nsight Systems module or install it on the compute node" >&2
        exit 1
    }
    nsys_executable=$(readlink -f "$(command -v nsys)")
    nsys_root=$(dirname "$nsys_executable")
    while [[ $nsys_root != / && ! -d $nsys_root/target-linux-x64 ]]; do
        nsys_root=$(dirname "$nsys_root")
    done
    if [[ $nsys_root == / ]]; then
        echo "error: could not locate the Nsight Systems installation for $nsys_executable" >&2
        exit 1
    fi
    # nsys injects libraries from this host path into the profiled process.
    # Bind it at the same path so those libraries are visible in the container.
    container_binds+=(--bind "$nsys_root:$nsys_root:ro")
fi
if [[ $ncu_profile == 1 ]]; then
    ncu_command=${NCU_EXECUTABLE:-}
    if [[ -z $ncu_command ]]; then
        ncu_command=$(command -v ncu || true)
    fi
    if [[ -z $ncu_command ]]; then
        for ncu_candidate in /opt/nvidia/nsight-compute/*/ncu; do
            if [[ -x $ncu_candidate ]]; then
                ncu_command=$ncu_candidate
                break
            fi
        done
    fi
    [[ -n $ncu_command ]] || {
        echo "error: ncu is unavailable on $(hostname); set NCU_EXECUTABLE or install Nsight Compute on the compute node" >&2
        exit 1
    }
    ncu_executable=$(readlink -f "$ncu_command")
    ncu_root=$(dirname "$ncu_executable")
    [[ -d $ncu_root/sections ]] || {
        echo "error: could not locate the Nsight Compute installation for $ncu_executable" >&2
        exit 1
    }
    # NCU needs its executable, sections, and injected libraries at the same
    # absolute path from inside the container.
    container_binds+=(--bind "$ncu_root:$ncu_root:ro")
fi

container_command=(
    "$runtime" exec --nv --pwd /workspace
    "${container_binds[@]}"
    "$image_path" "$@"
)

if [[ $nsys_profile == 1 ]]; then
    nsys_directory="$output_path/nsys"
    nsys_output_host="$nsys_directory/seg-opt-${SLURM_JOB_ID}"
    nsys_output_container="/output/nsys/seg-opt-${SLURM_JOB_ID}"
    mkdir -p "$nsys_directory"
    echo "Nsight:    ${nsys_output_host}.nsys-rep (delay ${nsys_delay_seconds}s, duration ${nsys_duration_seconds}s)"
    # Run nsys inside the container.  Wrapping the Singularity launcher on the
    # host traces that launcher process, not the Python process that uses CUDA.
    profile_command=(
        "$runtime" exec --nv --pwd /workspace
        "${container_binds[@]}"
        "$image_path"
        "$nsys_executable" profile --force-overwrite=true --sample=none --cpuctxsw=none
        --trace=cuda,nvtx,osrt,cublas,cudnn
        --delay "$nsys_delay_seconds" --duration "$nsys_duration_seconds"
        --output "$nsys_output_container"
        "$@"
    )
    srun /usr/bin/env \
        "${container_environment[@]}" \
        "${profile_command[@]}"
elif [[ $ncu_profile == 1 ]]; then
    ncu_directory="$output_path/ncu"
    ncu_output_host="$ncu_directory/seg-opt-${SLURM_JOB_ID}"
    ncu_output_container="/output/ncu/seg-opt-${SLURM_JOB_ID}"
    mkdir -p "$ncu_directory"
    echo "NCU:       ${ncu_output_host}.ncu-rep (roofline, skip ${ncu_launch_skip_before_match}, collect ${ncu_launch_count})"
    profile_command=(
        "$runtime" exec --nv --pwd /workspace
        "${container_binds[@]}"
        "$image_path"
        "$ncu_executable" --set roofline --target-processes application-only
        --launch-skip-before-match "$ncu_launch_skip_before_match"
        --launch-count "$ncu_launch_count"
        --export "$ncu_output_container" --force-overwrite
        "$@"
    )
    srun /usr/bin/env \
        "${container_environment[@]}" \
        "${profile_command[@]}"
else
    srun /usr/bin/env \
        "${container_environment[@]}" \
        "${container_command[@]}"
fi

echo "Finished: $(date --iso-8601=seconds)"
