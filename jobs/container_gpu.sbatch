#!/usr/bin/env bash
# Submit with: sbatch --account=<grant-id> --export=ALL,IMAGE_PATH=<path> jobs/container_gpu.sbatch
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --job-name=seg-opt-container-gpu
#SBATCH --output=seg-opt-container-gpu-%j.out
set -euo pipefail

: "${IMAGE_PATH:?Set IMAGE_PATH to an immutable shared SIF path}"
runtime=$(command -v singularity || command -v apptainer || true)
[[ -n $runtime ]] || { echo "error: Singularity/Apptainer is unavailable" >&2; exit 1; }

repo_path=${REPO_PATH:-$PWD}
binds=(--bind "$repo_path:/workspace")
if [[ -n ${DATA_PATH:-} ]]; then
  binds+=(--bind "$DATA_PATH:/data")
fi
if (($# == 0)); then
  set -- python3 /opt/seg-opt/smoke_test.py --require-gpu
fi

"$runtime" exec --nv "${binds[@]}" "$IMAGE_PATH" "$@"