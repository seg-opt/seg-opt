#!/usr/bin/env bash
# Submit with: sbatch --account=<grant-id> --export=ALL,IMAGE_PATH=<path> jobs/container_cpu.sbatch
#SBATCH --partition=proxima-cpu
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --job-name=seg-opt-container-cpu
#SBATCH --output=seg-opt-container-cpu-%j.out
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
  set -- python3 /opt/seg-opt/smoke_test.py
fi

"$runtime" exec "${binds[@]}" "$IMAGE_PATH" "$@"