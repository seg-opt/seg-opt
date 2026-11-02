#!/usr/bin/env bash
set -euo pipefail

: "${IMAGE_PATH:?Set IMAGE_PATH to the development SIF before submitting.}"
: "${DATA_PATH:?Set DATA_PATH to the datasets directory before submitting.}"

experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
smoke_job=$(sbatch --parsable --export=ALL,IMAGE_PATH,DATA_PATH \
    "$experiment_dir/smoke.sbatch")
full_job=$(sbatch --parsable --dependency="afterok:${smoke_job}" \
    --export=ALL,IMAGE_PATH,DATA_PATH \
    "$experiment_dir/train.sbatch")

printf 'Strong-teacher smoke array: %s\n' "$smoke_job"
printf 'Strong-teacher full array:  %s (afterok:%s)\n' "$full_job" "$smoke_job"
