#!/usr/bin/env bash
# Submit the SegFormer-B4 to SegFormer-B0 smoke test and gated full job.
set -euo pipefail

: "${IMAGE_PATH:?Set IMAGE_PATH to the development SIF before submitting.}"
: "${DATA_PATH:?Set DATA_PATH to the datasets directory before submitting.}"

experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
smoke_job=$(sbatch --parsable --export=ALL,IMAGE_PATH,DATA_PATH \
    "$experiment_dir/segformer_b0_from_segformer_b4_smoke.sbatch")
full_job=$(sbatch --parsable --dependency="afterok:${smoke_job}" \
    --export=ALL,IMAGE_PATH,DATA_PATH \
    "$experiment_dir/segformer_b0_from_segformer_b4_train.sbatch")

printf 'Smoke job: %s\n' "$smoke_job"
printf 'Full job:  %s (afterok:%s)\n' "$full_job" "$smoke_job"
