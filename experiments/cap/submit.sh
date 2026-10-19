#!/usr/bin/env bash
# Submit one smoke check followed by the no-CAP control and CAP treatment.
set -euo pipefail

: "${IMAGE_PATH:?Set IMAGE_PATH to the development SIF before submitting.}"
: "${DATA_PATH:?Set DATA_PATH to the datasets directory before submitting.}"

experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
smoke_job=$(sbatch --parsable --export=ALL,IMAGE_PATH,DATA_PATH \
    "$experiment_dir/smoke.sbatch")
no_cap_job=$(sbatch --parsable --dependency="afterok:${smoke_job}" \
    --export=ALL,IMAGE_PATH,DATA_PATH "$experiment_dir/no_cap.sbatch")
cap_job=$(sbatch --parsable --dependency="afterok:${smoke_job}" \
    --export=ALL,IMAGE_PATH,DATA_PATH "$experiment_dir/cap.sbatch")

printf 'Smoke job: %s\n' "$smoke_job"
printf 'No-CAP job: %s (afterok:%s)\n' "$no_cap_job" "$smoke_job"
printf 'CAP job:    %s (afterok:%s)\n' "$cap_job" "$smoke_job"
