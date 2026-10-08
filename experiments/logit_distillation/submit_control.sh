#!/usr/bin/env bash
# Submit the matched control only after its one-batch smoke job succeeds.
set -euo pipefail

: "${IMAGE_PATH:?Set IMAGE_PATH to the development SIF before submitting.}"
: "${DATA_PATH:?Set DATA_PATH to the datasets directory before submitting.}"

experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
smoke_job=$(sbatch --parsable --export=ALL,IMAGE_PATH,DATA_PATH "$experiment_dir/control_smoke.sbatch")
full_job=$(sbatch --parsable --dependency="afterok:${smoke_job}" \
    --export=ALL,IMAGE_PATH,DATA_PATH "$experiment_dir/control.sbatch")

printf 'Control smoke job: %s\n' "$smoke_job"
printf 'Control full job:  %s (afterok:%s)\n' "$full_job" "$smoke_job"
