#!/usr/bin/env bash
# Submit smoke -> teacher -> four-run student array, each gated on afterok.
# Pass --skip-teacher when results/baselines_benchmark/segformer_b4/seed42
# already holds a trained checkpoint.
set -euo pipefail

: "${IMAGE_PATH:?Set IMAGE_PATH to the development SIF before submitting.}"
: "${DATA_PATH:?Set DATA_PATH to the datasets directory before submitting.}"

skip_teacher=0
[[ ${1:-} == --skip-teacher ]] && skip_teacher=1

experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export_vars=ALL,IMAGE_PATH,DATA_PATH

smoke_job=$(sbatch --parsable --export="$export_vars" "$experiment_dir/smoke.sbatch")
printf 'Smoke job:   %s\n' "$smoke_job"
gate=$smoke_job
if ((!skip_teacher)); then
    teacher_job=$(sbatch --parsable --dependency="afterok:$gate" \
        --export="$export_vars" "$experiment_dir/teacher.sbatch")
    printf 'Teacher job: %s (afterok:%s)\n' "$teacher_job" "$gate"
    gate=$teacher_job
fi
array_job=$(sbatch --parsable --dependency="afterok:$gate" \
    --export="$export_vars" "$experiment_dir/train.sbatch")
printf 'Array job:   %s (afterok:%s)\n' "$array_job" "$gate"
