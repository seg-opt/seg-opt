#!/usr/bin/env bash
# Full SARD with seeds 42/43/44 on every dataset. Lunar reuses the existing
# SegFormer-B4 teacher (seed 42 already done); Cityscapes and ADE20K first
# train their own teacher and gate the three students on it with afterok.
set -euo pipefail

: "${IMAGE_PATH:?Set IMAGE_PATH to the development SIF before submitting.}"
: "${DATA_PATH:?Set DATA_PATH to the datasets directory before submitting.}"

experiment_dir=experiments/sard
export_vars=ALL,IMAGE_PATH,DATA_PATH

submit() {
    local name=$1
    shift
    sbatch --parsable --export="$export_vars" --job-name="$name" "$@"
}

for seed in 43 44; do
    job=$(submit "sard-lunar-s$seed" "$experiment_dir/run.sbatch" \
        "$experiment_dir/fast_scnn_sard_full_from_segformer_b4_seed$seed.yaml")
    printf 'lunar seed %s:      %s\n' "$seed" "$job"
done

for dataset in cityscapes ade20k; do
    teacher=$(submit "sard-$dataset-teacher" "$experiment_dir/run.sbatch" \
        "$experiment_dir/$dataset/segformer_b4.yaml")
    printf '%s teacher:    %s\n' "$dataset" "$teacher"
    for seed in 42 43 44; do
        job=$(submit "sard-$dataset-s$seed" --dependency="afterok:$teacher" \
            "$experiment_dir/run.sbatch" \
            "$experiment_dir/$dataset/fast_scnn_sard_full_seed$seed.yaml")
        printf '%s seed %s:    %s (afterok:%s)\n' "$dataset" "$seed" "$job" "$teacher"
    done
done
