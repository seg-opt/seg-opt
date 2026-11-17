#!/usr/bin/env bash
# One-batch teacher and SARD student on Cityscapes and ADE20K before the full runs.
#SBATCH --job-name=sard-smoke-benchmarks
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

export WANDB_MODE="offline"

for dataset in cityscapes ade20k; do
    smoke_dir="results/sard/smoke/$SLURM_JOB_ID/$dataset"
    mkdir -p "$smoke_dir"

    bash scripts/train.sbatch \
        python -m scripts.train \
        --config "experiments/sard/$dataset/segformer_b4.yaml" \
        --output-dir "$smoke_dir/teacher" \
        --devices 1 \
        --fast-dev-run \
        --offline

    # The repository is mounted read-only, so the student config pointing at
    # the smoke teacher is written under results/.
    sed "s#results/sard/$dataset/segformer_b4/seed42/checkpoints/best.ckpt#$smoke_dir/teacher/checkpoints/best.ckpt#" \
        "experiments/sard/$dataset/fast_scnn_sard_full_seed42.yaml" \
        > "$smoke_dir/sard_full.yaml"

    bash scripts/train.sbatch \
        python -m scripts.train \
        --config "$smoke_dir/sard_full.yaml" \
        --output-dir "$smoke_dir/sard_full" \
        --devices 1 \
        --fast-dev-run \
        --offline
done
