#!/usr/bin/env bash
# One-batch GPU check of the full SARD pipeline: a throwaway SegFormer-B4
# teacher checkpoint, then one SARD Fast-SCNN step distilled from it.
#SBATCH --job-name=sard-smoke
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

export WANDB_MODE="offline"
smoke_dir="results/sard/smoke/$SLURM_JOB_ID"
mkdir -p "$smoke_dir"

bash scripts/train.sbatch \
    python -m scripts.train \
    --config experiments/baselines_benchmark/segformer_b4.yaml \
    --output-dir "$smoke_dir/teacher" \
    --devices 1 \
    --fast-dev-run \
    --offline

# The repository is mounted read-only in the container, so the student config
# that points at the smoke teacher is written under results/.
sed "s#results/baselines_benchmark/segformer_b4/seed42/checkpoints/best.ckpt#$smoke_dir/teacher/checkpoints/best.ckpt#" \
    experiments/sard/fast_scnn_sard_full_from_segformer_b4.yaml \
    > "$smoke_dir/sard_full.yaml"

bash scripts/train.sbatch \
    python -m scripts.train \
    --config "$smoke_dir/sard_full.yaml" \
    --output-dir "$smoke_dir/sard_full" \
    --devices 1 \
    --fast-dev-run \
    --offline
