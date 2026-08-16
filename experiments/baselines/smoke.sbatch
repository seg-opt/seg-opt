#!/usr/bin/env bash
#SBATCH --job-name=seg-baselines-smoke
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

CONFIGS=(
    experiments/baselines/fast_scnn.yaml
    experiments/baselines/resnet34_unet.yaml
    experiments/baselines/segformer_b0.yaml
    experiments/baselines/dinov3_vitl16.yaml
    experiments/baselines/mask2former_swinl.yaml
)

for config in "${CONFIGS[@]}"; do
    echo "Smoke testing $config"
    bash scripts/train.sbatch \
        python -m scripts.train \
        --config "$config" \
        --devices 1 \
        --fast-dev-run \
        --offline
done
