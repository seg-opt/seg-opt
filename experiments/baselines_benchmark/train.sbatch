#!/usr/bin/env bash
# Seven clean/aligned supervised baselines: Fast-SCNN plus six teacher candidates.
#SBATCH --job-name=seg-clean-bench
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --array=0-6%1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=results/slurm-%x-%A_%a.out
#SBATCH --error=results/slurm-%x-%A_%a.err

set -euo pipefail

export REPO_PATH="${REPO_PATH:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd -- "$REPO_PATH"

if (( $# == 0 )); then
    set -- \
        experiments/baselines_benchmark/fast_scnn.yaml \
        experiments/baselines_benchmark/resnet34_unet.yaml \
        experiments/baselines_benchmark/segformer_b0.yaml \
        experiments/baselines_benchmark/segformer_b2.yaml \
        experiments/baselines_benchmark/segformer_b4.yaml \
        experiments/baselines_benchmark/dinov3_vitl16.yaml \
        experiments/baselines_benchmark/mask2former_swinl.yaml
fi

export WANDB_PROJECT="${WANDB_PROJECT:-seg-opt-clean-benchmarks}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-clean-aligned-seed42}"

exec python3 -S scripts/experiment.py launch "$@"
