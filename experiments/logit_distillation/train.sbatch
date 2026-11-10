#!/usr/bin/env bash
# Train the supplied experiment configs, or the default SegFormer-B4 KD config.
# Submit through submit.sh so this job starts only after the smoke job succeeds.
#SBATCH --job-name=seg-experiment
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

export REPO_PATH="${REPO_PATH:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd -- "$REPO_PATH"

if (( $# == 0 )); then
    set -- experiments/logit_distillation/fast_scnn_from_segformer_b4.yaml
fi

export WANDB_PROJECT="${WANDB_PROJECT:-seg-opt-logit-distillation}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-segformer-b4-to-fastscnn-clean-seed42}"

exec python3 -S scripts/experiment.py launch "$@"
