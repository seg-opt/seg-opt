#!/usr/bin/env bash
# Submit only after mask2former_teacher_clean.yaml has produced best.ckpt.
# The teacher run itself is the matched task-only Mask2Former control.
#SBATCH --job-name=lunar-mask2former-bpkd
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --array=0-1%1
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
        experiments/bpkd/mask2former_vanilla_clean.yaml \
        experiments/bpkd/mask2former_bpkd_w7_clean.yaml
fi

export WANDB_PROJECT="${WANDB_PROJECT:-seg-opt-bpkd}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-mask2former-born-again-clean-seed42}"

exec python3 -S scripts/experiment.py launch "$@"
