#!/usr/bin/env bash
# Deferred: submit only after the clean benchmark selects a teacher and its
# checkpoint is propagated into the student configurations.
#SBATCH --job-name=lunar-bpkd
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --array=0-7%1
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
        experiments/bpkd/fast_scnn_task_only_clean.yaml \
        experiments/bpkd/fast_scnn_vanilla_clean.yaml \
        experiments/bpkd/fast_scnn_body_only_clean.yaml \
        experiments/bpkd/fast_scnn_edge_only_clean.yaml \
        experiments/bpkd/fast_scnn_bpkd_w3_clean.yaml \
        experiments/bpkd/fast_scnn_bpkd_w5_clean.yaml \
        experiments/bpkd/fast_scnn_bpkd_w7_clean.yaml \
        experiments/bpkd/fast_scnn_bpkd_w9_clean.yaml
fi

export WANDB_PROJECT="${WANDB_PROJECT:-seg-opt-bpkd}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-mask2former-to-fastscnn-clean-seed42}"

exec python3 -S scripts/experiment.py launch "$@"
