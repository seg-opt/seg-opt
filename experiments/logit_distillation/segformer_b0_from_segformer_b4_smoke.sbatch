#!/usr/bin/env bash
# One-batch, H100 dependency and pipeline check for SegFormer-B4 to B0 logit KD.
# The full training submission is gated on this job with afterok.
#SBATCH --job-name=segformer-b4-to-b0-kd-smoke
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

export REPO_PATH="${REPO_PATH:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd -- "$REPO_PATH"

if (( $# == 0 )); then
    set -- experiments/logit_distillation/segformer_b0_from_segformer_b4.yaml
fi

exec python3 -S scripts/experiment.py launch --smoke "$@"
