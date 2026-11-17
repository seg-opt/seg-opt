#!/usr/bin/env bash
# Submit the SegFormer-B4 to SegFormer-B0 smoke test and gated full job.
set -euo pipefail

export REPO_PATH="${REPO_PATH:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd -- "$REPO_PATH"

if (( $# == 0 )); then
    set -- experiments/logit_distillation/segformer_b0_from_segformer_b4.yaml
fi

export WANDB_PROJECT="${WANDB_PROJECT:-seg-opt-logit-distillation}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-segformer-b4-to-segformer-b0-clean-seed42}"

exec python3 -S scripts/experiment.py submit "$@"
