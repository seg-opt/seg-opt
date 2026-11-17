#!/usr/bin/env bash
# Submit the smoke test and a full job that begins only after smoke success.
set -euo pipefail

export REPO_PATH="${REPO_PATH:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd -- "$REPO_PATH"

if (( $# == 0 )); then
    set -- experiments/logit_distillation/fast_scnn_from_segformer_b4.yaml
fi

exec python3 -S scripts/experiment.py submit "$@"
