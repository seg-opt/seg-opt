#!/usr/bin/env bash
# One seed per GPU; submit.sh releases a three-job array after online smoke.
#SBATCH --job-name=b4-b2-lunar-80e
#SBATCH --account=pl1200-01
#SBATCH --partition=proxima
#SBATCH --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

: "${CAMPAIGN_ID:?Submit using submit.sh}"
: "${OUTPUT_PATH:?Set the persistent HPC results directory}"
: "${WANDB_ENTITY:?Set the W&B username or team}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) seed=42 ;;
    1) seed=43 ;;
    2) seed=44 ;;
    *) echo "Unexpected array index" >&2; exit 2 ;;
esac

export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-seg-opt}"
export WANDB_RUN_GROUP="b4-to-b2-lunar-80e-${CAMPAIGN_ID}"
if [[ -n ${WANDB_API_KEY_FILE:-} ]]; then
    IFS= read -r WANDB_API_KEY < "$WANDB_API_KEY_FILE" || [[ -n ${WANDB_API_KEY:-} ]]
    export WANDB_API_KEY
fi
# Explicit container variables also work when the image defines W&B settings.
for name in WANDB_MODE WANDB_PROJECT WANDB_ENTITY WANDB_RUN_GROUP; do
    export "SINGULARITYENV_${name}=${!name}" "APPTAINERENV_${name}=${!name}"
done
if [[ -n ${WANDB_API_KEY:-} ]]; then
    export SINGULARITYENV_WANDB_API_KEY="$WANDB_API_KEY"
    export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
fi

experiment_dir=experiments/logit_distillation/b4_to_b2_lunar_80e
output_dir="results/logit_distillation/b4_to_b2_lunar_80e/${CAMPAIGN_ID}/seed${seed}"
extra_args=()
if [[ ${CAMPAIGN_SMOKE:-0} == 1 ]]; then
    output_dir="results/logit_distillation/b4_to_b2_lunar_80e/${CAMPAIGN_ID}/smoke"
    export WANDB_RUN_GROUP="${WANDB_RUN_GROUP}-smoke"
    export SINGULARITYENV_WANDB_RUN_GROUP="$WANDB_RUN_GROUP"
    export APPTAINERENV_WANDB_RUN_GROUP="$WANDB_RUN_GROUP"
    extra_args+=(--fast-dev-run)
fi

exec bash scripts/train.sbatch \
    bash -c 'python experiments/logit_distillation/b4_to_b2_lunar_80e/check_wandb.py && exec python -m scripts.train "$@"' _ \
    --config "$experiment_dir/seed${seed}.yaml" \
    --output-dir "$output_dir" --devices 1 "${extra_args[@]}"
