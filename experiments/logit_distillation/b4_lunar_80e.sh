#!/usr/bin/env bash
set -euo pipefail
mode=${1:?}
campaign=${2:?}
label=${3:?}
case "$mode" in
submit)
# Authenticate first, then smoke-check online logging before the three seeds.

: "${IMAGE_PATH:?Set the development SIF path}"
: "${DATA_PATH:?Set the datasets directory}"
export WANDB_ENTITY="${WANDB_ENTITY:-ogrodowskijedrzej-put}"
export WANDB_PROJECT="${WANDB_PROJECT:-seg-opt}"
export REPO_PATH="${REPO_PATH:-$PWD}"
export OUTPUT_PATH="${OUTPUT_PATH:-$HOME/pl1200-01/project_data/results}"
export CAMPAIGN_ID="${CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! $CAMPAIGN_ID =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "CAMPAIGN_ID must contain only letters, digits, underscores, or hyphens" >&2
    exit 2
fi
if [[ -z ${WANDB_API_KEY:-} && -z ${WANDB_API_KEY_FILE:-} && ! -f $HOME/.netrc ]]; then
    echo "Configure wandb login, WANDB_API_KEY, or WANDB_API_KEY_FILE before submission" >&2
    exit 2
fi
if [[ -n ${WANDB_API_KEY_FILE:-} && ! -r $WANDB_API_KEY_FILE ]]; then
    echo "WANDB_API_KEY_FILE is not readable" >&2
    exit 2
fi
[[ -f $IMAGE_PATH && -d $DATA_PATH ]] || { echo "Missing container or dataset directory" >&2; exit 2; }
[[ -f $OUTPUT_PATH/sard/teacher_segformer_b4/checkpoints/best.ckpt ]] || {
    echo "Missing trained SegFormer-B4 teacher under OUTPUT_PATH/sard/teacher_segformer_b4" >&2
    exit 2
}

cd "$REPO_PATH"
experiment_dir=experiments/logit_distillation/$campaign
campaign_dir="$OUTPUT_PATH/logit_distillation/$campaign/$CAMPAIGN_ID"
mkdir -p "$(dirname "$campaign_dir")"
# Refuse to reuse an existing campaign directory.
mkdir "$campaign_dir"
smoke=$(sbatch --parsable --export=ALL,CAMPAIGN_SMOKE=1 \
    --time=00:30:00 --job-name="$label" \
    --output="$campaign_dir/smoke-%j.out" --error="$campaign_dir/smoke-%j.err" \
    "$experiment_dir/train.sbatch")
full=$(sbatch --parsable --export=ALL,CAMPAIGN_SMOKE=0 \
    --array=0-2%3 --dependency="afterok:$smoke" \
    --output="$campaign_dir/train-%A_%a.out" --error="$campaign_dir/train-%A_%a.err" \
    "$experiment_dir/train.sbatch")
printf 'Campaign: %s\nSmoke: %s\nFull array: %s\nResults: %s\n' "$CAMPAIGN_ID" "$smoke" "$full" "$campaign_dir"
printf 'campaign_id=%s\nsmoke_job=%s\narray_job=%s\n' "$CAMPAIGN_ID" "$smoke" "$full" > "$campaign_dir/submission.txt"
  ;;
train)

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
export WANDB_RUN_GROUP="${label}-${CAMPAIGN_ID}"
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

experiment_dir=experiments/logit_distillation/$campaign
output_dir="results/logit_distillation/$campaign/${CAMPAIGN_ID}/seed${seed}"
extra_args=()
if [[ ${CAMPAIGN_SMOKE:-0} == 1 ]]; then
    output_dir="results/logit_distillation/$campaign/${CAMPAIGN_ID}/smoke"
    export WANDB_RUN_GROUP="${WANDB_RUN_GROUP}-smoke"
    export SINGULARITYENV_WANDB_RUN_GROUP="$WANDB_RUN_GROUP"
    export APPTAINERENV_WANDB_RUN_GROUP="$WANDB_RUN_GROUP"
    extra_args+=(--fast-dev-run)
fi

exec bash scripts/train.sbatch \
    bash -c "python experiments/logit_distillation/check_wandb.py && exec python -m scripts.train \"\$@\"" _ \
    --config "$experiment_dir/seed${seed}.yaml" \
    --output-dir "$output_dir" --devices 1 "${extra_args[@]}"
  ;;
*)
  echo "unknown mode ${mode}" >&2
  exit 2
  ;;
esac
