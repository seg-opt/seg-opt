#!/usr/bin/env bash
# Authenticate first, then smoke-check online logging before the three seeds.
set -euo pipefail

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
experiment_dir=experiments/logit_distillation/b4_to_b1_lunar_80e
campaign_dir="$OUTPUT_PATH/logit_distillation/b4_to_b1_lunar_80e/$CAMPAIGN_ID"
mkdir -p "$(dirname "$campaign_dir")"
# Refuse to reuse an existing campaign directory.
mkdir "$campaign_dir"
smoke=$(sbatch --parsable --export=ALL,CAMPAIGN_SMOKE=1 \
    --time=00:30:00 --job-name=b4-b1-online-smoke \
    --output="$campaign_dir/smoke-%j.out" --error="$campaign_dir/smoke-%j.err" \
    "$experiment_dir/train.sbatch")
full=$(sbatch --parsable --export=ALL,CAMPAIGN_SMOKE=0 \
    --array=0-2%3 --dependency="afterok:$smoke" \
    --output="$campaign_dir/train-%A_%a.out" --error="$campaign_dir/train-%A_%a.err" \
    "$experiment_dir/train.sbatch")
printf 'Campaign: %s\nSmoke: %s\nFull array: %s\nResults: %s\n' "$CAMPAIGN_ID" "$smoke" "$full" "$campaign_dir"
printf 'campaign_id=%s\nsmoke_job=%s\narray_job=%s\n' "$CAMPAIGN_ID" "$smoke" "$full" > "$campaign_dir/submission.txt"
