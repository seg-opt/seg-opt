# Baselines benchmark

This is the single active experiment stage before any distillation work. It
trains one Fast-SCNN student and six independent teacher candidates on exactly
the same clean masks, aligned seed-42 split, 512x768 frames, augmentations,
loss, and 80-epoch budget. The only model-specific settings are initialization
and the learning-rate/batch configuration already used by the canonical
baselines.

| `student` preset | Role | Batch × accumulation | LR | Backbone LR |
|---|---|---:|---:|---:|
| `fast_scnn` | Student baseline | 16 × 1 | `0.0005` | `null` |
| `resnet34_unet` | Teacher candidate | 8 × 2 | `0.0003` | `3e-05` |
| `segformer_b0` | Teacher candidate | 8 × 2 | `0.0006` | `0.00006` |
| `segformer_b2` | Teacher candidate | 4 × 4 | `0.0003` | `3e-05` |
| `segformer_b4` | Teacher candidate | 2 × 8 | `0.0003` | `3e-05` |
| `dinov3_vitl16` | Teacher candidate | 2 × 8 | `0.001` | `null` |
| `mask2former_swinl` | Teacher candidate | 1 × 16 | `0.0001` | `1e-05` |

Run the Fast-SCNN smoke check first:

```bash
scripts/submit --smoke-only dataset=lunar method=supervised teacher=none student=fast_scnn \
  training.batch_size=16 training.accumulate_grad_batches=1 training.learning_rate=0.0005 \
  training.backbone_learning_rate=null
```

Submit each model from the table separately. To run serially, pass the preceding
training job ID as `PREVIOUS_JOB_ID`. Use the smoke job ID for the first model.
Set the batch, accumulation and learning rates from the table.

```bash
scripts/submit --full-only --after-job "$PREVIOUS_JOB_ID" \
  dataset=lunar method=supervised teacher=none "student=$MODEL" \
  "training.batch_size=$BATCH_SIZE" "training.accumulate_grad_batches=$ACCUMULATION" \
  "training.learning_rate=$LR" "training.backbone_learning_rate=$BACKBONE_LR"
```

For the seed-42 summary, link each printed run's `train` directory into a
shared summary root under its model name.

```bash
mkdir -p "$SUMMARY_ROOT/$MODEL"
ln -s "$RUN_DIRECTORY/train" "$SUMMARY_ROOT/$MODEL/seed42"
# Run after linking all seven completed models.
uv run python -m scripts.summarize_baselines_benchmark --results-root "$SUMMARY_ROOT"
```

Select the teacher solely by validation mIoU; use small-rock and kernel-7
Trimap small-rock IoU as boundary diagnostics. The summary then reports the
selected teacher's held-out test metrics and its margins over Fast-SCNN. Do
not start a distillation run unless the selected teacher improves both
validation mIoU and validation small-rock IoU over Fast-SCNN. The BPKD suite
in `experiments/bpkd/` is retained but deferred until this benchmark provides
that margin.
