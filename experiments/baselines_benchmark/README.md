# Baselines benchmark

This is the single active experiment stage before any distillation work. It
trains one Fast-SCNN student and four independent teacher candidates on exactly
the same clean masks, aligned seed-42 split, 512x768 frames, augmentations,
loss, and 20-epoch budget. The only model-specific settings are initialization
and the learning-rate/batch configuration already used by the canonical
baselines.

| Config | Role |
|---|---|
| `fast_scnn.yaml` | Student baseline |
| `resnet34_unet.yaml` | Teacher candidate |
| `segformer_b0.yaml` | Teacher candidate |
| `segformer_b2.yaml` | Teacher candidate |
| `segformer_b4.yaml` | Teacher candidate |
| `dinov3_vitl16.yaml` | Teacher candidate |
| `mask2former_swinl.yaml` | Teacher candidate |

Run the Fast-SCNN smoke check first:

```bash
sbatch --account=<SERVICE_ID> --export=ALL,IMAGE_PATH,DATA_PATH \
  experiments/baselines_benchmark/smoke.sbatch
```

Submit the seven-run serial array:

```bash
sbatch --account=<SERVICE_ID> --export=ALL,IMAGE_PATH,DATA_PATH \
  experiments/baselines_benchmark/train.sbatch
```

After all runs succeed, summarize them with:

```bash
uv run python -m scripts.summarize_baselines_benchmark
```

Select the teacher solely by validation mIoU; use small-rock and kernel-7
Trimap small-rock IoU as boundary diagnostics. The summary then reports the
selected teacher's held-out test metrics and its margins over Fast-SCNN. Do
not start a distillation run unless the selected teacher improves both
validation mIoU and validation small-rock IoU over Fast-SCNN. The BPKD suite
in `experiments/bpkd/` is retained but deferred until this benchmark provides
that margin.
