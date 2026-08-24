# Clean-mask BPKD ablation

This experiment distils a clean/aligned Mask2Former-SwinL teacher into
Fast-SCNN. All student variants use physical batch 1 and gradient accumulation
16 so BatchNorm and effective batch size are matched.

Train the teacher first:

```bash
sbatch --account=<SERVICE_ID> scripts/train.sbatch \
  --image-path /path/to/development.sif \
  --data-path /mnt/storage_6/project_data/pl1200-01/datasets \
  --python-path /cache/runtime-deps/kornia-0.8.2-py313 -- \
  python -m scripts.train \
  --config experiments/bpkd/mask2former_teacher_clean.yaml \
  --devices 1
```

## First: Mask2Former small-rock screen

Before the Fast-SCNN suite, use the trained teacher itself as the matched
task-only Mask2Former-SwinL control. Then train two fresh Mask2Former-SwinL
students: vanilla KD and full BPKD at the predeclared primary width, 7.
This is a same-architecture (born-again) distillation test: it isolates whether
the boundary objective helps the Mask2Former target model, rather than testing
only transfer to Fast-SCNN.

```bash
sbatch --account=<SERVICE_ID> --export=ALL,IMAGE_PATH,DATA_PATH \
  experiments/bpkd/train_mask2former_screen.sbatch
```

Compare the teacher/control, vanilla KD, and BPKD on validation small-rock IoU
and fixed-kernel-7 Trimap small-rock IoU. A positive BPKD result over both
controls justifies the larger Fast-SCNN and kernel-width suite. Because teacher
and student have the same capacity, expect any gain to be modest; a larger
teacher would be a stronger subsequent test of distillation capacity.

## Fast-SCNN suite

After `results/bpkd/teachers/mask2former_swinl_clean/seed42/checkpoints/best.ckpt`
exists, submit the eight-run array:

```bash
sbatch --account=<SERVICE_ID> --export=ALL,IMAGE_PATH,DATA_PATH \
  experiments/bpkd/train.sbatch
```

The primary comparison is `bpkd_w7`; widths 3, 5, 7, and 9 are exploratory.
Select a width by validation Trimap small-rock IoU only, then report its held-out
test metrics. Checkpoints are still selected by validation mIoU.

| Array task | Variant |
|---:|---|
| 0 | Task-only Fast-SCNN control |
| 1 | Target-aware vanilla pixel KD |
| 2 | BPKD body only |
| 3 | BPKD edge only |
| 4–7 | Full BPKD, kernels 3/5/7/9 |

Artifacts record the mask/split setup, Trimap metrics, complete BPKD settings,
and teacher checkpoint provenance. Distilled checkpoints contain only Fast-SCNN
weights and can be evaluated with `python -m scripts.evaluate`.

## Interpretation protocol

Keep width 7 as the predeclared primary comparison. For the exploratory width
sweep, choose a single width using validation `trimap_iou_small_rock` only,
then inspect that selected run's held-out test metrics once. Report every
validation delta against task-only, and report boundary deltas against vanilla
KD, including negative results. Treat seed 42 as directional: BPKD is useful
only when it improves both validation small-rock IoU and Trimap small-rock IoU
over task-only, improves the boundary diagnostic over vanilla KD, and reduces
validation mIoU by no more than 0.5 percentage points.

The paper leaves implementation details and authoritative training code open.
This implementation fixes the comparison at output stride 8, uses replicate-
padded square morphology with adaptive average pooled soft masks, excludes
ignored label cells, and uses the paper-default body/edge weights 20/50,
alpha 2, temperatures 1, kernel 7, PRM, and POM.
