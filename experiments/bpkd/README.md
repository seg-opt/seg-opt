# Clean-mask BPKD ablation

> **Deferred:** do not submit this ablation suite until the clean baseline
> benchmark in `experiments/baselines_benchmark/` has selected a teacher with a
> positive validation margin over Fast-SCNN.

This experiment distils a clean/aligned Mask2Former-SwinL teacher into
Fast-SCNN. All student variants use physical batch 2 and gradient accumulation
8, preserving an effective batch size of 16 while giving Fast-SCNN's deepest
BatchNorm layer more than one value per channel.

Train the teacher first:

```bash
scripts/submit dataset=lunar method=supervised teacher=none student=mask2former_swinl \
  training.batch_size=1 training.accumulate_grad_batches=16 training.learning_rate=0.0001 \
  training.backbone_learning_rate=1e-05
```

## First: Mask2Former small-rock screen

Set `TEACHER_CHECKPOINT` to the container-visible checkpoint path before
running either student campaign.

Before the Fast-SCNN suite, use the trained teacher itself as the matched
task-only Mask2Former-SwinL control. Then train two fresh Mask2Former-SwinL
students: vanilla KD and full BPKD at the predeclared primary width, 7.
This is a same-architecture (born-again) distillation test: it isolates whether
the boundary objective helps the Mask2Former target model, rather than testing
only transfer to Fast-SCNN.

```bash
for method in vanilla bpkd; do
  scripts/submit dataset=lunar "method=$method" \
    teacher=bpkd_mask2former_swinl student=mask2former_swinl \
    training.batch_size=1 training.accumulate_grad_batches=16 \
    training.learning_rate=0.0001 training.backbone_learning_rate=0.00001 \
    "distillation.teacher.checkpoint_path=$TEACHER_CHECKPOINT"
done
```

Compare the teacher/control, vanilla KD, and BPKD on validation small-rock IoU
and fixed-kernel-7 Trimap small-rock IoU. A positive BPKD result over both
controls justifies the larger Fast-SCNN and kernel-width suite. Because teacher
and student have the same capacity, expect any gain to be modest; a larger
teacher would be a stronger subsequent test of distillation capacity.

## Fast-SCNN suite

The default teacher is
`results/bpkd/teachers/mask2former_swinl_clean/seed42/checkpoints/best.ckpt`.
To use a new teacher run, pass `distillation.teacher.checkpoint_path` with the
container-visible path to its checkpoint. Submit the supervised control without
a teacher override, then submit each KD variant separately.

```bash
scripts/submit dataset=lunar method=supervised teacher=none student=fast_scnn training.batch_size=2 \
  training.accumulate_grad_batches=8 training.learning_rate=0.0005 \
  training.backbone_learning_rate=null
student_args=(dataset=lunar teacher=bpkd_mask2former_swinl student=fast_scnn
  training.batch_size=2 training.accumulate_grad_batches=8
  training.learning_rate=0.0005 training.backbone_learning_rate=null
  "distillation.teacher.checkpoint_path=$TEACHER_CHECKPOINT")
scripts/submit method=vanilla "${student_args[@]}"
scripts/submit method=bpkd "${student_args[@]}" distillation.edge_weight=0
scripts/submit method=bpkd "${student_args[@]}" distillation.body_weight=0
for width in 3 5 7 9; do
  scripts/submit method=bpkd "${student_args[@]}" "distillation.edge_kernel_size=$width"
done
```

The primary comparison is `bpkd_w7`; widths 3, 5, 7, and 9 are exploratory.
Select a width by validation Trimap small-rock IoU only, then report its held-out
test metrics. Checkpoints are still selected by validation mIoU.

| Variant index | Variant |
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
