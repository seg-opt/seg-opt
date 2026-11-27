# DINOv3+Mask2Former strong-teacher distillation

This experiment tests whether a DINOv3+Mask2Former teacher improves two students
compared with SegFormer-B4 teacher runs. The historical 80-epoch teacher reached
test mIoU 0.9052 and small-rock IoU 0.7821; those measurements belong to that
historical run, not verified evidence for the current teacher.

The two runs isolate logit distillation with the same strong teacher. Both
students and the teacher now use the standard 80-epoch full-run budget:

| Task | Student | Objective | Epochs | Matched comparison |
|---:|---|---|---:|---|
| 0 | SegFormer-B0 | logit KD | 80 | B4 -> B0 logit KD, matched 80-epoch control |
| 1 | Fast-SCNN | logit KD | 80 | B4 -> Fast-SCNN logit KD, matched 80-epoch control |

Mask2Former dense evidence is normalized into a categorical distribution and
converted to log-potentials before KD. No feature distillation is enabled. The
task-only controls can be reused only when their verified artifacts match the
80-epoch budget and remaining protocol.

Train a teacher with:

```bash
scripts/submit dataset=lunar method=supervised teacher=none student=dinov3_vitl16_mask2former \
  training.batch_size=1 training.accumulate_grad_batches=16 training.learning_rate=0.0001 \
  training.backbone_learning_rate=null
```

Before submitting students, train with the current 80-epoch protocol or
adopt a teacher after verifying its protocol and artifacts.
Its standalone checkpoint reference is
`results/strong_teacher_distillation/teacher_dinov3_mask2former/seed42/checkpoints/best.ckpt`,
not the historical `lunar_teacher_80e` artifact. Its full protocol
and artifacts are currently not readable by our HPC user; do not adopt, copy,
or relabel it as a verified current teacher until that verification is possible.

Prefer [`pipeline`](../README.md) with each student config and matched completed
baseline and teacher artifacts. The runner rewrites teacher references to those
verified artifacts and applies the validation gate. Do not submit a duplicate
teacher if a fully verified 80-epoch run already exists.

Only after the standalone teacher reference is prepared, the legacy student
smoke-gated sequence is available with:

```bash
scripts/submit dataset=lunar method=vanilla teacher=strong_dinov3_mask2former student=segformer_b0
scripts/submit --after-job "$FIRST_TRAIN_JOB_ID" dataset=lunar method=vanilla \
  teacher=strong_dinov3_mask2former student=fast_scnn training.learning_rate=0.0005 \
  training.backbone_learning_rate=null
```

Submit the second command with the first run's printed training job ID to
run the students serially.
