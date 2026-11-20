# DINOv3+Mask2Former strong-teacher distillation

This experiment tests whether a DINOv3+Mask2Former teacher improves two students
compared with SegFormer-B4 teacher runs. The historical 80-epoch teacher reached
test mIoU 0.9052 and small-rock IoU 0.7821; those measurements belong to that
historical run, not verified evidence for the current recipe.

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

The legacy `submit.sh` schedules student smoke/full arrays only; it does not
train a teacher. Before submitting students, train from the current 80-epoch
teacher config or adopt a teacher only after full protocol/artifact verification.
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
bash experiments/strong_teacher_distillation/submit.sh
```

Only one strong-teacher job runs at a time to expose memory issues clearly
during the smoke array.
