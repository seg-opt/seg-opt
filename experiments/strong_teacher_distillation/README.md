# DINOv3+Mask2Former strong-teacher distillation

This experiment tests whether the stronger 80-epoch DINOv3+Mask2Former teacher
improves two students compared with the existing SegFormer-B4 teacher runs.
The teacher reaches test mIoU 0.9052 and small-rock IoU 0.7821.

The two runs isolate logit distillation with the same strong teacher. Both
students train for 80 epochs:

| Task | Student | Objective | Epochs | Matched comparison |
|---:|---|---|---:|---|
| 0 | SegFormer-B0 | logit KD | 80 | B4 -> B0 logit KD |
| 1 | Fast-SCNN | logit KD | 80 | B4 -> Fast-SCNN 80e logit KD |

Mask2Former dense evidence is normalized into a categorical distribution and
converted to log-potentials before KD. No feature distillation is enabled. The
existing task-only controls are reused, so no additional control training is
required.

Run the smoke-gated sequence with:

```bash
bash experiments/strong_teacher_distillation/submit.sh
```

Only one strong-teacher job runs at a time to expose memory issues clearly
during the smoke array.
