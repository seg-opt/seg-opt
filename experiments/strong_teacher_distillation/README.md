# DINOv3+Mask2Former strong-teacher distillation

This experiment tests whether a DINOv3+Mask2Former teacher improves two students
compared with SegFormer-B4 teacher runs. The historical 80-epoch teacher reached
test mIoU 0.9052 and small-rock IoU 0.7821; those are not measurements of the
current 100-epoch configuration.

The two runs isolate logit distillation with the same strong teacher. Both
students and the teacher now use the standard 100-epoch full-run budget:

| Task | Student | Objective | Epochs | Matched comparison |
|---:|---|---|---:|---|
| 0 | SegFormer-B0 | logit KD | 100 | B4 -> B0 logit KD, matched 100-epoch control |
| 1 | Fast-SCNN | logit KD | 100 | B4 -> Fast-SCNN logit KD, matched 100-epoch control |

Mask2Former dense evidence is normalized into a categorical distribution and
converted to log-potentials before KD. No feature distillation is enabled. The
task-only controls can be reused only when their verified artifacts match the
100-epoch budget and remaining protocol. Do not compare new runs against
unmatched historical 20- or 80-epoch controls.

The legacy `submit.sh` schedules student smoke/full arrays only; it does not
train a teacher. Before submitting students, train or adopt a verified teacher
from the current 100-epoch config. Its standalone checkpoint reference is
`results/strong_teacher_distillation/teacher_dinov3_mask2former/seed42/checkpoints/best.ckpt`,
not the historical `lunar_teacher_80e` artifact. Do not copy or relabel that old
checkpoint as a 100-epoch teacher.

Prefer the shared artifact-verifying runner: submit the teacher config, then
use `pipeline` with each student config and matched completed baseline and
teacher artifacts. The runner rewrites teacher references to those verified
artifacts and applies the validation gate. Do not submit a duplicate teacher
if a suitable verified 100-epoch run already exists.

Only after the standalone teacher reference is prepared, the legacy student
smoke-gated sequence is available with:

```bash
bash experiments/strong_teacher_distillation/submit.sh
```

Only one strong-teacher job runs at a time to expose memory issues clearly
during the smoke array.
