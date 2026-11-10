# Controlled CNN distillation: 100 epochs

This series is separate from the historical 20-epoch experiments. All full
runs use 100 epochs, the aligned seed-42 split, clean 512x768 masks, and the
same augmentation and losses as the historical CNN configurations.

| Run | Training seeds | Effective batch | Learning rate |
| --- | --- | --- | --- |
| Supervised Fast-SCNN | 42, 43, 44 | 16 | 5e-4 |
| Supervised ResNet101 U-Net teacher | 42 | 16 (8 x 2) | 3e-4; backbone 3e-5 |
| Fast-SCNN vanilla KD from ResNet101 | 42, 43, 44 | 16 | 5e-4 |

Use `scripts/experiment.py --seed` to override the training seed; the data
split remains fixed. Keep the teacher seed fixed at 42 for all paired student
comparisons. Vanilla KD retains weight 1, temperature 1, and spatial stride 8.
The warmup fraction remains 0.05, now measured over the 100-epoch schedule.
Select and restore the checkpoint with the best validation mIoU, not test mIoU.

## HPC execution

Use the existing snapshot-owned runner and development SIF on Eagle. Set
`IMAGE_PATH`, `DATA_PATH` (the parent containing `artificial_lunar_landscape`),
`OUTPUT_PATH`, and `CACHE_PATH` to cluster paths. W&B uses
`WANDB_ENTITY=ogrodowskijedrzej-put`, `WANDB_PROJECT=seg-opt`, and online mode.
Credentials stay in the HPC home directory, never in source snapshots.

Submit the seed-42 baseline and teacher together with `submit`. Submit the
seed-43 and seed-44 baselines separately with `--seed`. For each student,
use `pipeline --teacher-artifacts` with the new teacher artifact directory,
the matching `--baseline-artifacts`, and the matching `--seed`.

Connect each pipeline's `--after-job` to a successful completion barrier for
both its baseline workflow and the teacher workflow. Never depend only on
the teacher when its paired baseline may still be running. The runner then
performs the verified validation gate, KD smoke, full KD, and summary.
The teacher must strictly outperform the paired baseline in validation mIoU
and small-rock IoU. A failed gate blocks KD; do not substitute an older teacher.

Output directories in the YAMLs are standalone defaults. The shared runner
assigns immutable workflow/stage/attempt directories and rewrites teacher
references after validating artifact hashes. Preserve these manifests,
checkpoints, configs, split files, and offline W&B data on HPC.

BPKD/SARD calibration is a later stage after the new teacher and paired
baselines are verified. Their historical untuned weights are not part of this
full-run submission. Do not pool 20- and 100-epoch results as matched controls.