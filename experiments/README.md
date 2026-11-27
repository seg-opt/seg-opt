# Experiment map

## Training budget

Active full runs use 80 epochs in `baselines_benchmark/`,
`logit_distillation/`, `sard/`, and `strong_teacher_distillation/`.
Smoke limits are unchanged. `dinov3_mask2former/lunar_teacher_80e`
identifies a historical teacher run. Matching epochs alone
does not authorize checkpoint reuse: verify the full protocol and artifacts (see
[strong-teacher guidance](strong_teacher_distillation/README.md)).

Historical runs retain their budgets in immutable source snapshots and resolved
artifact configs; changing shared presets does not rewrite those records.
Do not mix budgets or reuse unmatched teacher/control artifacts. New runs use
unique runner outputs, not historical standalone directories.

## Clean baseline benchmark

[`baselines_benchmark/`](baselines_benchmark/) compares Fast-SCNN and six teachers
on clean masks with the aligned seed-42 split. ResNet101-U-Net is an optional
extra outside the seven-run comparison. Require a positive validation teacher margin.

## SARD

[`sard/`](sard/) tests structure-aware feature distillation from a SegFormer-B4
teacher into Fast-SCNN against task-only, vanilla logit KD and uniform-weight
controls.

## Exploratory work

[`dinov3_demo/`](dinov3_demo/) and [`dinov3_probe_v2/`](dinov3_probe_v2/) are
standalone DINOv3 exploration, not part of the benchmark decision path.

## Hydra submit on Eagle

Use `scripts/submit` for all experiments. Run it from the Eagle login host. To save an
experiment as a Bash wrapper, put these lines in a `.sh` file and run it from
the repository root:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Set this to a writable directory containing the selected teacher checkpoint.
export OUTPUT_PATH="/path/to/writable/results"

exec scripts/submit \
  dataset=lunar method=vanilla \
  teacher=lunar_segformer_b4 student=segformer_b2 \
  training.seed=43 training.learning_rate=0.001
```

| Argument | Available presets |
| --- | --- |
| `dataset` | `lunar`, `ade20k`, `cityscapes`, `lusnar` |
| `method` | `supervised`, `vanilla`, `sard` |
| `teacher` | `none` for supervised learning, or a preset in [`hydra/teacher/`](hydra/teacher/) |
| `student` | A SegFormer, Fast-SCNN, ResNet U-Net, Mask2Former, or DINOv3 preset in [`hydra/student/`](hydra/student/) |

Specify all four presets. Override config values with `section.key=value`,
for example `training.seed=43`, `training.precision=fp32` or
`distillation.vanilla_temperature=2.0`. The common defaults are in
[`hydra/config.yaml`](hydra/config.yaml). If you override batch size or
gradient accumulation, their product must equal 16. Quote an entire override
argument when its value contains spaces or shell metacharacters.

| Environment variable | What it sets |
| --- | --- |
| `IMAGE_PATH` | Development SIF path |
| `DATA_PATH` | Dataset parent directory, mounted at `/datasets` |
| `OUTPUT_PATH` | Writable results root (default `results/` in the checkout), mounted at `/output` and `/workspace/results` |
| `CACHE_PATH` | Writable container cache directory |
| `WANDB_PROJECT` | W&B project, default `seg-opt` |

For `teacher=lunar_segformer_b4`, the checkpoint is read from
`$OUTPUT_PATH/sard/teacher_segformer_b4/checkpoints/best.ckpt`. The launcher
uses Eagle's `proxima` partition with one H100 per job. It submits a smoke
job with W&B offline and a full job with W&B online that starts after the
smoke succeeds. It prints both job IDs and the results directory. Each stage
saves its resolved config with its results. Keep the same checkout in place
until both jobs finish.

### Submit options

Each invocation submits one experiment. Use `--after-job ID` to wait for an
existing successful job, `--smoke-only` or `--full-only` to select one stage,
and `--disable-wandb` to disable the logger. Pass the preceding training job
ID with `--after-job` for serial runs.

Use `--evaluate CHECKPOINT` to evaluate or `--init-checkpoint PATH` to initialize
training for transfer. For KD with a newly trained teacher, override
`distillation.teacher.checkpoint_path`. All checkpoint paths must be visible
inside the container.

For example, use the same LUSNAR checkpoint for evaluation at 512×512 and
transfer into Lunar:

```bash
scripts/submit --evaluate "$LUSNAR_CHECKPOINT" \
  dataset=lusnar method=supervised teacher=none student=dinov3_vitl16_mask2former \
    training.batch_size=2 training.accumulate_grad_batches=8 training.learning_rate=0.0001 \
    training.backbone_learning_rate=null
scripts/submit --init-checkpoint "$LUSNAR_CHECKPOINT" \
  dataset=lunar method=supervised teacher=none student=dinov3_vitl16_mask2former \
    training.batch_size=1 training.accumulate_grad_batches=16 training.learning_rate=0.0001 \
    training.backbone_learning_rate=null
```

### Managed artifact validation and historical runs

The existing snapshot-oriented runner also accepts a Hydra config exported
for the scientific teacher-quality gate. Create the export inside the checkout,
then run the managed pipeline from the repository root:

```bash
config_dir=$(mktemp -d experiments/hydra-export.XXXXXXXX)
KD_CONFIG="$config_dir/student.yaml"
uv run --only-group submit python -m scripts.compose_experiment \
  dataset=lunar method=vanilla teacher=benchmark_resnet34_unet student=fast_scnn \
    training.batch_size=16 training.accumulate_grad_batches=1 training.learning_rate=0.0005 \
    training.backbone_learning_rate=null --write-config "$KD_CONFIG"
```

Each config gets a smoke check before full training. Plain `submit` does not
establish a scientific teacher-quality gate. `pipeline` submits teacher smoke,
teacher training, a CPU validation gate, KD smoke, KD training, and a terminal
summary. Supply an existing KD config with `--config`, completed supervised
student artifacts with `--baseline-artifacts`, and either `--teacher-config`
or `--teacher-artifacts`. Only adopted teacher artifacts accept `--after-job JOB_ID`,
one job ID for a successful barrier completing both teacher and paired supervised
baseline, not the teacher alone. The gate requires full, hash-verified artifacts, matching
data/split protocols, and strictly better validation mIoU and small-rock IoU.
Held-out test scores do not select the teacher.

```bash
python3 -S scripts/experiment.py pipeline --config "$KD_CONFIG" \
    --baseline-artifacts "$BASELINE_ARTIFACTS" --teacher-artifacts "$TEACHER_ARTIFACTS"
```

Both `submit` and `pipeline` accept `--seed INTEGER`, a non-bool unsigned
32-bit training seed. It is recorded on GPU stages and applied to resolved
smoke/full configs without modifying source YAMLs or the checked-in data split.
Omitting the override preserves the config seed, whose default remains 42.

Scheduler/storage defaults live in `scripts/clusters/eagle.json`. Select
another JSON profile with `--profile JSON`. Profiles contain account, image,
dataset-parent, output, cache, W&B defaults, and `gpu`/`smoke`/`cpu` resources,
not training settings or credentials. Environment overrides `IMAGE_PATH`,
`DATA_PATH`, `OUTPUT_PATH`, `CACHE_PATH`, `WANDB_MODE`, and `WANDB_PROJECT`
take precedence over profile defaults. Keep writable output and caches in
grant storage; `DATA_PATH` mounts the dataset parent at `/datasets`.

W&B defaults to offline with project `seg-opt`; set `WANDB_ENTITY=ogrodowskijedrzej-put`.
Use `--wandb-mode offline|online|disabled` on managed submissions; online requires
HPC authentication. Never store credentials in source. Offline logging is an
environment setting, not the trainer's `--offline` flag, which disables W&B.

## Durable runs and operations

Successful submission prints a workflow directory. Its `manifest.json`
records stages, paths, source/image hashes and profile; scheduler IDs live in
per-attempt `job.json` records. Source snapshots exclude results, hidden files,
caches, virtual environments and credentials. Artifacts use unique
workflow/stage/attempt paths rather than overwriting configured output directories.
Scientific artifacts retain schema version 2. Do not edit managed snapshots.
Managed jobs clear inherited host `PYTHONPATH`/`PYTHONHOME` and pin both
Singularity/Apptainer Python paths to `/workspace`, the verified source snapshot.
Direct `scripts/train.sbatch --python-path` usage remains available for legacy jobs.

```bash
python3 -S scripts/experiment.py status "$RUN_DIRECTORY"
python3 -S scripts/experiment.py compare "$BASELINE_ARTIFACTS" "$KD_ARTIFACTS"
```

`compare` verifies config, checkpoint and split hashes before
reporting scores and protocol warnings, without ranking methods. Training
settings come only from verified `config.yaml`, parsed with PyYAML (available
in the development container). Stdlib-only `python3 -S` comparison reports
training protocol as unavailable/unverified; unauthenticated protocol sidecars
are not used, even when they include a valid config hash.
Optimizer warnings include weight decay, warmup fraction and backbone learning
rate; output-directory differences are intentionally ignored.

SLURM owns the submitted dependencies, including the terminal summary on
failure. Existing legacy jobs are not automatically adopted by these operations.

Hydra submissions use `scripts/train.sbatch` as the container entry point.

## Controlled CNN Vanilla Protocol

The CNN configs distill ResNet34-U-Net or ResNet101-U-Net into
Fast-SCNN using vanilla logit KD:

| Teacher | `teacher` preset |
| --- | --- |
| ResNet34-U-Net | `benchmark_resnet34_unet` |
| ResNet101-U-Net | `benchmark_resnet101_unet` |

Both preserve the supervised Fast-SCNN model, clean masks, aligned seed-42
split, 512 x 768 images, augmentation, CE 1.0 + Dice 0.5 task loss, and
kernel-7 trimap metrics. Full training uses 80 epochs, physical batch
16, accumulation 1, learning rate 5e-4, weight decay 0.05, warmup fraction
0.05, bf16 precision, and seed 42. KD uses temperature 1, weight 1, and a
stride-8 loss grid; this grid is not the student's output resolution.
Teacher config/checkpoint references and student output directories are
distinct for each teacher. Only student weights are needed at inference.

Run with `pipeline`. Student seed overrides
43 and 44 leave the split and seed-42 teacher fixed. Batch 8 with accumulation
2 is not BatchNorm-equivalent to physical batch 16, so SegFormer runs using
that setting are not controlled CNN loss comparisons. The historical results
below used 20 epochs; current configurations use 80. Compare only matched
training budgets and preserve the original resolved configs on HPC.

## CNN SARD pilot

Two seed-42 ResNet34 to Fast-SCNN ablations. Configs, weights, feature
alignment, and the 20-epoch record versus the 80-epoch YAMLs:
[SARD README](sard/README.md#controlled-cnn-pilot). Do not pick settings from test scores.

## Completed Controlled CNN Results

Historical 20-epoch results, not run instructions. Student checkpoints use
validation mIoU; test scores do not select teachers, methods, or weights.
`analysis.json` records verified checkpoint/config/split hashes and matched
data/training settings, excluding output routing. The fixed split SHA256 is
`e7486b858ee776695b9e5fbdd561ac072e37999d59e801e406c8a299d778c8ab`.
Receipts retain hashes, run/job IDs, and resolved configs independent of account
or artifact root.

### Paired ResNet34 Vanilla KD

The supervised and KD Fast-SCNN arms share student seeds 42, 43, and 44,
with the same split and fixed seed-42 ResNet34 teacher. Scores are
percentages; gains are percentage points (pp).

| Student seed | Supervised test mIoU | ResNet34 KD test mIoU | Paired gain (pp) |
| --- | --- | --- | --- |
| 42 | 83.314508 | 84.047002 | +0.732493 |
| 43 | 82.659167 | 83.230829 | +0.571662 |
| 44 | 83.037347 | 83.572072 | +0.534725 |

Mean paired test mIoU gain is **+0.612960 pp**, with sample SD
**0.105153 pp**. Mean paired test small-rock IoU gain is **+1.319371 pp**
(sample SD 0.233414 pp). All three mIoU and small-rock pairs improve;
the kernel-7 trimap gain is not consistent across seeds (seed 44 decreases).
Three student seeds with one fixed teacher are not a statistical-significance
claim or a replication across teacher seeds.

### Larger Teacher, Single Seed

| KD teacher, student seed 42 | Validation mIoU | Test mIoU | Test small-rock IoU | Test trimap small-rock IoU |
| --- | --- | --- | --- | --- |
| ResNet34-U-Net | 83.299100 | 84.047002 | 67.634463 | 48.056686 |
| ResNet101-U-Net | 83.378357 | 83.859318 | 67.133361 | 47.118571 |

The larger teacher's KD student has +0.079256 pp validation mIoU but
-0.359535 pp validation small-rock IoU versus the smaller teacher's student.
Reported test mIoU is -0.187683 pp. This single-seed, mixed validation result
does not establish that a larger teacher distills better.

### SARD pilot

Table, loss audit, and the validation small-rock delta:
[sard/README.md](sard/README.md#completed-cnn-pilot-results).

## Completed ResNet101 CNN Campaign

This historical campaign predates the 80-epoch norm adopted on 2026-11-17.
It used the Fast-SCNN baseline, ResNet101-U-Net teacher, and vanilla KD protocol
with 100 epochs for every full run; these records retain their actual budget.
Student training seeds are 42/43/44; all share one seed-42 teacher and the fixed
seed-42 data split. The teacher used batch 8 with accumulation 2, LR 3e-4 and
backbone LR 3e-5; students used physical batch 16 and LR 5e-4. Vanilla KD retained
weight 1, temperature 1 and spatial stride 8; warmup fraction remained 0.05.
Checkpoint selection used validation mIoU, not test scores.

| Model | Validation mIoU (%) | Test mIoU (%) | Test small-rock IoU (%) |
| --- | ---: | ---: | ---: |
| ResNet101 teacher, seed 42 | 89.255774 | 89.671141 | 77.254838 |
| Supervised Fast-SCNN, seed 42 | 85.804051 | 86.345065 | 72.204095 |
| Supervised Fast-SCNN, seed 43 | 85.440838 | 86.233538 | 72.430390 |
| Supervised Fast-SCNN, seed 44 | 85.725331 | 86.485678 | 72.465074 |
| Vanilla KD, seed 42 | 85.644251 | 86.387557 | 72.071993 |
| Vanilla KD, seed 43 | 85.696727 | 86.184657 | 72.090346 |
| Vanilla KD, seed 44 | 85.788363 | 86.279631 | 72.117674 |

Mean paired KD-minus-baseline test mIoU is -0.070812 pp; mean validation change
is +0.053040 pp. The teacher has a real advantage, but this recipe does not
consistently transfer it. Three seeds do not establish method-wide failure.
Do not use these held-out test scores to select future temperatures or weights.

The immutable campaign remains on Eagle under
`/mnt/storage_6/project_data/pl1200-01/issue-15-cnn-kd/campaign-100e-20261109/`.
Preserve all manifests, checkpoints, configs, split files and W&B logs there.

SARD calibration is not part of this completed campaign and has not been submitted.

## Validation-only Transfer Diagnosis (2026-11-10)

All seven best-validation checkpoints were strictly restored with verified config,
checkpoint, and split hashes. GPU job 9561983 evaluated 896 validation images, with
no optimizer steps or test inference. Receipts and source snapshot remain on Eagle:
`/mnt/storage_6/project_data/pl1200-01/issue-15-cnn-kd/diagnostics/kd-transfer-20261110/`.
`validation.json` SHA256:
`9862a94343ea1a7fc2b87d9e51789bfb268ccdf6e63781cf5e9fa7c6b86c4df4`.

| Restored prediction | Validation mIoU (%) | Small-rock IoU (%) |
| --- | ---: | ---: |
| ResNet101 teacher | 89.25 | 77.29 |
| Supervised Fast-SCNN, mean of three seeds | 85.66 | 72.45 |
| Vanilla KD Fast-SCNN, mean of three seeds | 85.71 | 72.31 |
| Teacher logits downsampled at stride 8, then bilinearly restored | 88.22 | 75.71 |

Post-hoc bf16-mixed measurements, batch 4; not checkpoint-selection metrics.
Stride-8 downsampling and bilinear restoration are a proxy for lost spatial detail,
not a student result or an upper bound on achievable KD performance.

For supervised seeds 42/43/44, teacher-correct predictions cover
51.05/51.91/51.29% of student errors and 42.66/43.07/41.83% of ground-truth
small-rock errors. Teacher-only/student-only correct pixels are about 2:1:
pixel overlaps, not IoU gains or a deployable oracle; the teacher has unique errors.

The snapshot's `scripts/diagnose_kd.py` measures all trainable student gradients on
validation batches 0/31/63/95/127/159/191/223 (32 images). Weight 1, stride 8,
temperature 1: mean KD/task norm ratios 0.58/0.46, cosines 0.51/0.67,
negative cosine in 4/24 and 3/24 checkpoint-batch pairs (supervised/KD checkpoints).
The sampled signal is neither uniformly negligible nor uniformly opposed to CE+Dice.
Teacher small-rock top-class confidence averages 0.973; entropy is 0.065 nats.

At temperature 4 with unchanged weight, mean norm ratios rise to 9.56/8.50
and cosines fall to 0.03/0.18: calibrate temperature and weight together.
Stride 1 gives similar aggregate gradient directions in this sample; coarsening
alone is not a demonstrated cause. Eval-mode final-checkpoint gradients do
not reconstruct training dynamics or establish one cause.
No calibration training was launched by this diagnosis.

### W&B run organization

Training routes each run to `seg-opt-<dataset>` and group `<dataset>_<method>`
from the resolved Hydra config. For example, Lunar SARD runs use project
`seg-opt-lunar` and group `lunar_sard`. Set `WANDB_PROJECT` or
`WANDB_RUN_GROUP` to override routing. Managed submissions also accept a
project override in the cluster profile.

Run config exposes dataset, method, student, teacher, seed, and run kind.
Tags such as `student:segformer_b2` and `method:sard` mirror these fields
except for seed. Supervised runs omit the teacher tag. In the project UI,
filter `run_kind=full` for comparisons. Filter by method and group by student
to compare models, or filter by student and group by method to compare
methods. Match seeds and, for distillation, teachers when aggregating.

Completed training saves predictions from `best.ckpt` in
`segmentation_samples/`. LUSNAR uses its existing examples manifest. Other
datasets use up to eight validation examples sorted by frame ID. W&B shows
the previews and provides the directory as an artifact when logging is enabled.
`seg-opt-canonical-eval` is reserved for future canonical evaluations.
