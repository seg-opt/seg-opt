# Experiment map

## Training budget

100 epochs is the default budget for active full runs, not a separate experiment
family. Configure baselines in `baselines_benchmark/`, vanilla KD in
`logit_distillation/`, and method variants in `bpkd/`, `sard/`, or
`strong_teacher_distillation/`. Smoke limits are unchanged. The explicitly named
`dinov3_mask2former/lunar_teacher_80e.yaml` remains a historical recipe, not an
active default.

Historical results keep their actual 20- or 80-epoch budgets. Their immutable
source snapshots and resolved artifact configs preserve the original protocol;
updating canonical YAML defaults must not rewrite those records. Do not mix
budgets in controlled comparisons or reuse unmatched teacher/control artifacts.
Use the managed runner's unique outputs, not direct training into historical
standalone output directories.

## Clean baseline benchmark

[`baselines_benchmark/`](baselines_benchmark/) compares the Fast-SCNN student
with six teacher candidates on clean masks and the aligned seed-42 split.
The default benchmark remains seven models; ResNet101 is an optional,
explicitly selected extra teacher. Establish a positive validation teacher
margin before distillation.

## BPKD

[`bpkd/`](bpkd/) contains the prepared BPKD experiment matrix. Do not submit
it until the clean baseline benchmark establishes a positive teacher margin.

## SARD

[`sard/`](sard/) tests structure-aware feature distillation from a SegFormer-B4
teacher into Fast-SCNN against task-only, vanilla logit KD and uniform-weight
controls.

## Exploratory work

[`dinov3_demo/`](dinov3_demo/) and [`dinov3_probe_v2/`](dinov3_probe_v2/) are
standalone DINOv3 exploration, not part of the benchmark decision path.

## Shared CLI and cluster profile

Run these commands from the deployed repository root on HPC. The runner uses
only the Python standard library; on Eagle, use `python3 -S` to bypass site
hooks. Tests, training, and dependency installation belong on HPC only.

```bash
python3 -S scripts/experiment.py submit experiments/baselines_benchmark/fast_scnn.yaml
python3 -S scripts/experiment.py submit \
	experiments/baselines_benchmark/fast_scnn.yaml \
	experiments/baselines_benchmark/resnet34_unet.yaml --seed 43
```

Each config gets a smoke check before full training. Plain `submit` does not
establish a scientific teacher-quality gate. `pipeline` submits teacher smoke,
teacher training, a CPU validation gate, KD smoke, KD training, and a terminal
summary. Supply an existing KD config with `--config`, completed supervised
student artifacts with `--baseline-artifacts`, and either `--teacher-config`
or `--teacher-artifacts`. Adopted teacher artifacts can optionally use
`--after-job JOB_ID`; the gate requires full, hash-verified artifacts, matching
data/split protocols, and strictly better validation mIoU and small-rock IoU.
Held-out test scores do not select the teacher.

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

W&B defaults to offline with project `seg-opt`. Use `--wandb-mode
offline|online|disabled` on managed submissions; online requires HPC
authentication. Never store credentials in source. Offline logging is an
environment setting, not the trainer's `--offline` flag, which disables W&B.

## Durable runs and operations

Successful submission prints a workflow directory. Its `manifest.json`
records stages, paths, source/image hashes and profile; scheduler IDs live in
per-attempt `job.json` records. Source snapshots exclude results, hidden files,
caches, virtual environments and credentials. Artifacts use unique
workflow/stage/attempt paths rather than overwriting canonical config outputs.
Scientific artifacts retain schema version 2. Do not edit managed snapshots.
Managed jobs clear inherited host `PYTHONPATH`/`PYTHONHOME` and pin both
Singularity/Apptainer Python paths to `/workspace`, the verified source snapshot.
Direct `scripts/train.sbatch --python-path` usage remains available for legacy jobs.

```bash
python3 -S scripts/experiment.py status "$RUN_DIRECTORY"
python3 -S scripts/experiment.py retry "$RUN_DIRECTORY"
python3 -S scripts/experiment.py compare "$BASELINE_ARTIFACTS" "$KD_ARTIFACTS"
```

`retry` creates new attempts for terminal failed/blocked stages and all their
downstream dependents, including completed dependents. Old attempts and artifacts
remain untouched; unaffected completed ancestors are reused. It rejects live or
unknown scheduler states and does not resume a training checkpoint. Inspect
status first. `compare` verifies config, checkpoint and split hashes before
reporting scores and protocol warnings, without ranking methods. Training
settings come only from verified `config.yaml`, parsed with PyYAML (available
in the development container). Stdlib-only `python3 -S` comparison reports
training protocol as unavailable/unverified; unauthenticated protocol sidecars
are not used, even when they include a valid config hash. Schema 2 is unchanged.
Optimizer warnings include weight decay, warmup fraction and backbone learning
rate; output-directory differences are intentionally ignored. Matching report
formats alone do not establish controlled comparisons.

SLURM owns the submitted dependencies, including the terminal summary on
failure. After successful submission, no laptop-side wait, email setup, or
continuously connected assistant is needed. Existing legacy jobs are not
automatically adopted by these operations.

## Legacy compatibility

The ten existing launchers in `baselines_benchmark/`, `bpkd/`, and
`logit_distillation/` forward to shared `submit` or `launch [--smoke]`.
Explicit config lists, including quoted paths with spaces, are forwarded
unchanged. With no arguments, generic logit wrappers retain B4-to-Fast-SCNN,
and the specific B0 wrappers retain B4-to-B0. Baseline training retains its
ordered seven-model list; BPKD retains eight configs; the Mask2Former screen
retains two. Baseline smoke defaults to Fast-SCNN. Resource headers and suite
project/group defaults remain available; explicit `WANDB_MODE`,
`WANDB_PROJECT`, and `WANDB_RUN_GROUP` settings are preserved. Shared launch
also honors `WANDB_GROUP`, defaulting it to the selected config stem.

Direct `sbatch` launches one config per array task. Adjust `--array` when
overriding a suite list; shared launch validates the array index:

```bash
mkdir -p results
sbatch --array=0-1%1 experiments/baselines_benchmark/train.sbatch \
	experiments/baselines_benchmark/fast_scnn.yaml \
	experiments/baselines_benchmark/resnet34_unet.yaml
bash experiments/logit_distillation/submit.sh "$CONFIG_WITH_SPACES"
```

Repository resolution uses `REPO_PATH`, then `SLURM_SUBMIT_DIR`, then the
original script's repository root. Submit direct jobs from the real repository
root or export `REPO_PATH`: a Slurm-spooled script cannot infer the checkout
from its temporary location. Wrappers change to that root and propagate the
shared command's exit status.

Direct training/smoke launchers do not create managed snapshots or a
teacher-quality gate. Smoke outputs use the trainer's job-specific paths
rather than fixed legacy smoke directories. Legacy submitters delegate to
managed `submit`; inspect its printed workflow directory and `status` for
job IDs instead of relying on the former two-job output format. Prefer
`pipeline` when KD requires the validation gate.

## Controlled CNN Vanilla Protocol

The canonical CNN configs distill ResNet34-U-Net or ResNet101-U-Net into
Fast-SCNN using vanilla logit KD:

| Teacher | Config relative to experiments/ |
| --- | --- |
| ResNet34-U-Net | logit_distillation/fast_scnn_from_resnet34_unet.yaml |
| ResNet101-U-Net | logit_distillation/fast_scnn_from_resnet101_unet.yaml |

Both preserve the supervised Fast-SCNN model, clean masks, aligned seed-42
split, 512 x 768 images, augmentation, CE 1.0 + Dice 0.5 task loss, and
kernel-7 trimap metrics. Full training uses 100 epochs, physical batch
16, accumulation 1, learning rate 5e-4, weight decay 0.05, warmup fraction
0.05, bf16 precision, and seed 42. KD uses temperature 1, weight 1, and a
stride-8 loss grid; this grid is not the student's output resolution.
Teacher config/checkpoint references and student output directories are
distinct for each teacher. Only student weights are needed at inference.

Use the shared `pipeline` interface with completed, hash-verified supervised
student and teacher artifacts. Its validation mIoU and small-rock IoU gate
must pass before KD; test scores are reporting only. Student seed overrides
43 and 44 leave the split and seed-42 teacher fixed. Batch 8 with accumulation
2 is not BatchNorm-equivalent to physical batch 16, so SegFormer runs using
that setting are not controlled CNN loss comparisons. The historical results
below used 20 epochs; current canonical YAMLs use 100. Compare only matched
training budgets and preserve the original resolved configs on HPC.

## Controlled CNN BPKD/SARD Pilot Protocol

The five historical seed-42 ablations reuse the same ResNet34 teacher and
Fast-SCNN supervised/vanilla references. Their completed runs preserved the CNN
data, task loss, metrics, and a 20-epoch physical-batch-16/accumulation-1 training
protocol; only distillation settings and output routing differed. The canonical
configs now use the standard 100-epoch budget with matched controls.

| Variant | Config relative to experiments/ |
| --- | --- |
| BPKD body-only, weights 20/0 | bpkd/fast_scnn_from_resnet34_unet_body_only.yaml |
| BPKD edge-only, weights 0/50 | bpkd/fast_scnn_from_resnet34_unet_edge_only.yaml |
| BPKD combined, weights 20/50 | bpkd/fast_scnn_from_resnet34_unet_bpkd_w7.yaml |
| Uniform SARD | sard/fast_scnn_sard_uniform_from_resnet34_unet.yaml |
| Full SARD | sard/fast_scnn_sard_full_from_resnet34_unet.yaml |

BPKD retains temperature 1, edge alpha 2, window 7, PRM and POM. SARD uses
feature/distribution weights 1/0.5, noise alpha 0.5, density radius 7,
projection width 256, and denoiser width 128. Full versus uniform changes
only structure weights (2/1/1 versus 0/0/0). These are transferred starting
values from another dataset protocol, not calibrated optima. No weights,
features, or loss implementations are changed here. See [the SARD study
protocol](sard/README.md#controlled-cnn-pilot) for feature alignment and
artifact provenance. This single-seed pilot does not authorize further
training or support hyperparameter selection using held-out test scores.

## Completed Controlled CNN Results

These are completed historical 20-epoch studies, not new run instructions.
All student checkpoints were selected by validation mIoU; held-out test
metrics are reported without selecting teachers, methods, or weights by
test performance. Checkpoint/config/split hashes, identical data metadata,
and matched training settings (excluding only output routing) were verified
in each study's audited `analysis.json`. The fixed split SHA256 is
`e7486b858ee776695b9e5fbdd561ac072e37999d59e801e406c8a299d778c8ab`.
Per-study receipts retain artifact hashes, run/job IDs and resolved config
provenance without requiring a particular account or artifact-root path.

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

### BPKD/SARD Pilot

The [completed seven-row pilot and offline loss audit](sard/README.md#completed-cnn-pilot-results)
compare the seed-42 baseline and vanilla reference with five transferred,
untuned BPKD/SARD settings. Uniform SARD's reported test mIoU is
+0.771719 pp versus baseline and only +0.039226 pp versus vanilla, a
near-tie rather than evidence of superiority. Full SARD and all three BPKD
settings regress on validation as well as test. Large logged scalar KD/task
ratios motivate validation-based normalization/weight checks, not a causal
claim about gradients or a general failure claim about either method.
No additional training, hyperparameter calibration, or sweep was performed
during historical reporting. Additional studies need matched controls and a
protocol fixed using validation before viewing new test outcomes.

## Completed ResNet101 CNN Campaign

The completed campaign used the canonical Fast-SCNN baseline, ResNet101-U-Net
teacher, and vanilla KD protocol above, with 100 epochs for every full run.
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
That historical storage name and its recorded source/config hashes do not
require a separate experiment family in the current repository. Preserve all
manifests, checkpoints, configs, split files and W&B logs there.

For further approved replications, use the existing canonical paths above and
the snapshot-owned runner. Set `WANDB_ENTITY=ogrodowskijedrzej-put`, project
`seg-opt`, and the appropriate logging mode; credentials stay on HPC. Pass
`--seed` to `submit` or `pipeline`, not a new epoch-specific config copy. For
adopted teacher artifacts, connect `pipeline --after-job` to a successful
completion barrier for both the teacher and paired supervised baseline. Do not
depend only on the teacher or bypass the validation gate. BPKD/SARD calibration
is not part of this completed campaign and has not been submitted.

## Validation-only Transfer Diagnosis (2026-11-10)

All seven completed best-validation checkpoints were restored strictly and their
config, checkpoint, and split hashes verified. GPU job 9561983 evaluated all 896
validation images, without optimizer steps or test-set inference. Raw receipts
and the source snapshot remain on Eagle under
`/mnt/storage_6/project_data/pl1200-01/issue-15-cnn-kd/diagnostics/kd-transfer-20261110/`.
The report is `validation.json` (SHA256
`9862a94343ea1a7fc2b87d9e51789bfb268ccdf6e63781cf5e9fa7c6b86c4df4`).

| Restored prediction | Validation mIoU (%) | Small-rock IoU (%) |
| --- | ---: | ---: |
| ResNet101 teacher | 89.25 | 77.29 |
| Supervised Fast-SCNN, mean of three seeds | 85.66 | 72.45 |
| Vanilla KD Fast-SCNN, mean of three seeds | 85.71 | 72.31 |
| Teacher logits downsampled at stride 8, then bilinearly restored | 88.22 | 75.71 |

These are post-hoc bf16-mixed measurements at batch size 4, not replacements for
the original checkpoint-selection metrics; small numerical differences are
expected. Coarsening is a visualization proxy for lost spatial detail, not a
student result or an exact upper bound on achievable KD performance.

For supervised seeds 42/43/44, the teacher correctly predicts 51.05/51.91/51.29%
of the pixels those students misclassify. On ground-truth small-rock pixels, it
corrects 42.66/43.07/41.83% of their errors. Teacher-only correct pixels outnumber
student-only correct pixels by about 2:1 overall. These are pixel overlaps, not
IoU gains or a deployable oracle; the teacher also makes unique errors.

The diagnostic source snapshot's `scripts/diagnose_kd.py` measures gradients over
every trainable student parameter on validation batches
0/31/63/95/127/159/191/223 (32 images). At weight 1, stride 8, temperature 1,
mean KD/task gradient-norm ratios are 0.58 for supervised checkpoints and 0.46
for KD checkpoints; mean cosine similarities are 0.51 and 0.67. Negative cosine
occurs in 4/24 and 3/24 checkpoint-batch pairs. The current KD signal is therefore
neither uniformly negligible nor uniformly opposed to CE+Dice in this sample.
Teacher small-rock predictions are sharp: mean top-class confidence 0.973 and
entropy 0.065 nats.

Temperature 4 at unchanged weight raises those mean norm ratios to 9.56/8.50
and reduces mean cosines to 0.03/0.18. Temperature and weight must be calibrated
together; raising temperature alone is not justified by these snapshots.
Stride 1 gives similar aggregate parameter-gradient directions in this sample,
so spatial coarsening alone does not establish a cause of the KD result.

These eval-mode, final-checkpoint gradients do not reconstruct training dynamics
or identify one causal failure. A next experiment should use a small,
validation-selected weight/stride ablation with matched controls and a separate
confirmation seed. No calibration training was launched by this diagnosis.
