# Experiment map

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

The historical CNN configs distill ResNet34-U-Net or ResNet101-U-Net into
Fast-SCNN using vanilla logit KD:

| Teacher | Config relative to experiments/ |
| --- | --- |
| ResNet34-U-Net | logit_distillation/fast_scnn_from_resnet34_unet.yaml |
| ResNet101-U-Net | logit_distillation/fast_scnn_from_resnet101_unet.yaml |

Both preserve the supervised Fast-SCNN model, clean masks, aligned seed-42
split, 512 x 768 images, augmentation, CE 1.0 + Dice 0.5 task loss, and
kernel-7 trimap metrics. Historical training uses 20 epochs, physical batch
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
that setting are not controlled CNN loss comparisons. These checked-in
20-epoch configs describe historical experiments, not a new full-run budget.
Future full runs target 100 epochs with matched supervised controls under a
separate validation-based protocol; none are submitted by this packaging task.

## Controlled CNN BPKD/SARD Pilot Protocol

The five historical seed-42 ablations reuse the same ResNet34 teacher and
Fast-SCNN supervised/vanilla references. All preserve the CNN data, task
loss, metrics, and 20-epoch physical-batch-16/accumulation-1 training protocol
above; only the distillation method/settings and output routing differ.

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
during reporting. Any future 100-epoch study needs matched controls and a
protocol fixed using validation before viewing new test outcomes.
