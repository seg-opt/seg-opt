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

```bash
python3 -S scripts/experiment.py status "$RUN_DIRECTORY"
python3 -S scripts/experiment.py retry "$RUN_DIRECTORY"
python3 -S scripts/experiment.py compare "$BASELINE_ARTIFACTS" "$KD_ARTIFACTS"
```

`retry` creates new attempts for terminal failed/blocked stages, preserving
completed stages. It rejects live or unknown scheduler states and does not
resume a training checkpoint. Inspect status first. `compare` reports existing
scores and protocol warnings without ranking methods; matching report formats
alone do not establish controlled comparisons.

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
