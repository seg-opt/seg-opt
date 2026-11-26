# SARD: structure-aware representation distillation

Tests Structure-Aware Representation Distillation (Liu et al., CVPR 2026) on
the clean artificial lunar landscape masks. SARD aligns student and teacher
*features* and weights every location by a structure map computed from the
frozen teacher's features: structure-tensor anisotropy (edges), its geometric
mean (corners) and local feature dispersion (crowding). Small, densely packed
rocks are the paper's target regime.

Teacher: SegFormer-B4 (`baselines_benchmark/segformer_b4`),
tapped on its fused decoder features at stride 4 (768 channels). Student:
Fast-SCNN, tapped on its fused features at stride 8 (128 channels). Both are
aligned on the stride-8 grid. Only the student runs at inference; checkpoints
contain student weights only.

Implementation choices the paper leaves open are documented at the top of
`src/distillation/sard.py`.

## Runs

| Variant | `method` | `teacher` |
|---|---|---|
| Task-only Fast-SCNN control | `supervised` | `none` |
| Vanilla logit KD | `vanilla` | `benchmark_segformer_b4` |
| Uniform SARD, `W(i) = 1` | `sard` | `benchmark_segformer_b4` |
| Full SARD, `beta = (2, 1, 1)` | `sard` | `benchmark_segformer_b4` |

For uniform SARD, set `distillation.sard_edge_weight=0`,
`distillation.sard_corner_weight=0`, and `distillation.sard_density_weight=0`.

Paper defaults: `lambda_f = 1`, `lambda_d = 0.5`, `mu = 1`, `alpha = 0.5`,
`r = 7`, denoiser hidden width 128.

## Submit

From the repository root on Eagle:

```bash
: "${IMAGE_PATH:?Set the development SIF path}"
: "${DATA_PATH:?Set the dataset-parent path}"
scripts/submit --disable-wandb dataset=lunar method=supervised teacher=none student=fast_scnn \
  training.batch_size=16 training.accumulate_grad_batches=1 training.learning_rate=0.0005 \
  training.backbone_learning_rate=null
scripts/submit --disable-wandb dataset=lunar method=sard teacher=benchmark_segformer_b4 \
  student=fast_scnn training.learning_rate=0.0005 training.backbone_learning_rate=null
```

Submit the other KD variants from the table the same way. When training a new
teacher, pass its full job ID as `--after-job` and the container-visible path
to its checkpoint as `distillation.teacher.checkpoint_path` on each KD submit.

## Interpretation

Compare full SARD with uniform SARD to isolate structure weighting, then with
the supervised control and vanilla logit KD. Report validation mIoU, small-rock IoU and
Trimap (kernel 7) small-rock IoU; the paper's claim is that boundary metrics
gain more than mIoU. Seed 42 alone is directional, not conclusive.

## Controlled CNN Pilot

The historical ResNet34-U-Net -> Fast-SCNN pilot reuses the completed
seed-42 teacher, supervised student, and vanilla KD reference. The clean
masks, aligned split, 512 x 768 images, augmentation, CE + Dice task loss,
metrics, optimizer, seed, and 20 epochs are held fixed. Physical batch 16
with accumulation 1 is preserved, not replaced with batch 8/accumulation 2.
ResNet34 provides last-decoder features (32 channels, stride 2); Fast-SCNN
provides fused features (128 channels, stride 8). SARD aligns both on the
stride-8 grid. Only the student runs at inference; no teacher retraining is
needed for this historical pilot.

Use `dataset=lunar teacher=benchmark_resnet34_unet student=fast_scnn` with
`training.batch_size=16 training.accumulate_grad_batches=1` and
`training.learning_rate=0.0005 training.backbone_learning_rate=null`.

| Variant | Method and overrides |
| --- | --- |
| Uniform SARD | `method=sard` with the three structure weights set to zero |
| Full SARD | `method=sard` |

Use the [managed pipeline](../README.md#managed-artifact-validation-and-historical-runs) with the CNN settings.
Gate, hashes, and snapshot outputs: [experiment guide](../README.md).

The SARD settings come from another dataset protocol. SARD uses feature/distribution
weights 1/0.5, noise alpha 0.5, radius 7, projection width 256, denoiser
width 128, and stride 8. Full/uniform differ only in structure weights
(2/1/1 versus 0/0/0).
This is a two-variant single-seed pilot. Do not select
settings using test metrics or treat the SegFormer-B4 runs (batch 8,
accumulation 2) as controlled CNN loss comparisons. Future 80-epoch full
runs need matched supervised controls and a validation-based protocol. The
YAMLs now use 80 epochs; the historical 20-epoch protocol remains
in the immutable run snapshots and resolved artifact configs.

## Completed CNN Pilot Results

The four student artifact sets and ResNet34 teacher passed checkpoint,
config, split-hash, and validation-selection verification. Data metadata and
training settings match after excluding only intended output-directory
differences. All students have 1,135,860 parameters. Scores are percentages;
all rows use student seed 42, the same fixed seed-42 teacher, 20 epochs,
and checkpoints selected by validation mIoU. Test metrics are reporting only.

| Variant | Validation mIoU | Test mIoU | Test small-rock IoU | Test large-rock IoU | Test trimap small-rock IoU |
| --- | --- | --- | --- | --- | --- |
| Supervised | 82.611269 | 83.314508 | 66.361755 | 69.196588 | 47.058216 |
| Vanilla KD | 83.299100 | 84.047002 | 67.634463 | 70.757794 | 48.056686 |
| Uniform SARD | 83.520544 | 84.086227 | 67.299175 | 71.231902 | 47.583976 |
| Full SARD | 80.147094 | 80.742520 | 61.827767 | 63.904828 | 43.336093 |

Uniform SARD improves reported test mIoU by +0.771719 pp versus baseline
and +0.039226 pp versus vanilla, a single-seed near-tie, not superiority.
Validation mIoU is +0.221443 pp versus vanilla, while validation small-rock
IoU is -0.260532 pp. Full SARD reduces validation mIoU by 3.373450 pp and
test mIoU by 3.343707 pp relative to uniform; structural weighting did not
help this pilot. The transferred weights, feature representation, fixed
teacher, and single student seed limit the conclusion.

### Offline Loss Audit

The existing CPU audit parsed binary offline W&B histories without cloud
sync or retraining. Each of the three distilled runs has 20 logged epoch loss
records; no non-finite **logged training values** were found. Last-epoch
mean scalar values and weighted KD/task ratios are:

| Variant | Task loss | Weighted distillation loss | Distillation / task |
| --- | --- | --- | --- |
| Vanilla KD | 0.154614 | 0.020244 | 0.13093 |
| Uniform SARD | 0.154179 | 0.397185 | 2.5761 |
| Full SARD | 0.184459 | 0.628394 | 3.4067 |

Ratios use unrounded receipt values. Full SARD's epoch-0
weighted feature term is 7.345402 versus 1.051878 for uniform. Structure
cues, weight concentration, feature normalization, and loss balance remain
diagnostic hypotheses, not confirmed causes. Finite logged scalars do not
prove that every intermediate value or gradient was finite.

The study's `study.json`, automatic `comparison.json`, and audited
`analysis.json` preserve hashes, artifact/run/job references, large-rock
scores, baseline/vanilla deltas, and offline loss samples. Historical analysis
and artifacts remain unchanged. No additional training or hyperparameter
sweep was submitted during analysis or packaging; any future study needs a
validation-based protocol before viewing new test results.
