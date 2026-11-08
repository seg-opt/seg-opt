# SARD: structure-aware representation distillation

Tests Structure-Aware Representation Distillation (Liu et al., CVPR 2026) on
the clean artificial lunar landscape masks. SARD aligns student and teacher
*features* and weights every location by a structure map computed from the
frozen teacher's features: structure-tensor anisotropy (edges), its geometric
mean (corners) and local feature dispersion (crowding). Small, densely packed
rocks are the paper's target regime.

Teacher: SegFormer-B4 (`experiments/baselines_benchmark/segformer_b4.yaml`),
tapped on its fused decoder features at stride 4 (768 channels). Student:
Fast-SCNN, tapped on its fused features at stride 8 (128 channels). Both are
aligned on the stride-8 grid. Only the student runs at inference; checkpoints
contain student weights only.

Implementation choices the paper leaves open are documented at the top of
`src/distillation/sard.py`.

## Runs

| Array task | Variant | Config |
|---:|---|---|
| 0 | Task-only Fast-SCNN control | `baselines_benchmark/fast_scnn.yaml` |
| 1 | Vanilla logit KD | `logit_distillation/fast_scnn_from_segformer_b4.yaml` |
| 2 | Uniform SARD, `W(i) = 1` | `sard/fast_scnn_sard_uniform_from_segformer_b4.yaml` |
| 3 | Full SARD, `beta = (2, 1, 1)` | `sard/fast_scnn_sard_full_from_segformer_b4.yaml` |

Paper defaults: `lambda_f = 1`, `lambda_d = 0.5`, `mu = 1`, `alpha = 0.5`,
`r = 7`, denoiser hidden width 128.

## Submit

From the repository root on Eagle:

```bash
: "${IMAGE_PATH:?Set the development SIF path}"
: "${DATA_PATH:?Set the dataset-parent path}"
mkdir -p results
bash experiments/sard/submit.sh                 # smoke -> teacher -> array
bash experiments/sard/submit.sh --skip-teacher  # teacher checkpoint exists
```

## Interpretation

The SARD question is answered by task 3 versus task 2 (does structure
weighting help over uniform feature KD?) and task 3 versus tasks 0 and 1
(does it beat no KD and logit KD?). Report validation mIoU, small-rock IoU and
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

| Variant | Config relative to experiments/ |
| --- | --- |
| BPKD body-only, weights 20/0 | bpkd/fast_scnn_from_resnet34_unet_body_only.yaml |
| BPKD edge-only, weights 0/50 | bpkd/fast_scnn_from_resnet34_unet_edge_only.yaml |
| BPKD combined, weights 20/50, window 7 | bpkd/fast_scnn_from_resnet34_unet_bpkd_w7.yaml |
| Uniform SARD | sard/fast_scnn_sard_uniform_from_resnet34_unet.yaml |
| Full SARD | sard/fast_scnn_sard_full_from_resnet34_unet.yaml |

Use the shared artifact-verifying `pipeline` interface for CNN configs,
not the SegFormer-specific SARD wrapper above. It accepts the selected
`--config`, completed `--baseline-artifacts`, and `--teacher-artifacts`.
The validation teacher gate checks matching data/split and real full-run
checkpoint/config/split hashes before KD. Managed runs retain immutable
source snapshots and unique attempt outputs; existing artifacts are never
overwritten. Per-study `study.json`, `comparison.json`, and audited
`analysis.json` receipts record run IDs, job IDs, artifact locations,
hashes, and resolved protocols in the chosen study directory.

BPKD weights and SARD settings are transferred starting values from another
dataset protocol, not tuned ResNet34 optima. SARD uses feature/distribution
weights 1/0.5, noise alpha 0.5, radius 7, projection width 256, denoiser
width 128, and stride 8. Full/uniform differ only in structure weights
(2/1/1 versus 0/0/0). This is a five-variant single-seed pilot. Do not select
settings using test metrics or treat the SegFormer-B4 runs (batch 8,
accumulation 2) as controlled CNN loss comparisons. Future 100-epoch full
runs need matched supervised controls and a separate validation-based
protocol; the historical YAMLs remain at 20 epochs and no new runs are
submitted in this packaging task.
