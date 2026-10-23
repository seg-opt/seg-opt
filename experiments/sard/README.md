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
export IMAGE_PATH="$HOME/pl1200-01/project_data/containers/seg-opt/development/current.sif"
export DATA_PATH="$HOME/pl1200-01/project_data/datasets"
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
