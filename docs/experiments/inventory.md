| Inventory scope | Details |
|---|---|
| Results checked | 15 November 2026 through read-only HPC access |
| Numeric results | Taken from saved `metrics.json`; full-run rows have `run_kind: full`, validation/test metrics, and an existing best checkpoint |
| Units | All IoU values are percentages, rounded to two decimals; epoch budget is the configured maximum, not the selected checkpoint epoch |
| Dataset shorthand | Lunar means Artificial Lunar Landscape with clean masks |
| Boundary diagnostic | Trimap small-rock IoU uses the recorded boundary metric; → means unavailable or not applicable |
| Missing results | Not located in accessible storage; this does not establish that training never happened |
| Access limits | The teammate HPC directory is inaccessible; W&B cloud history was not queried |
| Duplicate handling | Five shared SARD copies and the repeated lunar seed-42 Fast-SCNN baseline are excluded from the result tables; matching metrics were checked, checkpoint hashes were not compared |
| Comparisons | Compare matched datasets, splits, seed, and training setup; the 20-epoch and 100-epoch CNN campaigns are separate studies |
| Evidence labels | SARD lunar/benchmark identifies the SARD workspace; CNN 20e canonical identifies its original runs; timestamped IDs identify individual CNN runner experiments |
| Table order | Model roles; methods; verified supervised results; verified distillation results; unavailable configured results; smoke checks; configuration references |

| Model | Teacher role | Student role |
|---|---|---|
| SegFormer-B4 | Teacher for vanilla logit KD, SARD, and CAP | — |
| Mask2Former Swin-L | Teacher for vanilla KD | Same-architecture distillation student |
| SegFormer-B0 | Baseline teacher candidate | Student of SegFormer-B4 |
| SegFormer-B2 | Baseline teacher candidate | — |
| ResNet34–U-Net | Baseline teacher candidate | — |
| DINOv3 ViT-L/16 | Baseline teacher candidate; frozen backbone with linear segmentation head | — |
| Fast-SCNN | — | Main lightweight student |

| Method | Variants present |
|---|---|
| Vanilla pixel/logit KD | Temperature-scaled pixelwise KL divergence |
| SARD — Structure-Aware Representation Distillation | Full structure weighting; uniform-weight control |
| CAP — Cross-Attention Projector feature distillation | ScaleKD-derived feature loss added to vanilla logit KD |

| Model / teacher → student | Method | Dataset | Seed | Epoch budget | Val mIoU (%) | Test mIoU (%) | Test small-rock IoU (%) | Test Trimap small-rock IoU (%) | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| Fast-SCNN | Supervised / no KD | ADE20K | 42 | 20 | 12.30 | 12.62 | — | — | SARD benchmark |
| SegFormer-B4 | Supervised / no KD | ADE20K | 42 | 20 | 45.63 | 45.78 | — | — | SARD benchmark |
| Fast-SCNN | Supervised / no KD | Cityscapes | 42 | 50 | 49.09 | 48.12 | — | — | SARD benchmark |
| SegFormer-B4 | Supervised / no KD | Cityscapes | 42 | 50 | 75.42 | 76.08 | — | — | SARD benchmark |
| Fast-SCNN | Supervised / no KD | Lunar | 42 | 20 | 82.61 | 83.31 | 66.36 | 47.06 | SARD lunar |
| Fast-SCNN | Supervised / no KD | Lunar | 43 | 20 | 82.11 | 82.66 | 64.59 | 45.60 | CNN 20e `20261027T173808Z-a496f8c0bcd7` |
| Fast-SCNN | Supervised / no KD | Lunar | 44 | 20 | 82.53 | 83.04 | 65.52 | 46.46 | CNN 20e `20261027T173841Z-d7f305b45c82` |
| ResNet101–U-Net | Supervised / no KD | Lunar | 42 | 20 | 88.87 | 89.37 | 76.97 | 60.87 | CNN 20e canonical |
| ResNet34–U-Net | Supervised / no KD | Lunar | 42 | 20 | 88.39 | 88.88 | 76.19 | 59.48 | CNN 20e canonical |
| SegFormer-B4 | Supervised / no KD | Lunar | 42 | 20 | 90.13 | 90.35 | 78.85 | 61.39 | SARD lunar |
| Fast-SCNN | Supervised / no KD | Lunar | 42 | 100 | 85.80 | 86.35 | 72.20 | 54.64 | CNN 100e `20261108T234227Z-2ee5c84aba50` |
| Fast-SCNN | Supervised / no KD | Lunar | 43 | 100 | 85.44 | 86.23 | 72.43 | 54.59 | CNN 100e `20261108T234307Z-9689d6b4a405` |
| Fast-SCNN | Supervised / no KD | Lunar | 44 | 100 | 85.73 | 86.49 | 72.47 | 54.97 | CNN 100e `20261108T234318Z-595903bc4273` |
| ResNet101–U-Net | Supervised / no KD | Lunar | 42 | 100 | 89.26 | 89.67 | 77.25 | 60.83 | CNN 100e `20261108T234227Z-2ee5c84aba50` |

| Model / teacher → student | Method | Dataset | Seed | Epoch budget | Val mIoU (%) | Test mIoU (%) | Test small-rock IoU (%) | Test Trimap small-rock IoU (%) | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| SegFormer-B4 → Fast-SCNN | Full SARD | ADE20K | 42 | 20 | 12.95 | 13.02 | — | — | SARD benchmark |
| SegFormer-B4 → Fast-SCNN | Full SARD | ADE20K | 43 | 20 | 12.32 | 12.75 | — | — | SARD benchmark |
| SegFormer-B4 → Fast-SCNN | Full SARD | ADE20K | 44 | 20 | 12.68 | 12.83 | — | — | SARD benchmark |
| SegFormer-B4 → Fast-SCNN | Full SARD | Cityscapes | 42 | 50 | 51.48 | 49.63 | — | — | SARD benchmark |
| SegFormer-B4 → Fast-SCNN | Full SARD | Cityscapes | 43 | 50 | 51.89 | 49.70 | — | — | SARD benchmark |
| SegFormer-B4 → Fast-SCNN | Full SARD | Cityscapes | 44 | 50 | 51.23 | 49.70 | — | — | SARD benchmark |
| ResNet101–U-Net → Fast-SCNN | Vanilla logit KD | Lunar | 42 | 20 | 83.38 | 83.86 | 67.13 | 47.12 | CNN 20e `20261027T173725Z-83e7ac9f5077` |
| ResNet34–U-Net → Fast-SCNN | Full SARD | Lunar | 42 | 20 | 80.15 | 80.74 | 61.83 | 43.34 | CNN 20e `20261027T212100Z-6ca92fe48679` |
| ResNet34–U-Net → Fast-SCNN | Uniform SARD | Lunar | 42 | 20 | 83.52 | 84.09 | 67.30 | 47.58 | CNN 20e `20261027T212405Z-c9ce3924034b` |
| ResNet34–U-Net → Fast-SCNN | Vanilla logit KD | Lunar | 42 | 20 | 83.30 | 84.05 | 67.63 | 48.06 | CNN 20e canonical |
| ResNet34–U-Net → Fast-SCNN | Vanilla logit KD | Lunar | 43 | 20 | 82.94 | 83.23 | 66.17 | 46.37 | CNN 20e `20261027T173825Z-df9d6afc3fe8` |
| ResNet34–U-Net → Fast-SCNN | Vanilla logit KD | Lunar | 44 | 20 | 83.02 | 83.57 | 66.63 | 46.36 | CNN 20e `20261027T173850Z-f7644000adbc` |
| SegFormer-B4 → Fast-SCNN | Full SARD | Lunar | 42 | 20 | 83.74 | 84.27 | 67.68 | 47.87 | SARD lunar |
| SegFormer-B4 → Fast-SCNN | Full SARD | Lunar | 43 | 20 | 83.58 | 84.03 | 67.52 | 48.14 | SARD lunar |
| SegFormer-B4 → Fast-SCNN | Full SARD | Lunar | 44 | 20 | 83.39 | 84.11 | 67.17 | 47.05 | SARD lunar |
| SegFormer-B4 → Fast-SCNN | Uniform SARD | Lunar | 42 | 20 | 83.69 | 84.29 | 67.50 | 47.21 | SARD lunar |
| SegFormer-B4 → Fast-SCNN | Vanilla logit KD | Lunar | 42 | 20 | 82.96 | 83.68 | 66.37 | 45.82 | SARD lunar |
| ResNet101–U-Net → Fast-SCNN | Vanilla logit KD | Lunar | 42 | 100 | 85.64 | 86.39 | 72.07 | 52.84 | CNN 100e `20261108T234408Z-edabcc89e030` |
| ResNet101–U-Net → Fast-SCNN | Vanilla logit KD | Lunar | 43 | 100 | 85.70 | 86.18 | 72.09 | 53.73 | CNN 100e `20261108T234458Z-458030dd42e7` |
| ResNet101–U-Net → Fast-SCNN | Vanilla logit KD | Lunar | 44 | 100 | 85.79 | 86.28 | 72.12 | 53.22 | CNN 100e `20261108T234511Z-0732b0c0cb4d` |

| Configured model / teacher → student | Method / variant | Dataset | Seed | Val mIoU (%) | Test mIoU (%) | Result status |
|---|---|---|---|---|---|---|
| SegFormer-B4 → SegFormer-B0 | Vanilla KD, stride 8 | Lunar | 42 | — | — | Not located |
| SegFormer-B4 → SegFormer-B0 | Vanilla KD, stride 4, weight 1, temperature 1 | Lunar | 42 | — | — | Not located |
| SegFormer-B4 → SegFormer-B0 | Vanilla KD, stride 4, weight 1, temperature 2 | Lunar | 42 | — | — | Not located |
| SegFormer-B4 → SegFormer-B0 | Vanilla KD, stride 4, weight 4, temperature 1 | Lunar | 42 | — | — | Not located |
| SegFormer-B4 → SegFormer-B0 | Vanilla KD, stride 4, weight 4, temperature 2 | Lunar | 42 | — | — | Not located |
| SegFormer-B4 → SegFormer-B0 | Vanilla KD + CAP | Lunar | 42 | — | — | Not located |
| SegFormer-B4 → SegFormer-B0 | Vanilla KD, matched no-CAP control | Lunar | 42 | — | — | Not located |
| Mask2Former Swin-L → Fast-SCNN | Vanilla KD | Lunar | 42 | — | — | Not located |
| Mask2Former Swin-L → Mask2Former Swin-L | Vanilla KD | Lunar | 42 | — | — | Not located |
| SegFormer-B0 | Supervised baseline | Lunar | 42 | — | — | Not located |
| SegFormer-B2 | Supervised baseline | Lunar | 42 | — | — | Not located |
| DINOv3 ViT-L/16 | Supervised baseline | Lunar | 42 | — | — | Not located |
| Mask2Former Swin-L | Supervised baseline | Lunar | 42 | — | — | Not located |

| Smoke check | Dataset | Val mIoU (%) | Test mIoU (%) | Status |
|---|---|---|---|---|
| Teacher → Fast-SCNN, job 9221665 | lunar | 0.38 | 0.47 | One-batch smoke; not a full-run result |
| SegFormer-B4, job 9221665 | lunar | 38.22 | 43.70 | One-batch smoke; not a full-run result |
| Teacher → Fast-SCNN, job 9435686 | ade20k | 0.00 | 0.00 | One-batch smoke; not a full-run result |
| SegFormer-B4, job 9435686 | ade20k | 0.00 | 1.25 | One-batch smoke; not a full-run result |
| Teacher → Fast-SCNN, job 9435686 | cityscapes | 0.00 | 0.00 | One-batch smoke; not a full-run result |
| SegFormer-B4, job 9435686 | cityscapes | 7.80 | 6.01 | One-batch smoke; not a full-run result |

| Configuration source | Location |
|---|---|
| Vanilla logit KD | [Configurations](../../experiments/logit_distillation/) |
| SARD and lunar controls | [Experiment guide](../../experiments/sard/README.md) |
| Cityscapes SARD | [Configurations](../../experiments/sard/cityscapes/) |
| ADE20K SARD | [Configurations](../../experiments/sard/ade20k/) |
| CAP and stride-4 logit sweep | `origin/feat/scalekd`, commit `be8d374`, under `experiments/cap/` and `experiments/logit_distillation/`; absent from the reviewed branch |

| Method / evidence notes | Details |
|---|---|
| Vanilla KD, SARD | Available on the original reviewed branch; additional ResNet-based runs were discovered on HPC |
| CAP | `origin/feat/scalekd`, commit `be8d374`; feature loss added to vanilla KD |
| Smoke scope | Six SARD smoke artifacts are listed; 21 additional CNN smoke artifacts were found but are excluded from full experiment comparisons |
| Local files | The local `results/` contains only `.gitkeep`; numeric values above were read from HPC artifacts |
