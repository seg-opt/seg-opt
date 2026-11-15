# Teacher, student, and distillation experiment inventory

Inventory reviewed on 15 November 2026 using the current
`feat/sard-benchmark-datasets` branch, other locally available Git branches,
experiment configurations, and copied SLURM logs. Configured experiments and
verified executions are listed separately: a configuration does not establish
that a run was submitted or completed.

## Teachers and students

| Model | Teacher role | Student role |
|---|---|---|
| SegFormer-B4 | Teacher for vanilla logit KD, SARD, and CAP | — |
| Mask2Former Swin-L | Teacher for vanilla KD and BPKD | Same-architecture distillation student |
| SegFormer-B0 | Baseline teacher candidate | Student of SegFormer-B4 |
| SegFormer-B2 | Baseline teacher candidate | — |
| ResNet34–U-Net | Baseline teacher candidate | — |
| DINOv3 ViT-L/16 | Baseline teacher candidate; frozen backbone with linear segmentation head | — |
| Fast-SCNN | — | Main lightweight student |

There are six teacher candidates and three student architectures. Only
SegFormer-B4 and Mask2Former Swin-L are referenced as teachers by the
distillation configurations. See the [baseline benchmark](../../experiments/baselines_benchmark/README.md).

## Distillation methods

| Method | Variants present |
|---|---|
| Vanilla pixel/logit KD | Temperature-scaled pixelwise KL divergence |
| BPKD | Body only; edge only; full body + edge, with kernel widths 3/5/7/9 |
| SARD — Structure-Aware Representation Distillation | Full structure weighting; uniform-weight control |
| CAP — Cross-Attention Projector feature distillation | ScaleKD-derived feature loss added to vanilla logit KD |

Vanilla KD, BPKD, and SARD are implemented on the reviewed current branch.
CAP is available on `origin/feat/scalekd` at commit `be8d374` and is an
additional feature loss rather than a separate `distillation.method` value.
Task-only/no-KD experiments are supervised controls.

## Configured teacher-to-student experiments

These are experiment definitions, not a list of confirmed completed runs.
Lunar means Artificial Lunar Landscape with clean masks.

| Teacher → student | Method / variants | Dataset | Seeds |
|---|---|---|---|
| SegFormer-B4 → Fast-SCNN | Vanilla logit KD | Lunar | 42 |
| SegFormer-B4 → Fast-SCNN | Uniform SARD | Lunar | 42 |
| SegFormer-B4 → Fast-SCNN | Full SARD | Lunar | 42, 43, 44 |
| SegFormer-B4 → Fast-SCNN | Full SARD | Cityscapes | 42, 43, 44 |
| SegFormer-B4 → Fast-SCNN | Full SARD | ADE20K | 42, 43, 44 |
| SegFormer-B4 → SegFormer-B0 | Vanilla logit KD, stride 8 | Lunar | 42 |
| SegFormer-B4 → SegFormer-B0 | Vanilla KD, stride 4; weight {1, 4} × temperature {1, 2} (four configurations) | Lunar | 42 |
| SegFormer-B4 → SegFormer-B0 | Vanilla KD + CAP; matched no-CAP control | Lunar | 42 |
| Mask2Former Swin-L → Fast-SCNN | Vanilla KD | Lunar | 42 |
| Mask2Former Swin-L → Fast-SCNN | BPKD body only | Lunar | 42 |
| Mask2Former Swin-L → Fast-SCNN | BPKD edge only | Lunar | 42 |
| Mask2Former Swin-L → Fast-SCNN | Full BPKD, kernels 3/5/7/9 (four configurations) | Lunar | 42 |
| Mask2Former Swin-L → Mask2Former Swin-L | Vanilla KD | Lunar | 42 |
| Mask2Former Swin-L → Mask2Former Swin-L | Full BPKD, kernel 7 | Lunar | 42 |

Configuration sources:

- [Vanilla logit KD](../../experiments/logit_distillation/)
- [SARD and lunar controls](../../experiments/sard/README.md)
- [Cityscapes SARD](../../experiments/sard/cityscapes/)
- [ADE20K SARD](../../experiments/sard/ade20k/)
- [BPKD suite and interpretation protocol](../../experiments/bpkd/README.md)
- CAP and the stride-4 logit sweep: `origin/feat/scalekd`, commit `be8d374`,
  under `experiments/cap/` and `experiments/logit_distillation/`.
  These files are absent from the reviewed current branch.

The current BPKD README marks its suite as deferred until the clean baseline
benchmark establishes a positive teacher margin.

## Supervised controls and teacher training

| Suite | Models | Dataset | Seed |
|---|---|---|---|
| Baseline benchmark | Fast-SCNN, ResNet34–U-Net, SegFormer-B0/B2/B4, DINOv3 ViT-L/16, Mask2Former Swin-L | Lunar | 42 |
| BPKD | Mask2Former Swin-L teacher/control; task-only Fast-SCNN | Lunar | 42 |
| SARD benchmark datasets | SegFormer-B4 teacher; no-KD Fast-SCNN | Cityscapes and ADE20K | 42 |

## Execution evidence available locally

| Run | Evidence and status |
|---|---|
| SegFormer-B4 supervised smoke training | Completed in job 9221665 on 20 October 2026; one training/validation/test batch |
| SegFormer-B4 → Fast-SCNN, full SARD smoke test | Completed in job 9221665 using the temporary smoke teacher; one training/validation/test batch |
| SegFormer-B4 supervised full training | Started in job 9221666; the copied output ends during epoch 3, so completion is unverified |
| Earlier lunar seed-42 work | `submit_seeds.sh` says “seed 42 already done” and submits only additional lunar seeds; no corresponding full-run results are stored locally |

Evidence files:

- [Smoke run log](../../slurm-sard-smoke-9221665.err): completion entries for
  the teacher and SARD student.
- [Full teacher training output](../../slurm-sard-teacher-segformer-b4-9221666.out)
- [Full teacher training log](../../sard-teacher-segformer-b4-train.log)
- [Additional-seed submission script](../../experiments/sard/submit_seeds.sh)

The copied logs may be ignored by Git and unavailable in another checkout.
At review time, `results/` contains only `.gitkeep`. No completed full
distillation run can be independently confirmed from the local artifacts.
A definitive completed-run inventory requires cluster result artifacts or
experiment-tracker history.
