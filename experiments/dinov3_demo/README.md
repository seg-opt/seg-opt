# DINOv3 segmentation demo

This small qualitative experiment freezes DINOv3, caches features from a
deterministic subset, and trains only a linear segmentation head. It is useful
for quickly inspecting predictions, not as the main train/validation/test run.

Run locally:

```bash
uv run python -m experiments.dinov3_demo.demo
```

Submit on an H100:

```bash
sbatch experiments/dinov3_demo/demo.sbatch
```

Artifacts are written to `results/dinov3_demo/` by default.
