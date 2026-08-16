# DINOv3 segmentation probe v2

This experiment strengthens the small real-data probe without introducing
teacher/student distillation:

- 512-pixel, class-aware training crops with scale, flip, and photometric augmentation;
- aspect-ratio-preserving evaluation with overlapping sliding windows;
- loss against full-resolution labels using weighted cross-entropy plus soft Dice;
- an official-style BatchNorm linear head, with a lightweight convolutional head available;
- optional fine-tuning of only the final DINOv3 transformer blocks.

With W&B enabled, each run records the complete experiment configuration,
step/epoch total loss, CE and Dice loss components, learning rates, and these
epoch metrics for train/validation/test: mIoU, macro F1, pixel accuracy, plus
per-class IoU, precision, and recall.

Run one batch through the complete CPU/GPU pipeline without W&B:

```bash
uv run python -m experiments.dinov3_probe_v2.train \
  --config experiments/dinov3_probe_v2/config.yaml \
  --fast-dev-run \
  --offline
```

Submit the bounded 512/128/128-frame experiment on an H100:

```bash
sbatch experiments/dinov3_probe_v2/train.sbatch
```

Keep the linear frozen probe as the first reference. Then change one setting at
a time: `head_type: lightweight`, followed by `unfreeze_last_blocks: 2`.
