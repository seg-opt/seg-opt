# Canonical semantic-segmentation baselines

These five configs differ only in model initialization, physical
batch/accumulation, and model-specific learning rates. All use the checked-in
seed-42 manifest, full 512×768 frames, the same augmentations and metrics, and
best-checkpoint selection by validation mIoU.

| Config | Baseline |
|---|---|
| `fast_scnn.yaml` | Fast-SCNN from scratch |
| `resnet34_unet.yaml` | ImageNet ResNet34 with a skip-connected U-Net decoder |
| `segformer_b0.yaml` | ImageNet MiT-B0 with the official SegFormer decoder |
| `dinov3_vitl16.yaml` | Frozen DINOv3 ViT-L/16 with BN + 1×1 probe |
| `mask2former_swinl.yaml` | ADE-pretrained Mask2Former Swin-L, natively fine-tuned |

Smoke runs must use `--fast-dev-run --offline`. They retain and restore
`best.ckpt`, evaluate one validation/test batch, and write beneath
`results/smoke/`.

## Full canonical runs

Authenticate W&B once on the login node, then submit the canonical array:

```bash
uv run wandb login --relogin
mkdir -p results
sbatch experiments/baselines/train.sbatch
```

The array runs one baseline at a time (`0-4%1`) on one H100. W&B operates in
online mode and defaults to project `seg-opt-baselines` and run group
`canonical-seed42`. Set `WANDB_ENTITY`, `WANDB_PROJECT`, or `WANDB_RUN_GROUP`
in the submission environment to override those values.
