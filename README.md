# seg-opt

Semantic-segmentation and knowledge-distillation experiments for lunar imagery.
The supported development container uses Python 3.13, CUDA 13.0.2, and the
dependencies locked in `uv.lock`.

## Local setup and tests

```bash
uv sync --frozen
uv run pytest
```

PCSS integration tests are skipped unless `.env.test` is configured.

## Active experiment: clean baseline benchmark

The active experiment compares Fast-SCNN with ResNet34–U-Net, SegFormer
B0/B2/B4, a frozen DINOv3 ViT-L/16 probe, and Mask2Former Swin-L on clean masks
and the aligned seed-42 split. Its presets, submit commands, and
selection protocol are in `experiments/baselines_benchmark/`. See
[`experiments/README.md`](experiments/README.md) for the complete experiment map.

Run a checkpoint-preserving local smoke test with:

```bash
config_dir=$(mktemp -d)
uv run --only-group submit python -m scripts.compose_experiment \
  dataset=lunar method=supervised teacher=none student=fast_scnn training.batch_size=16 \
    training.accumulate_grad_batches=1 training.learning_rate=0.0005 \
    training.backbone_learning_rate=null --write-config "$config_dir/config.yaml"
uv run python -m scripts.train --config "$config_dir/config.yaml" --fast-dev-run --offline
```

Evaluate an existing checkpoint independently with:

```bash
uv run python -m scripts.evaluate \
  --config "$config_dir/config.yaml" \
  --checkpoint results/baselines_benchmark/fast_scnn/seed42/checkpoints/best.ckpt
```

Training writes the resolved config, split manifest, best checkpoint, log, and
metrics JSON. W&B is optional; local artifacts are authoritative.

## Containers on Eagle

Eagle jobs use the development SIF instead of a host virtual environment. See
[container guide](docs/development/containers.md) for the image lifecycle and [testing guide](docs/development/testing.md) for test layers.

```bash
export IMAGE_PATH="$HOME/<SERVICE_ID>/project_data/containers/seg-opt/development/images/<VERSION>/seg-opt-development-<VERSION>.sif"
export DATA_PATH="$HOME/<SERVICE_ID>/project_data/datasets"

scripts/submit --smoke-only dataset=lunar method=supervised teacher=none student=fast_scnn \
  training.batch_size=16 training.accumulate_grad_batches=1 training.learning_rate=0.0005 \
  training.backbone_learning_rate=null
scripts/submit --after-job "$SMOKE_JOB_ID" --full-only dataset=lunar method=supervised teacher=none \
  student=fast_scnn training.batch_size=16 training.accumulate_grad_batches=1 \
  training.learning_rate=0.0005 training.backbone_learning_rate=null
```

See the [benchmark guide](experiments/baselines_benchmark/README.md) for the
seven model presets and the [experiment map](experiments/README.md) for launcher options.

## Documentation

See the [experiment inventory](docs/experiments/inventory.md),
[development guides](docs/development/), and [cluster guides](docs/cluster/).
