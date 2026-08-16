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

## Canonical baselines

The reviewed baseline set is Fast-SCNN, ResNet34–U-Net, SegFormer-B0, a frozen
DINOv3 ViT-L/16 linear probe, and Mask2Former Swin-L. Their configurations and
fixed seed-42 split are in `experiments/baselines/`.

Run a checkpoint-preserving local smoke test with:

```bash
uv run python -m scripts.train \
  --config experiments/baselines/fast_scnn.yaml \
  --fast-dev-run --offline
```

Evaluate an existing checkpoint independently with:

```bash
uv run python -m scripts.evaluate \
  --config experiments/baselines/fast_scnn.yaml \
  --checkpoint results/baselines/fast_scnn/seed42/checkpoints/best.ckpt
```

Training writes the resolved config, split manifest, best checkpoint, log, and
metrics JSON. W&B is optional; local artifacts are authoritative.

## Containers on Eagle

Eagle jobs use the development SIF instead of a host virtual environment. See
`docs/containers.md` for the image lifecycle and `docs/test.md` for test layers.

```bash
export IMAGE_PATH="$HOME/<SERVICE_ID>/project_data/containers/seg-opt/development/images/<VERSION>/seg-opt-development-<VERSION>.sif"
export DATA_PATH="$HOME/<SERVICE_ID>/project_data/datasets"

sbatch --account=<SERVICE_ID> --export=ALL,IMAGE_PATH,DATA_PATH \
  scripts/train.sbatch \
  python -m scripts.train \
  --config experiments/baselines/fast_scnn.yaml \
  --devices 1 \
  --fast-dev-run
```

The canonical sequential array and smoke jobs are available under
`experiments/baselines/`; both delegate container execution to
`scripts/train.sbatch`.
