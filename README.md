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

The active experiment compares Fast-SCNN with ResNet34–U-Net, SegFormer-B0, a
frozen DINOv3 ViT-L/16 probe, and Mask2Former Swin-L on clean masks and the
aligned seed-42 split. Its configurations, smoke job, full array, and selection
protocol are in `experiments/baselines_benchmark/`. See
[`experiments/README.md`](experiments/README.md) for the complete experiment map.

Run a checkpoint-preserving local smoke test with:

```bash
uv run python -m scripts.train \
  --config experiments/baselines_benchmark/fast_scnn.yaml \
  --fast-dev-run --offline
```

Evaluate an existing checkpoint independently with:

```bash
uv run python -m scripts.evaluate \
  --config experiments/baselines_benchmark/fast_scnn.yaml \
  --checkpoint results/baselines_benchmark/fast_scnn/seed42/checkpoints/best.ckpt
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
  --config experiments/baselines_benchmark/fast_scnn.yaml \
  --devices 1 \
  --fast-dev-run
```

Run the active smoke job first, then the benchmark array:

```bash
sbatch --account=<SERVICE_ID> --export=ALL,IMAGE_PATH,DATA_PATH \
  experiments/baselines_benchmark/smoke.sbatch
sbatch --account=<SERVICE_ID> --export=ALL,IMAGE_PATH,DATA_PATH \
  experiments/baselines_benchmark/train.sbatch
```
