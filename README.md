# seg-opt

Semantic-segmentation and knowledge-distillation experiments for lunar imagery. The supported development container uses Python 3.13, CUDA 13.0.2, and the dependencies locked in `uv.lock`.

## Local setup

```bash
uv sync --frozen
uv run pytest tests/unit
```

## Containers on Eagle

The diagnostic SIF validates Singularity and H100 access. The development SIF supplies the locked Python/CUDA environment and mounts the current checkout. Immutable experiment SIFs will be added after the training pipeline is stable. See `docs/containers.md` for image lifecycle and `docs/test.md` for test layers.

## Tests

```bash
uv run pytest
```

PCSS integration tests are skipped unless `.env.test` is configured.

## Training

Run the configured experiment with:

```bash
uv run python -m scripts.train \
	--config experiments/baseline_resnet34/config.yaml
```

Add `--profile` to collect CUDA allocator/utilization metrics and a PyTorch operator trace. Metrics are sent to the `seg-opt` Weights & Biases project. Authenticate once before submitting jobs:

```bash
uv run wandb login
```

Profiler summaries and Chrome traces are written under the experiment's `profiler/` directory. Open a trace in `chrome://tracing` or Perfetto. Profiling adds overhead, so use it for diagnosis rather than every full training run.

Use `--fast-dev-run` to check the complete pipeline on one batch before a long job.

On Eagle, use the development SIF rather than a host virtual environment:

```bash
export IMAGE_PATH="$HOME/<SERVICE_ID>/project_data/containers/seg-opt/development/images/<VERSION>/seg-opt-development-<VERSION>.sif"
export DATA_PATH="$HOME/<SERVICE_ID>/project_data/datasets"
sbatch --account=<SERVICE_ID> --export=ALL,IMAGE_PATH,DATA_PATH \
	scripts/train.sbatch \
	python -m scripts.train \
	--config experiments/baseline_resnet34/config.yaml \
	--devices 1 \
	--fast-dev-run
```
