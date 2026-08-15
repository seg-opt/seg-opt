# ResNet-34 supervised baseline

This experiment verifies the supervised training and logging path without teacher distillation. Its configuration is `experiments/baseline_resnet34/config.yaml`.

Run one batch locally after `uv sync --frozen`:

```bash
uv run python -m scripts.train \
	--config experiments/baseline_resnet34/config.yaml \
	--fast-dev-run
```

On Eagle, set an immutable development SIF and grant-backed dataset root:

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

Remove `--fast-dev-run` only after the one-batch job passes. Results are written to the mounted `results/baseline_resnet34` directory. Retain the SIF checksum, source commit, `uv.lock` checksum, dataset version, command, SLURM job ID, GPU and driver, resolved config, and output path with every recorded run.
