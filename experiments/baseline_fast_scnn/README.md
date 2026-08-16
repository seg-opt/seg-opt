# Fast-SCNN supervised baseline

This first Fast-SCNN run is an equal-recipe baseline: it trains from scratch
with the same data split, square resolution, cross-entropy loss, optimizer
settings, and 10-epoch budget as the existing ResNet-34 experiment. It is not
a reproduction of the paper's much longer SGD/poly training recipe.

Run a one-batch pipeline check without W&B:

```bash
uv run python -m scripts.train \
  --config experiments/baseline_fast_scnn/config.yaml \
  --fast-dev-run \
  --offline
```

Submit the full run on an H100:

```bash
sbatch experiments/baseline_fast_scnn/train.sbatch
```

The submitted job extracts the tested source snapshot from
`results/source_snapshots/fast_scnn_equal_recipe_v1_retry1.tar.gz`, so later edits to
the shared working tree cannot silently change a queued run.

Python multiprocessing uses a short job-specific directory under `/tmp`, while
source and data remain staged on grant scratch. This keeps DataLoader IPC paths
within the Linux Unix-socket limit.

The batch job defaults to W&B offline mode because cluster credentials are not
currently configured. After logging in, upload the local history with
`uv run wandb sync results/baseline_fast_scnn/equal_recipe_v1_retry1/wandb/wandb/offline-run-*`.
