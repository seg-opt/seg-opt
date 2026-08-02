# seg-opt

## Tests

```bash
uv run pytest
```

PCSS integration tests are skipped unless `.env.test` is configured.

## Training

Run the configured experiment with:

```bash
uv run python scripts/train.py --config configs/distill_mask2former_resnet_v1.yaml
```

Add `--profile` to collect CUDA allocator/utilization metrics and a PyTorch
operator trace. Metrics are sent to the `seg-opt` Weights & Biases project.
Authenticate once before submitting jobs:

```bash
uv run wandb login
```

Profiler summaries and Chrome traces are written under the experiment's
`profiler/` directory. Open a trace in `chrome://tracing` or Perfetto. Profiling
adds overhead, so use it for diagnosis rather than every full training run.

Use `--fast-dev-run` to check the complete pipeline on one batch before a long
job.
