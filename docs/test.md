# Running the tests

The suite is split into isolated unit tests and integration tests that validate the configured lunar dataset and the live PCSS Eagle cluster.

## Unit tests

Unit tests do not require the dataset or cluster credentials:

```bash
uv run python -m pytest tests/unit -v
```

## Integration tests

The cluster tests use the credentials in your `.env.test` (by default) and run their remote commands **inside the service directory** named by `SERVICE_ID` (a folder in the cluster home, e.g. `~/pl1234-01`). The data pipeline integration test uses the dataset configured in `experiments/baseline_resnet34/config.yaml`.

### Prerequisites

- The lunar dataset at the path configured in `experiments/baseline_resnet34/config.yaml` for the data pipeline test.
- A configured `.env.test` (copy `.env.example` to `.env.test` and fill it in, including `SERVICE_ID`). If it is missing or incomplete, the tests **skip** rather than fail.
- The OpenSSH client (`ssh`) on your PATH.

### Run

uv sync --frozen

All tests (uses `.env.test`):

```bash
uv run python -m pytest tests/integration -v
```

#### Choosing the environment file

Pass `--env` to use a different dotenv file (relative to the repo root or an absolute path). For example, to run against production:

```bash
uv run python -m pytest tests/integration --env .env.prod -v
```

The chosen file must define `SSH_KEY`, `USERNAME`, `CLUSTER_ADDRESS`, and `SERVICE_ID`. If the cluster is reachable but `SERVICE_ID` points to a folder that does not exist, the tests **fail** (this is a real misconfiguration).

A single file:

```bash
uv run python -m pytest tests/integration/test_gpu.py -v
```

A single test:

```bash
uv run python -m pytest tests/integration/test_cpu.py::test_cpu_compute_correct_and_timely -v
```

### What each file tests

| File | Checks |
| --- | --- |
| `test_connection.py` | SSH login works and SLURM is reachable. |
| `test_cpu.py` | Login-node CPU: cores present, remote compute is correct. |
| `test_data_pipeline.py` | Configured dataset layout, pairing, and decoding. |
| `test_gpu.py` | H100 GPUs exist and are the only available GPU type. |

## Development container checks

Build and run the CPU-side OCI smoke test after changing the Python/dependency contract or development Dockerfile:

```bash
docker build \
  --file containers/development.Dockerfile \
  --tag seg-opt-development:local \
  .
docker run --rm seg-opt-development:local
```

The output must report Python 3.13, container CUDA 13.0.2, PyTorch with its CUDA 13.0 build, matching torchvision, and a non-root UID. A local machine without an NVIDIA runtime is expected to report `gpu.available: false`.

The development GitHub Actions workflow converts this same OCI image to SIF, runs the CPU-side smoke test through Apptainer, and verifies SHA-256. Downloaded SIFs must be checked before deployment:

```bash
sha256sum --check seg-opt-development-<VERSION>.sif.sha256
```

## H100 workload check

An H100 check consumes a real allocation and is run only after OCI/SIF checks pass. Mount the current source and a small dataset fixture, then submit one batch:

```bash
sbatch --account=<SERVICE_ID> \
  --export=ALL,IMAGE_PATH,DATA_PATH,REPO_PATH \
  scripts/train.sbatch \
  python -m scripts.train \
  --config experiments/baseline_resnet34/config.yaml \
  --devices 1 \
  --fast-dev-run
```

Record the SIF checksum, dependency lock checksum, source commit, dataset version, command, SLURM job ID, GPU/driver, and output path.
