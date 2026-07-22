# Running the tests

The integration tests check the live PCSS Eagle cluster (connection, CPU, GPU).
They use the credentials in your `.env.test` (by default) and run their remote
commands **inside the service directory** named by `SERVICE_ID` (a folder in the
cluster home, e.g. `~/pl1234-01`).

## Prerequisites

- A configured `.env.test` (copy `.env.example` to `.env.test` and fill it in,
  including `SERVICE_ID`). If it is missing or incomplete, the tests **skip**
  rather than fail.
- The OpenSSH client (`ssh`) on your PATH.
- `pytest`:

```bash
pip install pytest
```

## Run

All tests (uses `.env.test`):

```bash
python -m pytest tests/integration -v
```

### Choosing the environment file

Pass `--env` to use a different dotenv file (relative to the repo root or an
absolute path). For example, to run against production:

```bash
python -m pytest tests/integration --env .env.prod -v
```

The chosen file must define `SSH_KEY`, `USERNAME`, `CLUSTER_ADDRESS`, and
`SERVICE_ID`. If the cluster is reachable but `SERVICE_ID` points to a folder
that does not exist, the tests **fail** (this is a real misconfiguration).

A single file:

```bash
python -m pytest tests/integration/test_gpu.py -v
```

A single test:

```bash
python -m pytest tests/integration/test_cpu.py::test_cpu_compute_correct_and_timely -v
```

## What each file tests

| File                 | Checks                                                    |
| -------------------- | --------------------------------------------------------- |
| `test_connection.py` | SSH login works and SLURM is reachable.                   |
| `test_cpu.py`        | Login-node CPU: cores present, remote compute is correct. |
| `test_gpu.py`        | H100 GPUs exist and are the only available GPU type.      |
