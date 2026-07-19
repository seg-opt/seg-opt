# Running the tests

The integration tests check the live PCSS Eagle cluster (connection, CPU, GPU).
They use the credentials in your `.env`.

## Prerequisites

- A configured `.env` (see [`.env.example`](../.env.example)).
- The OpenSSH client (`ssh`) on your PATH.
- `pytest`:

```bash
pip install pytest
```

## Run

All tests:

```bash
python -m pytest tests/integration -v
```

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
