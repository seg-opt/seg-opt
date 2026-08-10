# PCSS Eagle - Compute Specification

## GPUs available

| GPU model       | SLURM `--gres` | Per node | Nodes | Total GPUs | Mem/GPU    | Partitions          | Status                       |
| --------------- | -------------- | -------- | ----- | ---------- | ---------- | ------------------- | ---------------------------- |
| **NVIDIA H100** | `gpu:h100:N`   | 4        | 76    | **304**    | ~94 GB     | `proxima`, `tesla`  | ✅ available (idle capacity) |
| **NVIDIA V100** | `gpu:tesla:N`  | 8        | 16    | 128        | (16/32 GB) | `accelrys`, `tesla` | ❌ DOWN / MAINTENANCE        |

## CUDA version

The GPU node driver caps the usable CUDA runtime:

| Item                       | Value                |
| -------------------------- | -------------------- |
| NVIDIA driver (H100 nodes) | **595.71.05**        |
| **Max CUDA supported**     | **13.2**             |
| `nvidia-smi` header        | `CUDA Version: 13.2` |

CUDA toolkit **modules** installed (`module avail cuda`):

| Module                   | CUDA | Bundled driver |
| ------------------------ | ---- | -------------- |
| `cuda/11.8.0-gcc-11.5.0` | 11.8 | —              |
| `cuda/12.4.1_550.54.15`  | 12.4 | 550.54.15      |
| `cuda/12.6.0_560.28.03`  | 12.6 | 560.28.03      |
| `cuda/12.8.0_570.86.10`  | 12.8 | 570.86.10      |
| `cuda/13.2.1_595.58.03`  | 13.2 | 595.58.03      |

## Recommended PyTorch

Because the driver supports up to CUDA 13.2, you can use PyTorch's newest
release. The highest `cp313` (Python 3.13) Linux wheels currently published by
PyTorch, per CUDA channel:

| Wheel channel (`--index-url .../whl/<ch>`) | Runtime CUDA | Highest `torch` | Runs on Eagle H100? |
| ------------------------------------------ | ------------ | --------------- | ------------------- |
| `cu132`                                    | 13.2         | **2.13.0**      | ✅ exact match      |
| `cu130`                                    | 13.0         | **2.13.0**      | ✅                  |
| `cu129`                                    | 12.9         | 2.13.0          | ✅                  |
| `cu128`                                    | 12.8         | 2.11.0          | ✅                  |
| `cu126`                                    | 12.6         | 2.13.0          | ✅                  |
| `cu124`                                    | 12.4         | (older)         | ✅ (repo default)   |

**Recommendation: `torch 2.13.0` on the `cu130` (or `cu132`) channel** — the
highest available PyTorch, on the CUDA channel that matches the 13.2 driver:

## Container runtime

PCSS documents Singularity (also known as Apptainer) for unprivileged container
execution. The initial project setup targets `proxima`, uses SIF images from
shared grant storage, and passes `--nv` for NVIDIA GPU jobs. Container jobs do
not load a host CUDA toolkit module by default: the host driver is exposed to
the container and the pinned image supplies CUDA 13.0.2 user-space libraries.

The exact runtime command and version can change independently of this
repository. Eagle exposes the runtime on allocated compute nodes rather than the
login node. SLURM jobs detect `singularity` first, fall back to `apptainer`, and
fail if neither is available.

See [the container runbook](containers.md) for the complete workflow.
