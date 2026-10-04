# Singularity containers on PCSS Eagle

This repository builds separate diagnostic and development images. The small diagnostic image checks Singularity and H100 access. The DINOv3-derived development image supplies Python 3.13, CUDA 13.0.2, and dependencies locked in `uv.lock`, while source and data are mounted at runtime. GitHub Actions converts each OCI build to a checksummed SIF. Eagle jobs use SIFs from shared grant storage, so teammates do not need Docker or Apptainer build privileges.

| Image | Purpose | Source code | Rebuild trigger |
| --- | --- | --- | --- |
| Diagnostic | Validate Eagle, Python, and NVIDIA passthrough | Not included | Diagnostic Docker inputs |
| Development | Daily DINOv3 development and training | Mounted at `/workspace` | Python/dependency lock or development Docker inputs |
| Experiment | Reproducible archived run (planned) | Exact commit embedded | Explicit experiment tag or manual release |

## Prerequisites

1. Copy `.env.example` to `.env.prod` and set your own SSH key, Eagle username, cluster address, and shared grant in `SERVICE_ID`.
2. Confirm your grant can use `proxima` with `sacctmgr show assoc user=$USER format=account,partition,qos` on Eagle.
3. Keep `.env.*`, SSH keys, registry tokens, datasets, and generated SIF files out of Git. The repository ignore files already exclude them.

## Image versions

The container workflow runs only when its workflow, Docker ignore rules, diagnostic Dockerfile, locked requirements, or smoke test changes. Model, training, experiment, documentation, and SLURM changes do not rebuild it.

The separate development workflow runs when `.python-version`, `pyproject.toml`, `uv.lock`, `containers/development.Dockerfile`, its smoke test, or the workflow changes. Ordinary edits under `src/`, `scripts/`, `tests/`, and `experiments/` do not rebuild it because those paths are mounted at runtime.

Pull requests build and test the OCI and SIF images without publishing. Relevant changes on `main` publish `sha-<commit>` and `main` OCI tags. A `v*` tag also publishes the version and `latest` tags and attaches the SIF plus checksum to a GitHub release. The shared `current.sif` link is convenient, but recorded jobs should use an immutable version path.

Development releases use `dev-v*` tags, the separate `ghcr.io/seg-opt/seg-opt-development` package, and artifact names beginning with `seg-opt-development-`. They never replace diagnostic tags or `current.sif`.

## Development image contract

The package set is derived from the DINOv3 branch. The current contract is:

```text
Python       3.13
CUDA image   13.0.2
PyTorch      2.9.1+cu130
torchvision  0.24.1+cu130
```

The exact transitive graph comes from `uv.lock`. PyTorch and torchvision resolve from the official `https://download.pytorch.org/whl/cu130` index configured in `pyproject.toml`. Dependency updates must change project metadata and the lock file together.

## One-time Eagle setup

Each team member runs:

```bash
bash scripts/setup_container_storage.sh --env prod
```

This verifies `proxima` and creates:

```text
~/<SERVICE_ID>/project_data/containers/seg-opt/
├── images/                 # shared, immutable release directories
├── cache/<username>/       # per-user registry/build cache
├── tmp/<username>/         # per-user conversion temporary files
└── development/
  ├── images/             # shared development releases
  ├── cache/<username>/   # framework/model cache
  └── tmp/<username>/     # development conversion/job temporary files
```

It does not replace `~/.singularity`. When pulling an image manually, use the `SINGULARITY_CACHEDIR` and `SINGULARITY_TMPDIR` exports printed by the script so the approximately 1 GB home quota is not consumed.

## Release and deployment

A release manager creates a version tag and downloads its SIF artifacts:

```bash
git tag -a v0.1.0 -m "Container v0.1.0"
git push origin v0.1.0
gh release download v0.1.0 --pattern "*.sif*" --dir dist/v0.1.0
sha256sum --check dist/v0.1.0/seg-opt-v0.1.0.sif.sha256
```

After the container workflow succeeds, deploy it:

```bash
bash scripts/deploy_container.sh \
  --file dist/v0.1.0/seg-opt-v0.1.0.sif \
  --version v0.1.0 \
  --env prod
```

Deployment verifies hashes locally and remotely, refuses to replace a different image under an existing version, and atomically updates `current.sif`. Repeating the same deployment is safe. Singularity is available on Eagle compute nodes rather than the login node, so the SLURM smoke jobs perform runtime validation.

### Development deployment

Eagle compute nodes cannot currently reach public OCI registries, including Docker Hub and GHCR. Download the checksummed SIF artifact from the successful development-container workflow on a machine with GitHub access, then deploy it over SSH. The upload is resumable and verifies the checksum locally and on Eagle.

```bash
env -u GITHUB_TOKEN gh run download <RUN_ID> \
  --name seg-opt-development-dep-sha-<commit> \
  --dir dist/development-<commit>
sha256sum --check dist/development-<commit>/*.sif.sha256
```

Deploy the development SIF into its independent namespace:

```bash
bash scripts/deploy_container.sh \
  --file dist/development-<commit>/seg-opt-development-dep-sha-<commit>.sif \
  --version dep-sha-<commit> \
  --kind development \
  --env prod
```

This updates `containers/seg-opt/development/current.sif`, not the diagnostic pointer.

To roll back, verify the target and replace the link on Eagle:

```bash
cd ~/<SERVICE_ID>/project_data/containers/seg-opt/images/v0.0.9
sha256sum --check seg-opt-v0.0.9.sif.sha256
cd ../..
ln -sfn images/v0.0.9/seg-opt-v0.0.9.sif current.sif.next
mv -Tf current.sif.next current.sif
```

## Submit jobs

Use an immutable image for recorded experiments:

```bash
export IMAGE_PATH="$HOME/<SERVICE_ID>/project_data/containers/seg-opt/images/v0.1.0/seg-opt-v0.1.0.sif"
sbatch --account=<SERVICE_ID> --export=ALL,IMAGE_PATH jobs/container_cpu.sbatch
sbatch --account=<SERVICE_ID> --export=ALL,IMAGE_PATH jobs/container_gpu.sbatch
```

With no extra arguments, the templates run the diagnostic probe. To execute a project command, append it after the script name:

```bash
sbatch --account=<SERVICE_ID> --export=ALL,IMAGE_PATH,DATA_PATH \
  jobs/container_gpu.sbatch python3 /workspace/experiments/train.py --data /data
```

Set `REPO_PATH` to the checked-out source and `DATA_PATH` to grant-backed data. They are mounted as `/workspace` and `/data`. The CPU template uses `proxima-cpu`. The GPU template requests one H100 on `proxima` and uses `--nv`; do not load a host CUDA module for this image. The host NVIDIA driver is injected while user-space CUDA comes from the container.

The initial templates are single-node only. Multi-node MPI requires PCSS's separate `srun --mpi pmix` and binding setup and should be added with the actual distributed workload.

## Development training jobs

The development SIF contains dependencies but not source, datasets, credentials, caches, or outputs. `scripts/train.sbatch` uses these mounts:

| Host variable | Container path | Access |
| --- | --- | --- |
| `REPO_PATH` (submission directory by default) | `/workspace` | read-only |
| `DATA_PATH` | `/data` and `/datasets` | read-only |
| `CACHE_PATH` | `/cache` | read-write |
| `OUTPUT_PATH` | `/output` and `/workspace/results` | read-write |

The `/datasets` and `/workspace/results` aliases preserve paths currently used by the baseline configuration. New configurations should prefer `/data` and `/output`.

Submit a one-batch H100 check:

```bash
export IMAGE_PATH="$HOME/<SERVICE_ID>/project_data/containers/seg-opt/development/images/<VERSION>/seg-opt-development-<VERSION>.sif"
export DATA_PATH="$HOME/<SERVICE_ID>/project_data/datasets"
export REPO_PATH="$PWD"

sbatch --account=<SERVICE_ID> \
  --export=ALL,IMAGE_PATH,DATA_PATH,REPO_PATH \
  scripts/train.sbatch \
  python -m scripts.train \
  --config experiments/baselines_benchmark/fast_scnn.yaml \
  --devices 1 \
  --fast-dev-run
```

Omit the command after `scripts/train.sbatch` for the default baseline run. To collect a short Nsight Systems trace, export `NSYS_PROFILE=1`; the host-side `nsys` executable must be available on the allocated compute node. The launcher writes a `.nsys-rep` file to `results/nsys/` by default, after a 60-second delay and for 45 seconds. Override those timings with `NSYS_DELAY_SECONDS` and `NSYS_DURATION_SECONDS`. A 300-step profiling job can be submitted with:

```bash
export NSYS_PROFILE=1 NSYS_DELAY_SECONDS=20 NSYS_DURATION_SECONDS=45
sbatch --account=<SERVICE_ID> --export=ALL,IMAGE_PATH,DATA_PATH,NSYS_PROFILE,NSYS_DELAY_SECONDS,NSYS_DURATION_SECONDS \
  scripts/train.sbatch \
  python -m scripts.train \
  --config experiments/baselines_benchmark/fast_scnn.yaml \
  --devices 1 \
  --max-steps 300 \
  --offline
```

`--max-steps` makes this a bounded diagnostic run, not a comparable training result. Nsight Systems runs automatically add coarse NVTX ranges for fit, epoch, and train/validation/test batches; these make GPU gaps visible without instrumenting model code. Record both the development SIF checksum and mounted source commit for every result; they are independent provenance inputs.

For an Nsight Compute roofline report of one post-warmup CUDA kernel, use `NCU_PROFILE=1` instead. It writes a `.ncu-rep` file to `results/ncu/`; open this file in the separate Nsight Compute desktop application (not Nsight Systems). `NCU_LAUNCH_SKIP_BEFORE_MATCH` and `NCU_LAUNCH_COUNT` control the small number of kernels collected.

## Planned immutable experiment images

After the training pipeline passes a real H100 smoke run, experiment images will start from a pinned development-image digest and embed an exact clean source commit, resolved configuration, lockfile, and provenance manifest. Datasets, credentials, caches, and checkpoints remain external. Experiment builds will be explicit rather than triggered by every source commit.

## Private GHCR access

SIF deployment from a GitHub release does not require Eagle to access GHCR. For a manual registry pull from a private package, create a GitHub token with read-only package access and enter it interactively when prompted:

```bash
singularity registry login --username <github-user> docker://ghcr.io
srun --account=<SERVICE_ID> --partition=proxima --mem=32G --time=00:30:00 \
  singularity pull seg-opt.sif docker://ghcr.io/seg-opt/seg-opt:sha-<commit>
```

Never put the token in `.env.*`, a SLURM script, shell history, or Git. If the GHCR package becomes public, registry login is unnecessary.

## Validation

Credential-free checks:

```bash
docker build -f containers/Dockerfile -t seg-opt:local .
docker run --rm seg-opt:local
bash -n scripts/*.sh jobs/*.sbatch
```

Cluster checks are documented in [test.md](test.md). Keep the H100 test opt-in because it queues and charges a real GPU allocation.

## Troubleshooting

- **Home quota exhausted:** ensure caches and temporary files use the paths printed by `setup_container_storage.sh`; remove stale files from `~/.singularity`.
- **Pull or conversion is killed:** request 16-32 GB on a `proxima` worker for CUDA images and use grant-backed temporary storage.
- **GPU missing in a job:** request an H100 and execute with `--nv`. The GPU template does both and fails if the probe cannot see NVIDIA devices.
- **Container option rejected on another partition:** use `proxima`, the partition currently documented by PCSS for containers.
- **Release asset too large:** use the immutable GHCR `sha-<commit>` tag and convert/pull it on a `proxima` worker, then retain the generated checksum.
