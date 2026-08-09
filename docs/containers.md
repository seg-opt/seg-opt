# Singularity containers on PCSS Eagle

This repository builds a small diagnostic CUDA/Python image for the whole team.
GitHub Actions publishes it to GHCR and converts the same build to a checksummed
SIF. Eagle jobs use SIFs from shared grant storage, so team members do not need
Docker, Apptainer build privileges, or another person's credentials.

## Prerequisites

1. Copy `.env.example` to `.env.prod` and set your own SSH key, Eagle username,
   cluster address, and shared grant in `SERVICE_ID`.
2. Confirm your grant can use `proxima` with
   `sacctmgr show assoc user=$USER format=account,partition,qos` on Eagle.
3. Keep `.env.*`, SSH keys, registry tokens, datasets, and generated SIF files
   out of Git. The repository ignore files already exclude them.

## Image versions

The container workflow runs only when its workflow, Docker ignore rules,
diagnostic Dockerfile, locked requirements, or smoke test changes. Model,
training, experiment, documentation, and SLURM changes do not rebuild it.

Pull requests build and test the OCI and SIF images without publishing. Relevant
changes on `main` publish `sha-<commit>` and `main` OCI tags. A `v*` tag also
publishes the version and `latest` tags and attaches the SIF plus checksum to a
GitHub release. The shared `current.sif` link is convenient, but recorded jobs
should use an immutable version path.

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
└── tmp/<username>/         # per-user conversion temporary files
```

It does not replace `~/.singularity`. When pulling an image manually, use the
`SINGULARITY_CACHEDIR` and `SINGULARITY_TMPDIR` exports printed by the script so
the approximately 1 GB home quota is not consumed.

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

Deployment verifies hashes locally and remotely, refuses to replace a different
image under an existing version, and atomically updates `current.sif`.
Repeating the same deployment is safe. Singularity is available on Eagle compute
nodes rather than the login node, so the SLURM smoke jobs perform runtime
validation.

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

With no extra arguments, the templates run the diagnostic probe. To execute a
project command, append it after the script name:

```bash
sbatch --account=<SERVICE_ID> --export=ALL,IMAGE_PATH,DATA_PATH \
  jobs/container_gpu.sbatch python3 /workspace/experiments/train.py --data /data
```

Set `REPO_PATH` to the checked-out source and `DATA_PATH` to grant-backed data.
They are mounted as `/workspace` and `/data`. The GPU template requests one H100
and uses `--nv`; do not load a host CUDA module for this image. The host NVIDIA
driver is injected while user-space CUDA comes from the container.

The initial templates are single-node only. Multi-node MPI requires PCSS's
separate `srun --mpi pmix` and binding setup and should be added with the actual
distributed workload.

## Private GHCR access

SIF deployment from a GitHub release does not require Eagle to access GHCR.
For a manual registry pull from a private package, create a GitHub token with
read-only package access and enter it interactively when prompted:

```bash
singularity registry login --username <github-user> docker://ghcr.io
srun --account=<SERVICE_ID> --partition=proxima --mem=32G --time=00:30:00 \
  singularity pull seg-opt.sif docker://ghcr.io/seg-opt/seg-opt:sha-<commit>
```

Never put the token in `.env.*`, a SLURM script, shell history, or Git. If the
GHCR package becomes public, registry login is unnecessary.

## Validation

Credential-free checks:

```bash
docker build -f containers/Dockerfile -t seg-opt:local .
docker run --rm seg-opt:local
bash -n scripts/*.sh jobs/*.sbatch
```

Cluster checks are documented in [test.md](test.md). Keep the H100 test opt-in
because it queues and charges a real GPU allocation.

## Troubleshooting

- **Home quota exhausted:** ensure caches and temporary files use the paths
  printed by `setup_container_storage.sh`; remove stale files from `~/.singularity`.
- **Pull or conversion is killed:** request 16-32 GB on a `proxima` worker for
  CUDA images and use grant-backed temporary storage.
- **GPU missing in a job:** request an H100 and execute with `--nv`. The GPU
  template does both and fails if the probe cannot see NVIDIA devices.
- **Container option rejected on another partition:** use `proxima`, the
  partition currently documented by PCSS for containers.
- **Release asset too large:** use the immutable GHCR `sha-<commit>` tag and
  convert/pull it on a `proxima` worker, then retain the generated checksum.