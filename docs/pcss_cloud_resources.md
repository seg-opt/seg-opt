# Using Grants & Resources on PCSS Eagle

After logging in you will see one directory per **grant** (resource allocation)
you belong to, e.g.:

```
README.md  pl1200-01  pl1201-02  seg-opt-gpu
```

- `pl1200-01` — scientific workspace grant (shared with the project's members).
- `pl1201-02` — personal student workspace grant.
- `seg-opt-gpu` — a plain working directory (code), **not** part of any grant:
  outside the grant quotas and **not** backed up.

A grant is two things at once: a **storage area** and a **SLURM account** you
charge compute to.

## Storage layout

Each grant gets its own directory tree in your home:

```
~/pl1200-01/
├── project_data/   # shared by all grant members; BACKED UP; kept 6 months after the grant ends
├── scratch/        # fast compute/staging space; NOT backed up
└── archive/        # slow, very large (10s–100s TB) cold storage; NOT backed up
```

Rules that matter in practice:

- **Home (`~`) is only ~1 GB** — it is just for organizing grant directories,
  not for datasets. Real data goes under a grant.
- `project_data` and `scratch` are **symlinks** to the real filesystems. Use the
  relative path (`~/pl1200-01/project_data`), not the physical mount, because the
  underlying storage can change.
- **Only `project_data` is backed up.** Back up anything important that lives in
  `scratch` or `archive` yourself.

## Running jobs against a grant

The grant name is also your **SLURM account**; charge compute to it with
`-A` / `--account`. If you omit the account the job may fail or bill the wrong
allocation.

Interactive session (CPU):

```bash
srun -A pl1200-01 -p standard --time=00:10:00 --pty bash
```

Interactive GPU session (Eagle is H100-only for now):

```bash
srun -A pl1200-01 -p tesla --gpus-per-node=1 --constraint=h100 --time=00:30:00 --pty bash
```

Batch script (`job.sh`), submitted with `sbatch -A pl1200-01 job.sh`:

```bash
#!/usr/bin/env bash
#SBATCH --account=pl1200-01          # your grant; swap pl1201-02 for personal work
#SBATCH --partition=tesla            # 'standard' for CPU-only
#SBATCH --gpus-per-node=1
#SBATCH --constraint=h100
#SBATCH --time=01:00:00
#SBATCH --job-name=seg-opt

# PCSS's recommended per-job temp dir, keyed to the charged grant:
export TMPDIR=$HOME/grant_$SLURM_JOB_ACCOUNT/scratch/$USER/$SLURM_JOB_ID
mkdir -p "$TMPDIR"

srun python ~/seg-opt-gpu/train.py --data ~/pl1200-01/project_data/...
cp -r "$TMPDIR"/results ~/pl1200-01/project_data/results/   # save to backed-up storage
```

Monitoring:

```bash
squeue -u $USER          # your queued/running jobs
scancel <job_id>         # cancel a job
sinfo -p tesla           # partition/node availability
```

## Container jobs

The shared Singularity setup uses the `proxima` partition, which is the
partition currently documented by PCSS for container execution. This differs
from the direct host examples above, which use `standard` or `tesla`.

Versioned SIF images live under:

```text
~/<grant-id>/project_data/containers/seg-opt/images/<version>/
```

Do not keep container images or caches directly in the approximately 1 GB home
directory. Follow [the container runbook](containers.md) for one-time storage
setup, deployment, CPU/GPU templates, and rollback.

## Which grant to use

- **`pl1200-01` (scientific)** — use for the shared project so collaborators
  share `project_data`.
- **`pl1201-02` (student)** — use for personal experiments.

Switch grants by changing `--account` (and the `project_data`/`scratch` paths).

## Check your own entitlements

Partition and QOS names below come from PCSS documentation and may differ
slightly on your instance. These commands give the authoritative answer for
**your** accounts, run on the cluster:

```bash
sacctmgr show assoc user=$USER format=account,partition,qos   # accounts/partitions you may use
sshare -A pl1200-01                                           # usage/share for a grant
sinfo                                                          # partitions and node states
```

> Sources for this page are listed in [`docs/resources.md`](resources.md).
