# Uploading files to PCSS Eagle

Eagle has no special data-transfer service for our use case: files go over SSH to
the login node, with `scp` or `rsync`. This page covers the helper script and the
raw commands behind it.

## Quick start

```bash
bash scripts/upload_file_to_cluster.sh --file data/artificial_lunar_landscape.zip
```

This uploads the given file to `~/pl1200-01/project_data/` on the cluster.

## Using the helper script

[`scripts/upload_file_to_cluster.sh`](../scripts/upload_file_to_cluster.sh):

```bash
bash scripts/upload_file_to_cluster.sh --file <path> [--env prod|test] [--out-path <subdir>]
```

| Parameter    | Meaning                                                                                                  |
| ------------ | -------------------------------------------------------------------------------------------------------- |
| `--file`     | **Required.** Path to the local file to upload — any file type, not just archives.                        |
| `--env`      | Which env file supplies the credentials and `SERVICE_ID`: `prod` → `.env.prod`, `test` → `.env.test`. Default `prod`. Anything else is an error. |
| `--out-path` | Subfolder inside that grant's `project_data`. Optional; always relative.                                  |
| `-h, --help` | Print usage and exit.                                                                                     |

There are no other parameters and no shorthand aliases for `--file`, `--env` or
`--out-path`. One environment variable also affects a run: `PROGRESS_INTERVAL`
(see [Progress reporting](#progress-reporting)).

### How the destination is built

```
<SERVICE_ID>/project_data/<out-path>
```

`SERVICE_ID` comes from the selected env file, so `--env` picks the grant and
`--out-path` picks the folder within it:

| `--env` | `SERVICE_ID` | `--out-path` | Destination                            |
| ------- | ------------ | ------------ | -------------------------------------- |
| `prod`  | `pl1200-01`  | *(omitted)*  | `~/pl1200-01/project_data/`            |
| `prod`  | `pl1200-01`  | `datasets`   | `~/pl1200-01/project_data/datasets/`   |
| `test`  | `pl1201-02`  | `runs/2026`  | `~/pl1201-02/project_data/runs/2026/`  |

The whole path is created with `mkdir -p` if it does not exist. `--out-path` is
rejected if it is absolute or tries to climb out with `..`, so an upload can
never land outside `project_data`; a trailing slash is harmless.

There is no default file: omitting `--file` is an error, so nothing is ever
uploaded by accident.

Examples:

```bash
bash scripts/upload_file_to_cluster.sh --file data/foo.zip                             # -> ~/pl1200-01/project_data
bash scripts/upload_file_to_cluster.sh --file data/foo.zip --out-path datasets
bash scripts/upload_file_to_cluster.sh --file checkpoint.pt --out-path models/unet
bash scripts/upload_file_to_cluster.sh --file data/foo.zip --env test --out-path scratch_copy
bash scripts/upload_file_to_cluster.sh --help
```

Only single regular files are supported: a directory is rejected with a clear
error rather than silently skipped by `scp`. For a whole tree, use the manual
`scp -r` / `rsync -a` commands below.

The script reuses the credential handling of
[`scripts/connect_to_cluster.sh`](../scripts/connect_to_cluster.sh): it reads the
env file (stripping CR so Windows-edited files work), writes `SSH_KEY` to a
temporary `0600` file, and deletes that file on exit. SSH runs with
`BatchMode=yes` and `StrictHostKeyChecking=accept-new`, so it never blocks on a
prompt.

## Where to put the data

`~` is only ~1 GB, so datasets always go under a grant. See
[`pcss_cloud_resources.md`](pcss_cloud_resources.md) for the full layout.

| Destination                     | Use it for                                       | Backed up | Script |
| ------------------------------- | ------------------------------------------------ | --------- | ------ |
| `<SERVICE_ID>/project_data`     | shared project datasets                          | ✅ yes    | ✅ this is where it writes |
| `<SERVICE_ID>/scratch`          | staging for a job; delete when the job is done   | ❌ no     | ❌ manual `scp`/`rsync`    |
| `<SERVICE_ID>/archive`          | cold storage of things you rarely read           | ❌ no     | ❌ manual `scp`/`rsync`    |
| `~`                             | nothing large — quota is ~1 GB                   | —         | ❌                         |

The script deliberately only writes under `project_data`, since that is the
backed-up area; `--out-path` cannot escape it. Use the manual commands below for
`scratch` or `archive`.

`project_data` and `scratch` are symlinks to the real filesystems, so always use
the path relative to `$HOME` rather than the physical mount, which can change.

## Progress reporting

- **In a terminal**, `rsync` or `scp` draws its own live meter. With `rsync` ≥ 3.1
  the script passes `--info=progress2`, giving one whole-transfer bar with
  percentage and ETA.
- **When output is redirected** (log file, `nohup`, CI), both tools silently drop
  their meters — they only print to a TTY. The script detects this and instead
  polls the remote file size every `PROGRESS_INTERVAL` seconds (default 10):

  ```
  ==> uploading artificial_lunar_landscape.zip (5.0 GiB) to ...:pl1200-01/project_data/ [env: prod]
      10.2 MiB / 5.0 GiB (0%)  3.4 MiB/s  ETA 00:19:42
      22.9 MiB / 5.0 GiB (0%)  4.2 MiB/s  ETA 00:19:20
  ```

  Set the interval per run:
  `PROGRESS_INTERVAL=30 bash scripts/upload_file_to_cluster.sh --file data/foo.zip`.

  Because this counts bytes *present on the cluster*, a resumed `rsync` transfer
  starts the count at the existing offset rather than at zero.

## Integrity check

After the transfer the script compares the local and remote `sha256` and exits
non-zero on a mismatch, so a truncated upload fails loudly instead of leaving a
corrupt archive in place:

```
==> verifying checksum
    sha256 53836fc62103dd2a7bfcdb6489b2de6c1265ef9545ec9974004b17de0521d7bb
==> done: pl1200-01/project_data/artificial_lunar_landscape.zip
```

Hashing a multi-GB file costs a minute or two on both ends. If no `sha256sum` /
`shasum` exists locally the script says so and skips the check.

## Large files, resume, and Windows

`rsync` is used when available, with `--partial --append-verify`, so an
interrupted upload continues from where it stopped. Compression is deliberately
off — a zip is already compressed.

**Git Bash on Windows ships no `rsync`**, so there the script falls back to
`scp`, which cannot resume: a dropped connection means starting over. For
multi-GB archives on a shaky link, run it from WSL instead to get resume.

Measured throughput to Eagle is roughly 4 MiB/s, i.e. about 20 minutes per 5 GB.
For an unattended run:

```bash
nohup bash scripts/upload_file_to_cluster.sh --file data/artificial_lunar_landscape.zip > upload.log 2>&1 &
tail -f upload.log
```

## After the upload

Unpack on the cluster (the login node is fine for a quick unzip; use an
`srun`/`sbatch` job for anything heavy):

```bash
cd ~/pl1200-01/project_data/datasets
unzip ~/pl1200-01/project_data/artificial_lunar_landscape.zip -d artificial_lunar_landscape
```

## Doing it by hand

The script is a convenience wrapper; these are the underlying commands.

```bash
# single file
scp -i <private_key> data/foo.zip <username>@eagle.man.poznan.pl:pl1200-01/project_data/

# resumable, with progress
rsync -h --info=progress2 --partial --append-verify \
  -e "ssh -i <private_key>" \
  data/foo.zip <username>@eagle.man.poznan.pl:pl1200-01/project_data/

# whole directory
scp -r -i <private_key> data/ <username>@eagle.man.poznan.pl:pl1200-01/project_data/

# pulling results back
scp -i <private_key> <username>@eagle.man.poznan.pl:pl1200-01/project_data/results.zip .
```

| Part                             | Meaning                                                            |
| -------------------------------- | ------------------------------------------------------------------ |
| `-i <private_key>`               | Private SSH key, same one [`connect_to_cluster.sh`](../scripts/connect_to_cluster.sh) uses. |
| `<user>@<host>:<path>`           | Remote target; `<path>` is relative to `$HOME` unless it starts with `/`. |
| trailing `/` on the destination  | Treat the target as a directory — it must already exist.           |
| `-r`                             | Recurse into a directory (`scp` only; `rsync` uses `-a`).          |
| `--partial --append-verify`      | Resume a part-transferred file instead of restarting it.           |

## Troubleshooting

| Symptom                                  | Cause / fix                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------ |
| `Permission denied (publickey)`           | Wrong `SSH_KEY` or `USERNAME` in the env file; check with `bash scripts/connect_to_cluster.sh`. |
| `error: .env.prod not found`               | Copy [`.env.example`](../.env.example) and fill it in.                               |
| `Disk quota exceeded`                     | You are writing to `~` instead of a grant directory, or the grant is full (`df -h ~/pl1200-01/project_data`). |
| `error: checksum mismatch`                 | Transfer truncated — delete the remote file and re-run.                              |
| `error: --env must be 'prod' or 'test'`    | Only those two env files are supported; add one if you need a third.                 |
| `error: --out-path must stay inside project_data` | The value was absolute or contained `..`; pass a plain relative subfolder.    |
| No progress output at all                  | Output is a TTY and the tool's own meter is being used; or lower `PROGRESS_INTERVAL`. |
| `scp: ... No such file or directory`       | Remote directory missing — the script creates it, by hand you must `mkdir -p` first. |

> Sources for this page are listed in [`resources.md`](resources.md).
