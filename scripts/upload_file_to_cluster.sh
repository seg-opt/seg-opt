#!/usr/bin/env bash
#
# Copy a file from this machine to PCSS Eagle.
#
# Usage:
#   bash scripts/upload_file_to_cluster.sh --file <path> [--env prod|test]
#                                          [--out-path <subdir>]
#
#   --file      Path to the local file to upload. Required; any file type.
#   --env       Which env file supplies SSH_KEY, USERNAME, CLUSTER_ADDRESS and
#               SERVICE_ID: "prod" -> .env.prod, "test" -> .env.test.
#               Defaults to prod.
#   --out-path  Subfolder inside the grant's project_data. Optional and always
#               relative, so the upload lands in
#               <SERVICE_ID>/project_data/<out-path>.
#   -h, --help  Show this help and exit.
#
# Progress is live on a terminal (rsync/scp draw their own meter). When output is
# redirected, the script instead polls the remote file size every
# PROGRESS_INTERVAL seconds (default 10) and logs sent/total, rate and ETA.
#
set -euo pipefail

usage() { sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# Resolve the repo root (this script lives in <root>/scripts) so the env file is
# found no matter where the script is invoked from.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)

env_name=prod
local_file=
out_path=
while (($#)); do
  case $1 in
    --file)     local_file=${2:?missing value for $1}; shift 2 ;;
    --env)      env_name=${2:?missing value for $1}; shift 2 ;;
    --out-path) out_path=${2:?missing value for $1}; shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "error: unexpected argument '$1'" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z $local_file ]]; then
  echo "error: --file is required" >&2
  usage >&2
  exit 1
fi

if [[ -d $local_file ]]; then
  echo "error: $local_file is a directory; this script uploads a single file" >&2
  exit 1
fi

if [[ ! -f $local_file ]]; then
  echo "error: $local_file is not a regular file" >&2
  exit 1
fi

case $env_name in
  prod|test) ;;
  *) echo "error: --env must be 'prod' or 'test', got '$env_name'" >&2; exit 1 ;;
esac

env_file="$root_dir/.env.$env_name"

if [[ ! -f "$env_file" ]]; then
  echo "error: .env.$env_name not found at $env_file" >&2
  exit 1
fi

# Load the env file, stripping any CR so Windows-edited files work too.
set -a
. <(tr -d '\r' < "$env_file")
set +a

: "${USERNAME:?USERNAME is not set in .env.$env_name}"
: "${CLUSTER_ADDRESS:?CLUSTER_ADDRESS is not set in .env.$env_name}"
: "${SSH_KEY:?SSH_KEY is not set in .env.$env_name}"

# Home is only ~1 GB, so data goes under the grant named by SERVICE_ID. The path
# stays relative to $HOME on the cluster: project_data is a symlink whose physical
# mount can change, so we never hardcode it.
: "${SERVICE_ID:?SERVICE_ID is not set in .env.$env_name}"

# --out-path is always interpreted inside project_data; reject anything that
# would climb out of it, and tolerate stray slashes.
if [[ $out_path == /* ]]; then
  echo "error: --out-path must be relative to project_data, got '$out_path'" >&2
  exit 1
fi
if [[ $out_path == ".." || $out_path == "../"* || $out_path == *"/../"* || $out_path == *"/.." ]]; then
  echo "error: --out-path must stay inside project_data, got '$out_path'" >&2
  exit 1
fi
out_path=${out_path%/}

remote_dir="$SERVICE_ID/project_data${out_path:+/$out_path}"

key=$(mktemp)
chmod 600 "$key"
printf '%s\n' "$SSH_KEY" > "$key"
trap 'rm -f "$key"; [[ -n ${monitor_pid:-} ]] && kill "$monitor_pid" 2>/dev/null' EXIT

# BatchMode: never hang on a prompt. accept-new: trust the login node the first
# time but still detect a changed host key later.
ssh_opts=(
  -i "$key"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
)
remote="$USERNAME@$CLUSTER_ADDRESS"
name=$(basename "$local_file")
total_bytes=$(wc -c < "$local_file" | tr -d '[:space:]')

# Bytes -> human units. numfmt isn't everywhere (macOS), so do it by hand.
human() {
  awk -v b="$1" 'BEGIN {
    split("B KiB MiB GiB TiB", u, " ")
    i = 1
    while (b >= 1024 && i < 5) { b /= 1024; i++ }
    printf (i == 1 ? "%d %s" : "%.1f %s"), b, u[i]
  }'
}

# Poll the growing remote file and report sent/total, rate and ETA. Used when
# stdout is not a terminal, because scp and rsync both suppress their own
# progress meters unless they can see a TTY (piping to a log, nohup, CI).
progress_monitor() {
  local interval=${PROGRESS_INTERVAL:-10}
  local prev=0 sent elapsed=0 delta rate eta pct
  while sleep "$interval"; do
    sent=$(ssh "${ssh_opts[@]}" "$remote" \
      "stat -c %s -- \"$remote_dir/$name\" 2>/dev/null || echo 0" 2>/dev/null) || sent=""
    [[ $sent =~ ^[0-9]+$ ]] || continue
    elapsed=$((elapsed + interval))
    delta=$((sent - prev))
    prev=$sent
    pct=$((total_bytes > 0 ? sent * 100 / total_bytes : 0))
    rate=$((delta / interval))
    if ((rate > 0)); then
      eta=$(( (total_bytes - sent) / rate ))
      eta=$(printf '%02d:%02d:%02d' $((eta / 3600)) $((eta % 3600 / 60)) $((eta % 60)))
    else
      eta="--:--:--"
    fi
    printf '    %s / %s (%d%%)  %s/s  ETA %s\n' \
      "$(human "$sent")" "$(human "$total_bytes")" "$pct" "$(human "$rate")" "$eta"
  done
}

echo "==> uploading $name ($(human "$total_bytes")) to $remote:$remote_dir/ [env: $env_name]"

ssh "${ssh_opts[@]}" "$remote" "mkdir -p -- \"$remote_dir\""

# On a terminal, let the transfer tool draw its own live meter; otherwise poll
# the remote size ourselves so a logged run still reports how far it got.
if [[ ! -t 1 ]]; then
  progress_monitor &
  monitor_pid=$!
fi

if command -v rsync >/dev/null 2>&1; then
  # --partial + --append-verify so an interrupted upload resumes instead of
  # restarting; no -z, most of what we ship is already compressed.
  progress_flag=--progress
  # progress2 gives one whole-transfer bar with percent and ETA (rsync >= 3.1).
  if rsync --info=help >/dev/null 2>&1; then
    progress_flag=--info=progress2
  fi
  rsync -h "$progress_flag" --partial --append-verify \
    -e "ssh ${ssh_opts[*]}" \
    -- "$local_file" "$remote:$remote_dir/"
else
  echo "note: rsync not found, falling back to scp (no resume)" >&2
  scp "${ssh_opts[@]}" -- "$local_file" "$remote:$remote_dir/"
fi

if [[ -n ${monitor_pid:-} ]]; then
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  monitor_pid=
fi

# Confirm the bytes that landed match what we sent.
# Hash from stdin: with a filename argument, GNU coreutils escapes names that
# contain backslashes (Windows paths) and prefixes the line with '\'.
local_sum=""
if command -v sha256sum >/dev/null 2>&1; then
  local_sum=$(sha256sum < "$local_file" | cut -d' ' -f1)
elif command -v shasum >/dev/null 2>&1; then
  local_sum=$(shasum -a 256 < "$local_file" | cut -d' ' -f1)
fi

if [[ -n "$local_sum" ]]; then
  echo "==> verifying checksum"
  remote_sum=$(ssh "${ssh_opts[@]}" "$remote" "sha256sum -- \"$remote_dir/$name\" | cut -d' ' -f1")
  if [[ "$local_sum" != "$remote_sum" ]]; then
    echo "error: checksum mismatch for $name" >&2
    echo "  local:  $local_sum" >&2
    echo "  remote: $remote_sum" >&2
    exit 1
  fi
  echo "    sha256 $local_sum"
else
  echo "note: no sha256 tool locally, skipping checksum verification" >&2
fi

echo "==> done: $remote_dir/$name"
