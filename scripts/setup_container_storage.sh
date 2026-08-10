#!/usr/bin/env bash
# Prepare shared image storage and per-user Singularity caches on PCSS Eagle.
set -euo pipefail

usage() {
  echo "Usage: $0 [--env prod|test]" >&2
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
env_name=prod

while (($#)); do
  case $1 in
    --env) env_name=${2:?missing value for $1}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unexpected argument '$1'" >&2; usage; exit 1 ;;
  esac
done

case $env_name in
  prod|test) ;;
  *) echo "error: --env must be 'prod' or 'test'" >&2; exit 1 ;;
esac

env_file="$root_dir/.env.$env_name"
if [[ ! -f $env_file ]]; then
  echo "error: $env_file not found" >&2
  exit 1
fi

set -a
# The selected ignored dotenv file is intentionally dynamic.
# shellcheck disable=SC1090
. <(tr -d '\r' < "$env_file")
set +a

: "${SSH_KEY:?SSH_KEY is not set in .env.$env_name}"
: "${USERNAME:?USERNAME is not set in .env.$env_name}"
: "${CLUSTER_ADDRESS:?CLUSTER_ADDRESS is not set in .env.$env_name}"
: "${SERVICE_ID:?SERVICE_ID is not set in .env.$env_name}"
[[ $SERVICE_ID =~ ^[A-Za-z0-9._-]+$ ]] || { echo "error: invalid SERVICE_ID" >&2; exit 1; }

key=$(mktemp)
chmod 600 "$key"
printf '%s\n' "$SSH_KEY" > "$key"
trap 'rm -f "$key"' EXIT

ssh -i "$key" -o IdentitiesOnly=yes -o BatchMode=yes \
  -o StrictHostKeyChecking=accept-new "$USERNAME@$CLUSTER_ADDRESS" \
  bash -s -- "$SERVICE_ID" <<'REMOTE'
set -euo pipefail
service_id=$1

if ! sinfo -h -p proxima >/dev/null 2>&1; then
  echo "error: SLURM partition 'proxima' is not available" >&2
  exit 1
fi

root="$HOME/$service_id/project_data/containers/seg-opt"
umask 0002
mkdir -p "$root/images" "$root/cache/$USER" "$root/tmp/$USER"
chmod g+rwx "$root" "$root/images" "$root/cache" "$root/tmp" 2>/dev/null || true
test -w "$root/images"
test -w "$root/cache/$USER"
test -w "$root/tmp/$USER"

echo "images:  $root/images"
echo "cache:   $root/cache/$USER"
echo "temp:    $root/tmp/$USER"
echo
echo "Add these exports to jobs that pull or convert images:"
echo "export SINGULARITY_CACHEDIR=\"$root/cache/$USER\""
echo "export SINGULARITY_TMPDIR=\"$root/tmp/$USER\""
REMOTE