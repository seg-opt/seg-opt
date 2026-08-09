#!/usr/bin/env bash
# Deploy a CI-built SIF to immutable shared storage and promote current.sif.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/deploy_container.sh --file <image.sif> --version <version> [--env prod|test]

The SIF is uploaded to:
  ~/<SERVICE_ID>/project_data/containers/seg-opt/images/<version>/
USAGE
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
env_name=prod
image_file=
version=

while (($#)); do
  case $1 in
    --file) image_file=${2:?missing value for $1}; shift 2 ;;
    --version) version=${2:?missing value for $1}; shift 2 ;;
    --env) env_name=${2:?missing value for $1}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unexpected argument '$1'" >&2; usage >&2; exit 1 ;;
  esac
done

[[ -f $image_file ]] || { echo "error: --file must name an existing SIF" >&2; exit 1; }
[[ $image_file == *.sif ]] || { echo "error: --file must end in .sif" >&2; exit 1; }
[[ $version =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "error: --version may contain only letters, digits, '.', '_' and '-'" >&2
  exit 1
}
case $env_name in
  prod|test) ;;
  *) echo "error: --env must be 'prod' or 'test'" >&2; exit 1 ;;
esac

env_file="$root_dir/.env.$env_name"
[[ -f $env_file ]] || { echo "error: $env_file not found" >&2; exit 1; }
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

if ! command -v sha256sum >/dev/null 2>&1; then
  echo "error: sha256sum is required" >&2
  exit 1
fi

name=$(basename "$image_file")
[[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]*\.sif$ ]] || {
  echo "error: SIF filename may contain only letters, digits, '.', '_' and '-'" >&2
  exit 1
}
expected_sum=$(sha256sum < "$image_file" | cut -d' ' -f1)
remote_subdir="containers/seg-opt/images/$version"
remote_dir="$SERVICE_ID/project_data/$remote_subdir"

key=$(mktemp)
checksum_dir=$(mktemp -d)
chmod 600 "$key"
printf '%s\n' "$SSH_KEY" > "$key"
printf '%s  %s\n' "$expected_sum" "$name" > "$checksum_dir/$name.sha256"
trap 'rm -f "$key"; rm -rf "$checksum_dir"' EXIT

ssh_opts=(
  -i "$key"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
)
remote="$USERNAME@$CLUSTER_ADDRESS"

existing_sum=$(ssh "${ssh_opts[@]}" "$remote" bash -s -- "$remote_dir" "$name" <<'REMOTE'
set -euo pipefail
remote_dir=$1
name=$2
if [[ -f "$remote_dir/$name" ]]; then
  sha256sum < "$remote_dir/$name" | cut -d' ' -f1
fi
REMOTE
)
if [[ -n $existing_sum && $existing_sum != "$expected_sum" ]]; then
  echo "error: immutable version $version already contains a different $name" >&2
  echo "  local:  $expected_sum" >&2
  echo "  remote: $existing_sum" >&2
  exit 1
fi

if [[ -z $existing_sum ]]; then
  "$script_dir/upload_file_to_cluster.sh" \
    --file "$image_file" --env "$env_name" --out-path "$remote_subdir"
  "$script_dir/upload_file_to_cluster.sh" \
    --file "$checksum_dir/$name.sha256" --env "$env_name" --out-path "$remote_subdir"
else
  echo "==> immutable image already present with matching checksum; skipping upload"
fi

echo "==> verifying and promoting $version on Eagle"
ssh "${ssh_opts[@]}" "$remote" bash -s -- "$SERVICE_ID" "$version" "$name" <<'REMOTE'
set -euo pipefail
service_id=$1
version=$2
name=$3
root="$HOME/$service_id/project_data/containers/seg-opt"
image="$root/images/$version/$name"

cd "$(dirname "$image")"
sha256sum --check "$name.sha256"

temporary_link="$root/.current.sif.$USER.$$"
ln -s "images/$version/$name" "$temporary_link"
mv -Tf "$temporary_link" "$root/current.sif"
echo "current.sif -> images/$version/$name"
REMOTE