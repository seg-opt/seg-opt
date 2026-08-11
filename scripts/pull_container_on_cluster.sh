#!/usr/bin/env bash
# Pull an OCI image directly from a registry into immutable Eagle SIF storage.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/pull_container_on_cluster.sh --image <registry-image> --version <version> [--kind diagnostic|development] [--env prod|test]

Runs Singularity pull on an Eagle proxima-cpu worker. The resulting SIF and its
SHA-256 checksum are written directly to grant-backed immutable image storage.
For private GHCR packages, first log in interactively on Eagle with:
  srun --account=<grant> --partition=proxima-cpu --mem=4G --time=00:10:00 singularity registry login --username <github-user> docker://ghcr.io
USAGE
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
env_name=prod
image=
version=
kind=development

while (($#)); do
  case $1 in
    --image) image=${2:?missing value for $1}; shift 2 ;;
    --version) version=${2:?missing value for $1}; shift 2 ;;
    --kind) kind=${2:?missing value for $1}; shift 2 ;;
    --env) env_name=${2:?missing value for $1}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unexpected argument '$1'" >&2; usage >&2; exit 1 ;;
  esac
done

[[ $image == docker://* ]] || { echo "error: --image must start with docker://" >&2; exit 1; }
[[ $version =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "error: invalid --version" >&2; exit 1; }
case $kind in diagnostic|development) ;; *) echo "error: invalid --kind" >&2; exit 1 ;; esac
case $env_name in prod|test) ;; *) echo "error: invalid --env" >&2; exit 1 ;; esac

env_file="$root_dir/.env.$env_name"
[[ -f $env_file ]] || { echo "error: $env_file not found" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
. <(tr -d '\r' < "$env_file")
set +a
: "${USERNAME:?USERNAME is not set in .env.$env_name}"
: "${CLUSTER_ADDRESS:?CLUSTER_ADDRESS is not set in .env.$env_name}"
: "${SERVICE_ID:?SERVICE_ID is not set in .env.$env_name}"

key=
trap '[[ -z $key ]] || rm -f "$key"' EXIT
ssh_opts=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
if [[ -n ${SSH_KEY:-} ]]; then
  key=$(mktemp)
  chmod 600 "$key"
  printf '%s\n' "$SSH_KEY" > "$key"
  ssh_opts=(-i "$key" -o IdentitiesOnly=yes "${ssh_opts[@]}")
elif [[ -z ${SSH_AUTH_SOCK:-} ]] || ! ssh-add -l >/dev/null 2>&1; then
  echo "error: set SSH_KEY in .env.$env_name or unlock an SSH key with ssh-add" >&2
  exit 1
fi

ssh "${ssh_opts[@]}" "$USERNAME@$CLUSTER_ADDRESS" bash -s -- "$SERVICE_ID" "$kind" "$version" "$image" <<'REMOTE'
set -euo pipefail
service_id=$1
kind=$2
version=$3
image=$4
root="$HOME/$service_id/project_data/containers/seg-opt"
[[ $kind == development ]] && root+=/development
name="seg-opt-${kind}-${version}.sif"
directory="$root/images/$version"
target="$directory/$name"

mkdir -p "$directory" "$root/cache/$USER" "$root/tmp/$USER"
promote() {
  local temporary_link="$root/.current.sif.$USER.$$"
  ln -s "images/$version/$name" "$temporary_link"
  mv -Tf "$temporary_link" "$root/current.sif"
  echo "current.sif -> images/$version/$name"
}

if [[ -f $target ]]; then
  echo "==> immutable image already present; verifying checksum"
  (cd "$directory" && sha256sum --check "$name.sha256")
  promote
  exit 0
fi

echo "==> pulling $image on a proxima-cpu worker"
srun --account="$service_id" --partition=proxima-cpu --mem=32G --time=00:30:00 \
  bash -s -- "$target" "$root/cache/$USER" "$root/tmp/$USER" "$image" <<'PULL'
set -euo pipefail
target=$1
cache=$2
temporary=$3
image=$4
export SINGULARITY_CACHEDIR="$cache" SINGULARITY_TMPDIR="$temporary"
temporary_target="${target}.partial.$$"
trap 'rm -f "$temporary_target"' EXIT
singularity pull "$temporary_target" "$image"
sha256sum "$temporary_target" > "${temporary_target}.sha256"
mv "$temporary_target" "$target"
mv "${temporary_target}.sha256" "${target}.sha256"
PULL

(cd "$directory" && sha256sum --check "$name.sha256")
promote
echo "==> deployed $target"
REMOTE