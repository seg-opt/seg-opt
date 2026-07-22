#!/usr/bin/env bash
set -euo pipefail

# Resolve the repo root (this script lives in <root>/scripts) so .env.test is
# found no matter where the script is invoked from.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
env_file="$root_dir/.env.test"

if [[ ! -f "$env_file" ]]; then
  echo "error: .env.test not found at $env_file" >&2
  exit 1
fi

# Load .env.test, stripping any CR so Windows-edited files work too.
set -a
. <(tr -d '\r' < "$env_file")
set +a

key=$(mktemp)
chmod 600 "$key"
printf '%s\n' "$SSH_KEY" > "$key"
trap 'rm -f "$key"' EXIT

ssh -i "$key" "$USERNAME@$CLUSTER_ADDRESS"
