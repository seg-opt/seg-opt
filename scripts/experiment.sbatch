#!/usr/bin/env bash
set -euo pipefail
manifest=${1:?missing manifest path}
run_directory=$(cd "$(dirname "$manifest")" && pwd)
exec python3 -S "$run_directory/source/scripts/experiment.py" worker "$@"