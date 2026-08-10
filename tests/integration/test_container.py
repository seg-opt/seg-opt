"""Integration tests for the shared Singularity/Apptainer setup on Eagle."""

from __future__ import annotations

import shlex

import pytest

RUNTIME_COMMAND = "runtime=$(command -v singularity || command -v apptainer || true)"


def _on_compute_node(env, command):
    account = shlex.quote(env["SERVICE_ID"])
    return (
        f"srun --account={account} --partition=proxima --time=00:02:00 --mem=1G "
        f"bash -lc {shlex.quote(command)}"
    )


def test_container_runtime_available(remote, connected, env):
    command = f'{RUNTIME_COMMAND}; test -n "$runtime" && "$runtime" --version'
    result = remote.sh(_on_compute_node(env, command), timeout=300)
    assert result.returncode == 0, (
        result.stderr.strip() or "container runtime unavailable"
    )
    assert (
        "singularity" in result.stdout.lower() or "apptainer" in result.stdout.lower()
    )


def test_container_storage_writable(remote, connected, env):
    service_id = shlex.quote(env["SERVICE_ID"])
    result = remote.sh(
        f'root="$HOME"/{service_id}/project_data/containers/seg-opt; '
        'test -d "$root/images" && test -w "$root/images"'
    )
    assert result.returncode == 0, "run scripts/setup_container_storage.sh first"


def test_configured_container_executes(remote, connected, env):
    image = env.get("CONTAINER_IMAGE", "")
    if not image:
        pytest.skip("CONTAINER_IMAGE is not configured")
    image_path = shlex.quote(image)
    command = (
        f'{RUNTIME_COMMAND}; test -n "$runtime" && '
        f'"$runtime" inspect {image_path} >/dev/null && '
        f'"$runtime" exec {image_path} python3 /opt/seg-opt/smoke_test.py'
    )
    result = remote.sh(_on_compute_node(env, command), timeout=300)
    assert result.returncode == 0, result.stderr.strip() or result.stdout.strip()


def test_container_sees_h100(remote, connected, env):
    if env.get("RUN_CONTAINER_GPU_TEST") != "1":
        pytest.skip("set RUN_CONTAINER_GPU_TEST=1 to allocate an H100 smoke job")
    image = env.get("CONTAINER_IMAGE", "")
    if not image:
        pytest.skip("CONTAINER_IMAGE is not configured")

    account = shlex.quote(env["SERVICE_ID"])
    image_path = shlex.quote(image)
    command = (
        f'{RUNTIME_COMMAND}; test -n "$runtime" && '
        f'"$runtime" exec --nv {image_path} '
        "python3 /opt/seg-opt/smoke_test.py --require-gpu"
    )
    result = remote.sh(
        f"srun --account={account} --partition=proxima --constraint=h100 "
        f"--gpus-per-node=1 --time=00:05:00 bash -lc {shlex.quote(command)}",
        timeout=600,
    )
    assert result.returncode == 0, result.stderr.strip() or result.stdout.strip()
