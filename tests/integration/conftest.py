"""Shared fixtures for PCSS Eagle integration tests.

These tests hit the real cluster over SSH using the credentials in ./.env
(the same file connect.sh uses). If .env is missing or incomplete, the tests
skip rather than fail, so the suite is safe to run in CI without secrets.
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Reuse the .env parser + key handling from the project scripts.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from run_gpu import load_env, write_keyfile, SSH_OPTS  # noqa: E402

REQUIRED = ("SSH_KEY", "USERNAME", "CLUSTER_ADDRESS")


@pytest.fixture(scope="session")
def env():
    envfile = REPO_ROOT / ".env"
    if not envfile.exists():
        pytest.skip(".env not found -- skipping PCSS integration tests")
    e = load_env(envfile)
    missing = [k for k in REQUIRED if not e.get(k)]
    if missing:
        pytest.skip(f".env missing {missing} -- skipping PCSS integration tests")
    return e


@pytest.fixture(scope="session")
def target(env):
    return f"{env['USERNAME']}@{env['CLUSTER_ADDRESS']}"


@pytest.fixture(scope="session")
def keyfile(env):
    kf = write_keyfile(env["SSH_KEY"])
    yield kf
    try:
        os.unlink(kf)
    except OSError:
        pass


class Remote:
    """Thin SSH runner returning CompletedProcess (never raises on remote error)."""

    def __init__(self, keyfile: str, target: str):
        self._key = keyfile
        self._target = target

    def sh(self, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", "-i", self._key, *SSH_OPTS, "-o", "ConnectTimeout=20",
             self._target, command],
            capture_output=True, text=True, timeout=timeout,
        )

    def py(self, code: str, timeout: int = 120) -> subprocess.CompletedProcess:
        """Run Python 3 code on the login node (base64-piped to avoid quoting)."""
        b64 = base64.b64encode(code.encode()).decode()
        return self.sh(f"python3 -c 'import base64;exec(base64.b64decode(\"{b64}\"))'", timeout)


@pytest.fixture(scope="session")
def remote(keyfile, target) -> Remote:
    return Remote(keyfile, target)


@pytest.fixture(scope="session")
def connected(remote):
    """Verify basic SSH connectivity once; skip the rest if the cluster is unreachable."""
    r = remote.sh("echo __alive__")
    if r.returncode != 0 or "__alive__" not in r.stdout:
        pytest.skip(f"cannot reach cluster over SSH: {r.stderr.strip() or r.returncode}")
    return True
