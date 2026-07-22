"""Shared fixtures for PCSS Eagle integration tests.

These tests hit the real cluster over SSH using the credentials in a dotenv
file. By default that file is ./.env.test; pass --env to use another (e.g.
--env .env.prod). Remote commands run inside the service directory named by
SERVICE_ID (a folder in the cluster home, e.g. ~/pl1234-01).

If the env file is missing or incomplete, the tests skip rather than fail, so
the suite is safe to run in CI without secrets.
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Local .env parser + key handling (kept inside the integration suite).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster import load_env, write_keyfile, SSH_OPTS  # noqa: E402

REQUIRED = ("SSH_KEY", "USERNAME", "CLUSTER_ADDRESS", "SERVICE_ID")


def pytest_addoption(parser):
    parser.addoption(
        "--env",
        action="store",
        default=".env.test",
        help="dotenv file with cluster credentials (relative to repo root or "
             "absolute). Default: .env.test. Example: --env .env.prod",
    )


@pytest.fixture(scope="session")
def env(pytestconfig):
    name = pytestconfig.getoption("--env")
    envfile = Path(name)
    if not envfile.is_absolute():
        envfile = REPO_ROOT / name
    if not envfile.exists():
        pytest.skip(f"{name} not found -- skipping PCSS integration tests")
    e = load_env(envfile)
    missing = [k for k in REQUIRED if not e.get(k)]
    if missing:
        pytest.skip(f"{name} missing {missing} -- skipping PCSS integration tests")
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
    """Thin SSH runner returning CompletedProcess (never raises on remote error).

    Commands run inside the service directory ~/<workdir> by default so tests
    operate within the configured service. Pass cd=False for connectivity/setup
    checks that must not depend on that directory existing.
    """

    def __init__(self, keyfile: str, target: str, workdir: str):
        self._key = keyfile
        self._target = target
        self._workdir = workdir

    def _wrap(self, command: str, cd: bool) -> str:
        if cd and self._workdir:
            return f'cd "$HOME/{self._workdir}" && {{ {command}; }}'
        return command

    def sh(self, command: str, timeout: int = 60, cd: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", "-i", self._key, *SSH_OPTS, "-o", "ConnectTimeout=20",
             self._target, self._wrap(command, cd)],
            capture_output=True, text=True, timeout=timeout,
        )

    def py(self, code: str, timeout: int = 120, cd: bool = True) -> subprocess.CompletedProcess:
        """Run Python 3 code on the login node (base64-piped to avoid quoting)."""
        b64 = base64.b64encode(code.encode()).decode()
        return self.sh(
            f"python3 -c 'import base64;exec(base64.b64decode(\"{b64}\"))'",
            timeout, cd=cd,
        )


@pytest.fixture(scope="session")
def remote(keyfile, target, env) -> Remote:
    return Remote(keyfile, target, env["SERVICE_ID"])


@pytest.fixture(scope="session")
def connected(remote, env):
    """Verify SSH connectivity and that the service directory exists (once).

    Unreachable cluster (bad network/creds) -> skip, so the suite stays safe in
    CI without secrets. Reachable but SERVICE_ID points to a missing folder ->
    fail, because that is a real misconfiguration the tests cannot work around.
    """
    r = remote.sh("echo __alive__", cd=False)
    if r.returncode != 0 or "__alive__" not in r.stdout:
        pytest.skip(f"cannot reach cluster over SSH: {r.stderr.strip() or r.returncode}")
    sid = env["SERVICE_ID"]
    d = remote.sh(f'test -d "$HOME/{sid}"', cd=False)
    if d.returncode != 0:
        pytest.fail(f"service directory ~/{sid} (SERVICE_ID) not found on cluster")
    return True
