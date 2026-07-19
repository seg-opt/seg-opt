"""Integration test: we can connect to PCSS Eagle over SSH."""


def test_ssh_connection(remote, env):
    """SSH login succeeds and lands us on the expected account."""
    r = remote.sh("echo pcss_ok && whoami && hostname")
    assert r.returncode == 0, f"ssh failed: {r.stderr.strip()}"

    lines = r.stdout.split()
    assert "pcss_ok" in lines, f"unexpected output: {r.stdout!r}"
    # whoami must match the configured username.
    assert env["USERNAME"] in r.stdout, f"logged in as unexpected user: {r.stdout!r}"


def test_slurm_available(remote, connected):
    """The SLURM scheduler is reachable from the login node."""
    r = remote.sh("sinfo --version")
    assert r.returncode == 0, f"sinfo not available: {r.stderr.strip()}"
    assert r.stdout.lower().startswith("slurm"), f"unexpected sinfo output: {r.stdout!r}"
