"""Integration test: CPU compute on the PCSS Eagle login node."""
import time


def test_cpu_cores_present(remote, connected):
    """The login node reports at least one usable CPU core."""
    r = remote.sh("nproc")
    assert r.returncode == 0, f"nproc failed: {r.stderr.strip()}"
    cores = int(r.stdout.strip())
    assert cores >= 1, f"expected >=1 CPU core, got {cores}"


def test_cpu_compute_correct_and_timely(remote, connected):
    """Run a real numeric computation remotely, verify correctness and speed."""
    n = 5_000_000
    code = (
        "import time\n"
        f"n = {n}\n"
        "t = time.time()\n"
        "total = sum(i * i for i in range(n))\n"
        "print(total, round(time.time() - t, 3))\n"
    )
    r = remote.py(code)
    assert r.returncode == 0, f"remote compute failed: {r.stderr.strip()}"

    total_str, elapsed_str = r.stdout.split()
    # Closed form for sum of squares 0..n-1: (n-1)*n*(2n-1)/6.
    expected = (n - 1) * n * (2 * n - 1) // 6
    assert int(total_str) == expected, "remote computed an incorrect result"
    # Sanity bound on compute power (pure-Python loop of 5M on one core).
    assert float(elapsed_str) < 30, f"compute unexpectedly slow: {elapsed_str}s"


def test_cpu_roundtrip_latency(remote, connected):
    """A trivial remote command returns quickly (connection is usable)."""
    start = time.time()
    r = remote.sh("true")
    assert r.returncode == 0
    assert time.time() - start < 20, "SSH round-trip too slow"
