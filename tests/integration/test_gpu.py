"""Integration test: GPU availability on PCSS Eagle (H100 only, for now)."""
import re

# Node states (SLURM %t) that mean the node is up and can run jobs.
SCHEDULABLE = ("idle", "mix", "alloc")


def _gpu_node_lines(remote):
    """Return sinfo lines '<state> <gres>' for every GPU node (deduped by node)."""
    r = remote.sh("sinfo -N -h -o '%N %t %G' | grep -i gpu || true")
    assert r.returncode == 0, f"sinfo failed: {r.stderr.strip()}"
    seen, rows = set(), []
    for line in r.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        node, state, gres = parts
        if node in seen:
            continue
        seen.add(node)
        rows.append((state, gres))
    return rows


def test_h100_nodes_exist(remote, connected):
    """The cluster advertises H100 GPUs."""
    rows = _gpu_node_lines(remote)
    h100 = [g for _, g in rows if "h100" in g.lower()]
    assert h100, "no H100 GPU nodes found in sinfo"


def test_h100_currently_available(remote, connected):
    """At least one H100 node is up and schedulable (idle/mix), not all down."""
    rows = _gpu_node_lines(remote)
    avail = [
        (s, g) for s, g in rows
        if "h100" in g.lower() and any(state in s for state in ("idle", "mix"))
    ]
    assert avail, (
        "no H100 node currently available (idle/mix); "
        f"states seen: {sorted({s for s, g in rows if 'h100' in g.lower()})}"
    )


def test_only_h100_gpu_type_available(remote, connected):
    """Among schedulable GPU nodes, H100 is the only usable type (V100 is in maintenance)."""
    rows = _gpu_node_lines(remote)
    types = set()
    for state, gres in rows:
        if not any(s in state for s in SCHEDULABLE):
            continue
        for m in re.finditer(r"gpu:([A-Za-z0-9]+):", gres):
            types.add(m.group(1).lower())
    assert types, "no schedulable GPU nodes found at all"
    assert types == {"h100"}, f"expected only 'h100' available, got {sorted(types)}"
