from pathlib import Path
import re

from src.utils.utils import load_config, load_yaml


REPLICA_SEEDS = {}


def test_training_configs_use_seed42_except_explicit_named_replicas():
    training_paths = set()
    for path in sorted(Path("experiments").rglob("*.yaml")):
        if "training" not in load_yaml(path):
            continue
        training_paths.add(path.as_posix())
        expected = REPLICA_SEEDS.get(path.as_posix(), 42)
        assert load_config(path).training.seed == expected, str(path)

    assert training_paths
    for name, seed in REPLICA_SEEDS.items():
        path = Path(name)
        assert path.is_file(), name
        assert name in training_paths, name
        match = re.search(r"(?:^|_)seed(43|44)$", path.stem)
        assert match is not None, name
        assert int(match.group(1)) == seed, name
        assert load_config(path).training.seed == seed, name
