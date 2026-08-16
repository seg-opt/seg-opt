from pathlib import Path

import pytest
import torch
from torch import nn

from src.utils.artifacts import parameter_counts, sha256_file, to_builtin, write_json
from src.utils.utils import BASELINE_NAMES, load_config, write_resolved_config


def test_all_canonical_baseline_configs_are_valid_and_distillation_free():
    config_paths = sorted(Path("experiments/baselines").glob("*.yaml"))

    configs = [load_config(path) for path in config_paths]

    assert {config.model.name for config in configs} == BASELINE_NAMES
    assert all(config.training.seed == 42 for config in configs)
    assert all(
        config.training.batch_size * config.training.accumulate_grad_batches == 16
        for config in configs
    )
    assert all(config.data.image_height == 512 for config in configs)
    assert all(config.data.image_width == 768 for config in configs)
    assert all(Path(config.data.dataset_root).is_absolute() for config in configs)
    assert all(Path(config.data.split_manifest).is_absolute() for config in configs)
    assert all(Path(config.training.output_dir).is_absolute() for config in configs)
    for path in config_paths:
        contents = path.read_text(encoding="utf-8")
        assert "distillation:" not in contents
        assert "teacher:" not in contents
        assert "student:" not in contents


def test_resolved_configuration_round_trip(tmp_path):
    config = load_config("experiments/baselines/fast_scnn.yaml")
    resolved_path = tmp_path / "config.yaml"

    write_resolved_config(config, resolved_path)

    assert load_config(resolved_path) == config


def test_artifact_helpers_return_json_native_values(tmp_path):
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    model[0].requires_grad_(False)
    source = tmp_path / "checkpoint.ckpt"
    source.write_bytes(b"baseline-checkpoint")
    destination = tmp_path / "metrics.json"
    contents = {
        "counts": parameter_counts(model),
        "metric": torch.tensor(0.75),
        "confusion": torch.tensor([[1, 2], [3, 4]]),
        "path": source,
    }

    write_json(destination, contents)

    assert sha256_file(source) == "a47916d5f2e5b324bd8d58f1a672e9dbcae084f4ce451ec0885501df9a87f61f"
    assert parameter_counts(model) == {"total": 26, "trainable": 10}
    assert to_builtin(contents)["metric"] == pytest.approx(0.75)
    assert '"confusion": [' in destination.read_text(encoding="utf-8")
