from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from src.baselines import load_distillation_teacher
from src.utils.artifacts import parameter_counts, sha256_file, to_builtin, write_json
from src.utils.utils import (
    BASELINE_NAMES,
    TeacherConfig,
    load_config,
    write_resolved_config,
)


def test_baselines_benchmark_configs_match_the_comparison_protocol():
    config_paths = sorted(
        Path("experiments/baselines_benchmark").glob("*.yaml")
    )
    configs = [load_config(path) for path in config_paths]

    assert {config.model.name for config in configs} == BASELINE_NAMES
    assert all(config.data.mask_variant == "clean" for config in configs)
    assert all(
        Path(config.data.split_manifest).name == "split_seed42_aligned.json"
        for config in configs
    )
    assert all(config.metrics.trimap_kernel_size == 7 for config in configs)
    assert all(config.training.seed == 42 for config in configs)
    assert all(config.data.image_height == 512 for config in configs)
    assert all(config.data.image_width == 768 for config in configs)
    assert all(Path(config.data.dataset_root).is_absolute() for config in configs)
    assert all(Path(config.data.split_manifest).is_absolute() for config in configs)
    assert all(Path(config.training.output_dir).is_absolute() for config in configs)
    assert all(
        config.training.batch_size * config.training.accumulate_grad_batches == 16
        for config in configs
    )
    assert all(config.distillation is None for config in configs)
    for path in config_paths:
        contents = path.read_text(encoding="utf-8")
        assert "distillation:" not in contents
        assert "teacher:" not in contents
        assert "student:" not in contents


def test_resolved_configuration_round_trip(tmp_path):
    config = load_config("experiments/baselines_benchmark/fast_scnn.yaml")
    resolved_path = tmp_path / "config.yaml"

    write_resolved_config(config, resolved_path)

    assert load_config(resolved_path) == config


def test_distillation_configuration_is_validated_and_round_trips(tmp_path):
    config_path = tmp_path / "bpkd.yaml"
    config_path.write_text(
        """
model: {name: fast_scnn, pretrained: false}
data:
  dataset_root: datasets/artificial_lunar_landscape
  split_manifest: experiments/baselines_benchmark/split_seed42_aligned.json
  mask_variant: clean
training:
  output_dir: results/bpkd/test
  batch_size: 1
  accumulate_grad_batches: 16
  learning_rate: 5.0e-4
metrics: {trimap_kernel_size: 7}
distillation:
  method: bpkd
  teacher:
    config_path: experiments/bpkd/mask2former_teacher_clean.yaml
    checkpoint_path: results/bpkd/teachers/mask2former/best.ckpt
  spatial_stride: 8
  edge_alpha: [1, 2, 3, 4]
  edge_kernel_size: 7
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data.mask_variant == "clean"
    assert config.metrics.trimap_kernel_size == 7
    assert config.distillation is not None
    assert config.distillation.edge_alpha == (1.0, 2.0, 3.0, 4.0)
    assert Path(config.distillation.teacher.config_path).is_absolute()


def test_segformer_b4_to_fast_scnn_logit_distillation_config():
    config = load_config(
        "experiments/logit_distillation/fast_scnn_from_segformer_b4.yaml"
    )

    assert config.model.name == "fast_scnn"
    assert not config.model.pretrained
    assert config.data.mask_variant == "clean"
    assert config.training.batch_size == 8
    assert config.training.accumulate_grad_batches == 2
    assert config.training.batch_size * config.training.accumulate_grad_batches == 16
    assert Path(config.training.output_dir).parts[-3:] == (
        "logit_distillation",
        "fast_scnn_from_segformer_b4",
        "seed42",
    )
    assert config.distillation is not None
    assert config.distillation.method == "vanilla"
    assert config.distillation.spatial_stride == 8
    assert config.distillation.vanilla_weight == 1.0
    assert config.distillation.vanilla_temperature == 1.0
    assert Path(config.distillation.teacher.config_path).parts[-2:] == (
        "baselines_benchmark",
        "segformer_b4.yaml",
    )
    assert Path(config.distillation.teacher.checkpoint_path).parts[-6:] == (
        "results",
        "baselines_benchmark",
        "segformer_b4",
        "seed42",
        "checkpoints",
        "best.ckpt",
    )


def test_checked_in_bpkd_configs_are_clean_and_match_the_declared_matrix():
    configs = [
        load_config(path)
        for path in sorted(Path("experiments/bpkd").glob("*.yaml"))
    ]

    assert len(configs) == 11
    assert all(config.data.mask_variant == "clean" for config in configs)
    assert all(config.metrics.trimap_kernel_size == 7 for config in configs)
    students = [config for config in configs if config.model.name == "fast_scnn"]
    assert len(students) == 8
    assert all(
        config.training.batch_size == 2
        and config.training.accumulate_grad_batches == 8
        for config in students
    )
    mask2former_students = [
        config
        for config in configs
        if config.model.name == "mask2former_swinl" and config.distillation is not None
    ]
    assert len(mask2former_students) == 2
    assert all(
        config.training.batch_size == 1
        and config.training.accumulate_grad_batches == 16
        for config in mask2former_students
    )


def test_config_rejects_unknown_distillation_fields(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
model: {name: fast_scnn, pretrained: false}
data:
  dataset_root: datasets/artificial_lunar_landscape
  split_manifest: experiments/baselines_benchmark/split_seed42_aligned.json
training:
  output_dir: results/bpkd/test
  batch_size: 1
  accumulate_grad_batches: 16
  learning_rate: 5.0e-4
distillation:
  method: vanilla
  teacher: {config_path: teacher.yaml, checkpoint_path: teacher.ckpt}
  typo: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported fields"):
        load_config(config_path)


def test_teacher_loader_validates_provenance_and_restores_only_model_weights(
    tmp_path,
    monkeypatch,
):
    config = load_config("experiments/bpkd/fast_scnn_bpkd_w7_clean.yaml")
    source = nn.Conv2d(3, 4, kernel_size=1)
    checkpoint = tmp_path / "teacher.ckpt"
    torch.save(
        {
            "state_dict": {
                f"model.{name}": value.detach().clone()
                for name, value in source.state_dict().items()
            }
        },
        checkpoint,
    )
    assert config.distillation is not None
    config = replace(
        config,
        distillation=replace(
            config.distillation,
            teacher=TeacherConfig(
                config_path=config.distillation.teacher.config_path,
                checkpoint_path=str(checkpoint),
            ),
        ),
    )
    monkeypatch.setattr("src.baselines.load_model", lambda *args, **kwargs: nn.Conv2d(3, 4, 1))

    teacher = load_distillation_teacher(config)

    assert not teacher.training
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    for actual, expected in zip(teacher.state_dict().values(), source.state_dict().values()):
        torch.testing.assert_close(actual, expected)

    wrong_mask_variant = replace(config, data=replace(config.data, mask_variant="ground"))
    with pytest.raises(ValueError, match="mask variants"):
        load_distillation_teacher(wrong_mask_variant)

    wrong_manifest_path = tmp_path / "wrong_split.json"
    reference_manifest = Path(
        "experiments/baselines_benchmark/split_seed42_aligned.json"
    )
    wrong_manifest_path.write_text(
        reference_manifest.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    wrong_manifest = replace(
        config,
        data=replace(
            config.data,
            split_manifest=str(wrong_manifest_path),
        ),
    )
    with pytest.raises(ValueError, match="split manifests"):
        load_distillation_teacher(wrong_manifest)


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
