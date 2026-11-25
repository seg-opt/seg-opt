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
    TrainingConfig,
    load_config,
    write_resolved_config,
)


from tests.unit.test_submit_hydra import load_hydra_config


def test_baselines_benchmark_configs_match_the_comparison_protocol(tmp_path):
    students = (
        "dinov3_vitl16", "fast_scnn", "mask2former_swinl", "resnet101_unet",
        "resnet34_unet", "segformer_b0", "segformer_b2", "segformer_b4",
    )
    configs = [
        load_hydra_config(tmp_path, "dataset=lunar", "method=supervised", "teacher=none", f"student={student}")
        for student in students
    ]

    assert {config.model.name for config in configs} <= BASELINE_NAMES
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
    for path in tmp_path.glob("*/config.yaml"):
        contents = path.read_text(encoding="utf-8")
        assert "distillation:" not in contents
        assert "teacher:" not in contents
        assert "student:" not in contents


def test_resolved_configuration_round_trip(tmp_path):
    config = load_hydra_config(
        tmp_path,
        'dataset=lunar',
        'method=supervised',
        'teacher=none',
        'student=fast_scnn',
        'training.output_dir=results/baselines_benchmark/fast_scnn/seed42',
        'training.batch_size=16',
        'training.accumulate_grad_batches=1',
        'training.learning_rate=0.0005',
        'training.backbone_learning_rate=null',
    )
    resolved_path = tmp_path / "config.yaml"

    write_resolved_config(config, resolved_path)

    assert load_config(resolved_path) == config


@pytest.mark.parametrize("seed", [0, 42, 43, 44, 2**32 - 1])
def test_training_seed_accepts_uint32_without_changing_data_split(seed, tmp_path):
    config = load_hydra_config(
        tmp_path,
        'dataset=lunar',
        'method=supervised',
        'teacher=none',
        'student=fast_scnn',
        'training.output_dir=results/baselines_benchmark/fast_scnn/seed42',
        'training.batch_size=16',
        'training.accumulate_grad_batches=1',
        'training.learning_rate=0.0005',
        'training.backbone_learning_rate=null',
    )
    split_manifest = Path(config.data.split_manifest)
    split_checksum = sha256_file(split_manifest)

    updated = replace(config, training=replace(config.training, seed=seed))
    resolved_path = tmp_path / "config.yaml"
    write_resolved_config(updated, resolved_path)
    restored = load_config(resolved_path)

    assert updated.training.seed == seed
    assert config.training.seed == 42
    assert updated.data is config.data
    assert restored == updated
    assert restored.data == config.data
    assert Path(restored.data.split_manifest) == split_manifest
    assert split_manifest.name == "split_seed42_aligned.json"
    assert sha256_file(split_manifest) == split_checksum


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "43"])
def test_training_seed_rejects_invalid_values(seed, tmp_path):
    config = load_hydra_config(
        tmp_path,
        'dataset=lunar',
        'method=supervised',
        'teacher=none',
        'student=fast_scnn',
        'training.output_dir=results/baselines_benchmark/fast_scnn/seed42',
        'training.batch_size=16',
        'training.accumulate_grad_batches=1',
        'training.learning_rate=0.0005',
        'training.backbone_learning_rate=null',
    )

    with pytest.raises(ValueError, match="seed"):
        replace(config.training, seed=seed)


def test_training_defaults_to_seed42_and_80_epochs(tmp_path):
    training = TrainingConfig(
        output_dir=str(tmp_path),
        batch_size=1,
        accumulate_grad_batches=16,
        learning_rate=5.0e-4,
    )

    assert training.seed == 42
    assert training.epochs == 80


def test_resnet101_teacher_preserves_resnet34_training_protocol(tmp_path):
    larger = load_hydra_config(
        tmp_path,
        'dataset=lunar',
        'method=supervised',
        'teacher=none',
        'student=resnet101_unet',
        'training.output_dir=results/baselines_benchmark/resnet101_unet/seed42',
        'training.learning_rate=0.0003',
        'training.backbone_learning_rate=3e-05',
    )
    previous = load_hydra_config(
        tmp_path,
        'dataset=lunar',
        'method=supervised',
        'teacher=none',
        'student=resnet34_unet',
        'training.output_dir=results/baselines_benchmark/resnet34_unet/seed42',
        'training.learning_rate=0.0003',
        'training.backbone_learning_rate=3e-05',
    )

    assert larger.model.model_id == "resnet101"
    assert larger.model.pretrained
    assert larger.data == previous.data
    assert larger.loss == previous.loss
    assert larger.metrics == previous.metrics
    assert replace(
        larger.training, output_dir=previous.training.output_dir
    ) == previous.training
    assert larger.training.output_dir != previous.training.output_dir


@pytest.mark.parametrize(
    ("method", "teacher"),
    [("supervised", "none"), ("vanilla", "benchmark_segformer_b4"),
     ("bpkd", "bpkd_mask2former_swinl"), ("sard", "benchmark_segformer_b4"),
     ("vanilla", "strong_dinov3_mask2former")],
)
def test_canonical_full_run_configs_use_80_epochs(method, teacher, tmp_path):
    config = load_hydra_config(
        tmp_path, "dataset=lunar", f"method={method}", f"teacher={teacher}", "student=fast_scnn",
    )
    assert config.training.epochs == 80


@pytest.mark.parametrize("student_name", ["fast_scnn", "segformer_b0"])
def test_strong_teacher_configs_do_not_reuse_the_unverified_historical_checkpoint(student_name, tmp_path):
    teacher = load_hydra_config(
        tmp_path,
        'dataset=lunar',
        'method=supervised',
        'teacher=none',
        'student=dinov3_vitl16_mask2former',
        'training.output_dir=results/strong_teacher_distillation/teacher_dinov3_mask2former/seed42',
        'training.batch_size=1',
        'training.accumulate_grad_batches=16',
        'training.learning_rate=0.0001',
        'training.backbone_learning_rate=null',
    )
    student = load_hydra_config(
        tmp_path, "dataset=lunar", "method=vanilla", "teacher=strong_dinov3_mask2former",
        f"student={student_name}",
    )

    assert teacher.training.epochs == student.training.epochs == 80
    assert Path(teacher.training.output_dir).parts[-3:] == (
        "strong_teacher_distillation", "teacher_dinov3_mask2former", "seed42"
    )
    assert student.distillation is not None
    assert load_config(student.distillation.teacher.config_path) == teacher
    assert Path(student.distillation.teacher.checkpoint_path) == (
        Path(teacher.training.output_dir) / "checkpoints" / "best.ckpt"
    )


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
    config_path: teacher.yaml
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


@pytest.mark.parametrize(
    ("student_name", "model_id", "pretrained"),
    [("fast_scnn", None, False), ("segformer_b0", "nvidia/mit-b0", True)],
)
def test_segformer_b4_logit_distillation_configs(student_name, model_id, pretrained, tmp_path):
    config = load_hydra_config(
        tmp_path, "dataset=lunar", "method=vanilla", "teacher=benchmark_segformer_b4",
        f"student={student_name}",
        f"training.output_dir=results/logit_distillation/{student_name}_from_segformer_b4/seed42",
    )

    assert config.model.name == student_name
    assert config.model.model_id == model_id
    assert config.model.pretrained == pretrained
    assert config.data.mask_variant == "clean"
    assert config.training.batch_size == 8
    assert config.training.accumulate_grad_batches == 2
    assert config.training.batch_size * config.training.accumulate_grad_batches == 16
    assert Path(config.training.output_dir).parts[-3:] == (
        "logit_distillation",
        f"{student_name}_from_segformer_b4",
        "seed42",
    )
    assert config.distillation is not None
    assert config.distillation.method == "vanilla"
    assert config.distillation.spatial_stride == 8
    assert config.distillation.vanilla_weight == 1.0
    assert config.distillation.vanilla_temperature == 1.0
    assert load_config(config.distillation.teacher.config_path).model.name == "segformer_b4"
    assert Path(config.distillation.teacher.checkpoint_path).parts[-6:] == (
        "results",
        "baselines_benchmark",
        "segformer_b4",
        "seed42",
        "checkpoints",
        "best.ckpt",
    )


@pytest.mark.parametrize("teacher_name", ["resnet34_unet", "resnet101_unet"])
def test_cnn_teacher_to_fast_scnn_logit_distillation_config(teacher_name: str, tmp_path) -> None:
    config = load_hydra_config(
        tmp_path, "dataset=lunar", "method=vanilla", f"teacher=benchmark_{teacher_name}", "student=fast_scnn",
        f"training.output_dir=results/logit_distillation/fast_scnn_from_{teacher_name}/seed42",
        "training.batch_size=16", "training.accumulate_grad_batches=1",
        "training.learning_rate=0.0005", "training.backbone_learning_rate=null",
    )
    baseline = load_hydra_config(
        tmp_path,
        'dataset=lunar',
        'method=supervised',
        'teacher=none',
        'student=fast_scnn',
        'training.output_dir=results/baselines_benchmark/fast_scnn/seed42',
        'training.batch_size=16',
        'training.accumulate_grad_batches=1',
        'training.learning_rate=0.0005',
        'training.backbone_learning_rate=null',
    )
    teacher = load_hydra_config(
        tmp_path, "dataset=lunar", "method=supervised", "teacher=none", f"student={teacher_name}",
        f"training.output_dir=results/baselines_benchmark/{teacher_name}/seed42",
        "training.learning_rate=0.0003", "training.backbone_learning_rate=3e-05",
    )

    assert config.model == baseline.model
    assert config.data == baseline.data == teacher.data
    assert config.loss == baseline.loss == teacher.loss
    assert config.metrics == baseline.metrics == teacher.metrics
    assert replace(
        config.training, output_dir=baseline.training.output_dir
    ) == baseline.training
    assert Path(config.training.output_dir).parts[-3:] == (
        "logit_distillation",
        f"fast_scnn_from_{teacher_name}",
        "seed42",
    )
    assert config.distillation is not None
    assert config.distillation.method == "vanilla"
    assert config.distillation.spatial_stride == 8
    assert config.distillation.vanilla_weight == 1.0
    assert config.distillation.vanilla_temperature == 1.0
    assert teacher.model.name == teacher_name
    assert teacher.distillation is None
    assert load_config(config.distillation.teacher.config_path) == teacher
    assert Path(config.distillation.teacher.checkpoint_path) == (
        Path(teacher.training.output_dir) / "checkpoints" / "best.ckpt"
    )


@pytest.mark.parametrize(
    ("variant", "body_weight", "edge_weight"),
    [("body_only", 20.0, 0.0), ("edge_only", 0.0, 50.0), ("bpkd_w7", 20.0, 50.0)],
)
def test_resnet34_bpkd_configs_match_vanilla_protocol(tmp_path, variant: str, body_weight: float, edge_weight: float
) -> None:
    config = load_hydra_config(
        tmp_path, "dataset=lunar", "method=bpkd", "teacher=benchmark_resnet34_unet", "student=fast_scnn",
        f"training.output_dir=results/bpkd/fast_scnn_from_resnet34_unet/{variant}/seed42",
        "training.batch_size=16", "training.accumulate_grad_batches=1",
        "training.learning_rate=0.0005", "training.backbone_learning_rate=null",
        *({"body_only": ("distillation.edge_weight=0.0",),
           "edge_only": ("distillation.body_weight=0.0",), "bpkd_w7": ()}[variant]),
    )
    vanilla = load_hydra_config(
        tmp_path,
        'dataset=lunar',
        'method=vanilla',
        'teacher=benchmark_resnet34_unet',
        'student=fast_scnn',
        'training.output_dir=results/logit_distillation/fast_scnn_from_resnet34_unet/seed42',
        'training.batch_size=16',
        'training.accumulate_grad_batches=1',
        'training.learning_rate=0.0005',
        'training.backbone_learning_rate=null',
    )

    assert config.model == vanilla.model
    assert config.data == vanilla.data
    assert config.loss == vanilla.loss
    assert config.metrics == vanilla.metrics
    assert replace(
        config.training, output_dir=vanilla.training.output_dir
    ) == vanilla.training
    assert Path(config.training.output_dir).parts[-4:] == (
        "bpkd", "fast_scnn_from_resnet34_unet", variant, "seed42"
    )
    assert config.distillation is not None
    assert vanilla.distillation is not None
    assert config.distillation.teacher.checkpoint_path == vanilla.distillation.teacher.checkpoint_path
    assert load_config(config.distillation.teacher.config_path) == load_config(vanilla.distillation.teacher.config_path)
    assert config.distillation.method == "bpkd"
    assert config.distillation.spatial_stride == vanilla.distillation.spatial_stride
    assert config.distillation.body_weight == body_weight
    assert config.distillation.edge_weight == edge_weight
    assert config.distillation.body_temperature == 1.0
    assert config.distillation.edge_temperature == 1.0
    assert config.distillation.edge_alpha == 2.0
    assert config.distillation.edge_kernel_size == 7
    assert config.distillation.prm
    assert config.distillation.pom


@pytest.mark.parametrize(
    ("config_name", "weight", "temperature"),
    [
        ("segformer_b0_from_segformer_b4_s4_w1_t1", 1.0, 1.0),
        ("segformer_b0_from_segformer_b4_s4_w1_t2", 1.0, 2.0),
        ("segformer_b0_from_segformer_b4_s4_w4_t1", 4.0, 1.0),
        ("segformer_b0_from_segformer_b4_s4_w4_t2", 4.0, 2.0),
    ],
)
def test_segformer_b4_to_b0_stride4_vanilla_kd_sweep(tmp_path, config_name, weight, temperature
):
    config = load_hydra_config(
        tmp_path,
        'dataset=lunar',
        'method=vanilla',
        'teacher=benchmark_segformer_b4',
        'student=segformer_b0',
        'training.output_dir=results/logit_distillation/segformer_b0_from_segformer_b4/seed42',
        'distillation.spatial_stride=4',
        f'distillation.vanilla_weight={weight}',
        f'distillation.vanilla_temperature={temperature}',
        f'training.output_dir=results/logit_distillation/{config_name}/seed42',
    )

    assert config.model.name == "segformer_b0"
    assert config.training.batch_size * config.training.accumulate_grad_batches == 16
    assert Path(config.training.output_dir).parts[-3:] == (
        "logit_distillation",
        config_name,
        "seed42",
    )
    assert config.distillation is not None
    assert config.distillation.method == "vanilla"
    assert config.distillation.spatial_stride == 4
    assert config.distillation.vanilla_weight == weight
    assert config.distillation.vanilla_temperature == temperature
    assert load_config(config.distillation.teacher.config_path).model.name == "segformer_b4"


def test_mask2former_bpkd_configs_are_clean_and_match_the_declared_matrix(tmp_path):
    variants = [
        ("fast_scnn", "supervised", "none", ()),
        ("fast_scnn", "vanilla", "bpkd_mask2former_swinl", ()),
        ("fast_scnn", "bpkd", "bpkd_mask2former_swinl", ("distillation.edge_weight=0.0",)),
        ("fast_scnn", "bpkd", "bpkd_mask2former_swinl", ("distillation.body_weight=0.0",)),
        *(("fast_scnn", "bpkd", "bpkd_mask2former_swinl", (f"distillation.edge_kernel_size={width}",))
          for width in (3, 5, 7, 9)),
        ("mask2former_swinl", "supervised", "none", ()),
        ("mask2former_swinl", "vanilla", "bpkd_mask2former_swinl", ()),
        ("mask2former_swinl", "bpkd", "bpkd_mask2former_swinl", ()),
    ]
    configs = []
    for student, method, teacher, overrides in variants:
        batch_size = 2 if student == "fast_scnn" else 1
        configs.append(load_hydra_config(
            tmp_path, "dataset=lunar", f"method={method}", f"teacher={teacher}", f"student={student}",
            f"training.batch_size={batch_size}", f"training.accumulate_grad_batches={16 // batch_size}",
            *overrides,
        ))

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
    config = load_hydra_config(
        tmp_path,
        'dataset=lunar',
        'method=bpkd',
        'teacher=bpkd_mask2former_swinl',
        'student=fast_scnn',
        'training.output_dir=results/bpkd/fast_scnn_clean/bpkd_w7/seed42',
        'training.batch_size=2',
        'training.accumulate_grad_batches=8',
        'training.learning_rate=0.0005',
        'training.backbone_learning_rate=null',
    )
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


def test_standalone_lusnar_configuration(tmp_path):
    config = load_hydra_config(
        tmp_path,
        'dataset=lusnar',
        'method=supervised',
        'teacher=none',
        'student=segformer_b0',
        'training.output_dir=results/lusnar/segformer_b0/seed42',
        'training.epochs=20',
    )
    assert config.data.dataset == "lusnar"
    assert config.data.class_names == ("regolith", "crater", "rock", "mountain", "sky")
    assert config.num_classes == 5
    assert config.data.mask_variant == "ground"
    assert (config.data.image_height, config.data.image_width) == (512, 512)
    assert config.training.seed == 42
    assert config.training.epochs == 20
    assert config.distillation is None
    assert Path(config.training.output_dir).parts[-4:] == ("results", "lusnar", "segformer_b0", "seed42")
    resolved_path = tmp_path / "lusnar.yaml"
    write_resolved_config(config, resolved_path)
    assert load_config(resolved_path) == config
    with pytest.raises(ValueError, match="class_names must be ordered"):
        replace(config.data, class_names=tuple(reversed(config.data.class_names)))
    with pytest.raises(ValueError, match="only supports mask_variant"):
        replace(config.data, mask_variant="clean")
