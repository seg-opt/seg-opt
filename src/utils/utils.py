from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml


BaselineName = Literal[
    "fast_scnn",
    "resnet34_unet",
    "segformer_b0",
    "dinov3_vitl16",
    "mask2former_swinl",
]

BASELINE_NAMES = {
    "fast_scnn",
    "resnet34_unet",
    "segformer_b0",
    "dinov3_vitl16",
    "mask2former_swinl",
}
CLASS_NAMES = ("background", "sky", "small_rock", "large_rock")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        contents = yaml.safe_load(file)
    if not isinstance(contents, dict):
        raise ValueError("configuration must contain a YAML mapping")
    return contents


@dataclass(frozen=True)
class ModelConfig:
    name: BaselineName
    model_id: str | None = None
    pretrained: bool = True
    freeze_backbone: bool = False

    def __post_init__(self) -> None:
        if self.name not in BASELINE_NAMES:
            raise ValueError(
                f"unsupported baseline {self.name!r}; expected one of "
                f"{sorted(BASELINE_NAMES)}"
            )
        if self.name == "fast_scnn" and self.pretrained:
            raise ValueError("Fast-SCNN has no pretrained weights")
        if self.name == "dinov3_vitl16" and not self.freeze_backbone:
            raise ValueError("the canonical DINOv3 baseline is a frozen linear probe")


@dataclass(frozen=True)
class AugmentationConfig:
    flip_probability: float = 0.5
    brightness: float = 0.15
    contrast: float = 0.15
    gamma_min: float = 0.85
    gamma_max: float = 1.15

    def __post_init__(self) -> None:
        if not 0 <= self.flip_probability <= 1:
            raise ValueError("flip_probability must be between zero and one")
        if self.brightness < 0 or self.contrast < 0:
            raise ValueError("brightness and contrast must be non-negative")
        if not 0 < self.gamma_min <= self.gamma_max:
            raise ValueError("gamma range must be positive and ordered")


@dataclass(frozen=True)
class DataConfig:
    dataset_root: str
    split_manifest: str
    class_names: tuple[str, ...] = CLASS_NAMES
    image_height: int = 512
    image_width: int = 768
    num_workers: int = 8
    mask_variant: Literal["ground", "clean"] = "ground"
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def __post_init__(self) -> None:
        if tuple(self.class_names) != CLASS_NAMES:
            raise ValueError(f"class_names must be ordered as {list(CLASS_NAMES)}")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image dimensions must be positive")
        if self.image_height % 32 or self.image_width % 32:
            raise ValueError("image dimensions must be divisible by 32")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.mask_variant not in {"ground", "clean"}:
            raise ValueError("mask_variant must be 'ground' or 'clean'")


@dataclass(frozen=True)
class LossConfig:
    cross_entropy_weight: float = 1.0
    dice_weight: float = 0.5
    ignore_index: int = 255

    def __post_init__(self) -> None:
        if self.cross_entropy_weight < 0 or self.dice_weight < 0:
            raise ValueError("loss weights cannot be negative")
        if self.cross_entropy_weight == 0 and self.dice_weight == 0:
            raise ValueError("at least one dense loss term must be enabled")


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str
    batch_size: int
    accumulate_grad_batches: int
    learning_rate: float
    backbone_learning_rate: float | None = None
    epochs: int = 20
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    precision: Literal["fp32", "fp16", "bf16"] = "bf16"
    logging_steps: int = 20
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.accumulate_grad_batches <= 0:
            raise ValueError("accumulate_grad_batches must be positive")
        if self.batch_size * self.accumulate_grad_batches != 16:
            raise ValueError("canonical effective batch size must equal 16")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.backbone_learning_rate is not None and self.backbone_learning_rate <= 0:
            raise ValueError("backbone_learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.seed != 42:
            raise ValueError("canonical baselines use seed 42")


@dataclass(frozen=True)
class MetricsConfig:
    """Optional boundary diagnostics in addition to standard segmentation metrics."""

    trimap_kernel_size: int | None = None

    def __post_init__(self) -> None:
        if self.trimap_kernel_size is not None and (
            self.trimap_kernel_size <= 0 or self.trimap_kernel_size % 2 == 0
        ):
            raise ValueError("trimap_kernel_size must be a positive odd integer")


@dataclass(frozen=True)
class TeacherConfig:
    """A supervised canonical teacher definition and its trained checkpoint."""

    config_path: str
    checkpoint_path: str

    def __post_init__(self) -> None:
        if not self.config_path:
            raise ValueError("teacher config_path cannot be empty")
        if not self.checkpoint_path:
            raise ValueError("teacher checkpoint_path cannot be empty")


@dataclass(frozen=True)
class DistillationConfig:
    """Validated settings shared by vanilla KD and BPKD experiments."""

    method: Literal["vanilla", "bpkd"]
    teacher: TeacherConfig
    spatial_stride: int = 8
    vanilla_weight: float = 1.0
    vanilla_temperature: float = 1.0
    body_weight: float = 20.0
    edge_weight: float = 50.0
    edge_alpha: float | tuple[float, ...] = 2.0
    edge_kernel_size: int = 7
    body_temperature: float = 1.0
    edge_temperature: float = 1.0
    prm: bool = True
    pom: bool = True

    def __post_init__(self) -> None:
        if self.method not in {"vanilla", "bpkd"}:
            raise ValueError("distillation method must be 'vanilla' or 'bpkd'")
        if self.spatial_stride <= 0:
            raise ValueError("distillation spatial_stride must be positive")
        if self.vanilla_weight < 0 or self.body_weight < 0 or self.edge_weight < 0:
            raise ValueError("distillation weights must be non-negative")
        if min(
            self.vanilla_temperature,
            self.body_temperature,
            self.edge_temperature,
        ) <= 0:
            raise ValueError("distillation temperatures must be positive")
        if self.edge_kernel_size <= 0 or self.edge_kernel_size % 2 == 0:
            raise ValueError("edge_kernel_size must be a positive odd integer")
        alpha = (
            (float(self.edge_alpha),)
            if isinstance(self.edge_alpha, (int, float))
            else tuple(float(value) for value in self.edge_alpha)
        )
        if not alpha or any(value < 0 for value in alpha):
            raise ValueError("edge_alpha values must be non-negative")
        object.__setattr__(
            self,
            "edge_alpha",
            alpha[0] if len(alpha) == 1 else alpha,
        )
        if self.method == "vanilla" and self.vanilla_weight == 0:
            raise ValueError("vanilla distillation requires vanilla_weight > 0")
        if self.method == "bpkd":
            if self.body_weight == 0 and self.edge_weight == 0:
                raise ValueError("BPKD requires a non-zero body_weight or edge_weight")


@dataclass(frozen=True)
class BaselineConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    loss: LossConfig = field(default_factory=LossConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    distillation: DistillationConfig | None = None

    def __post_init__(self) -> None:
        if self.distillation is not None and isinstance(
            self.distillation.edge_alpha,
            tuple,
        ) and len(self.distillation.edge_alpha) != self.num_classes:
            raise ValueError("edge_alpha must have one value per configured class")

    @property
    def num_classes(self) -> int:
        return len(self.data.class_names)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return dict(value)


def _reject_unknown_fields(
    values: Mapping[str, Any],
    allowed: set[str],
    name: str,
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unsupported fields: {unknown}")


def _project_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def _load_distillation_config(value: Any) -> DistillationConfig | None:
    if value is None:
        return None
    raw = _mapping(value, "distillation")
    _reject_unknown_fields(
        raw,
        {
            "method",
            "teacher",
            "spatial_stride",
            "vanilla_weight",
            "vanilla_temperature",
            "body_weight",
            "edge_weight",
            "edge_alpha",
            "edge_kernel_size",
            "body_temperature",
            "edge_temperature",
            "prm",
            "pom",
        },
        "distillation",
    )
    if "method" not in raw or "teacher" not in raw:
        raise ValueError("distillation requires method and teacher sections")
    teacher_raw = _mapping(raw.pop("teacher"), "distillation.teacher")
    _reject_unknown_fields(
        teacher_raw,
        {"config_path", "checkpoint_path"},
        "distillation.teacher",
    )
    for key in ("config_path", "checkpoint_path"):
        if key not in teacher_raw:
            raise ValueError(f"distillation.teacher is missing required {key!r}")
        teacher_raw[key] = _project_path(teacher_raw[key])
    if isinstance(raw.get("edge_alpha"), list):
        raw["edge_alpha"] = tuple(raw["edge_alpha"])
    return DistillationConfig(
        teacher=TeacherConfig(**teacher_raw),
        **raw,
    )


def load_config(path: str | Path) -> BaselineConfig:
    raw = load_yaml(path)
    _reject_unknown_fields(
        raw,
        {"model", "data", "training", "loss", "metrics", "distillation"},
        "configuration",
    )
    for required in ("model", "data", "training"):
        if required not in raw:
            raise ValueError(f"configuration is missing required {required!r} section")

    model_raw = _mapping(raw["model"], "model")
    _reject_unknown_fields(
        model_raw,
        {"name", "model_id", "pretrained", "freeze_backbone"},
        "model",
    )
    data_raw = _mapping(raw["data"], "data")
    _reject_unknown_fields(
        data_raw,
        {
            "dataset_root",
            "split_manifest",
            "class_names",
            "image_height",
            "image_width",
            "num_workers",
            "mask_variant",
            "augmentation",
        },
        "data",
    )
    for key in ("dataset_root", "split_manifest"):
        if key not in data_raw:
            raise ValueError(f"data section is missing required {key!r}")
        data_raw[key] = _project_path(data_raw[key])
    augmentation_raw = _mapping(data_raw.pop("augmentation", {}), "data.augmentation")
    _reject_unknown_fields(
        augmentation_raw,
        {"flip_probability", "brightness", "contrast", "gamma_min", "gamma_max"},
        "data.augmentation",
    )
    data_raw["augmentation"] = AugmentationConfig(**augmentation_raw)
    if "class_names" in data_raw:
        data_raw["class_names"] = tuple(data_raw["class_names"])
    training_raw = _mapping(raw["training"], "training")
    _reject_unknown_fields(
        training_raw,
        {
            "output_dir",
            "batch_size",
            "accumulate_grad_batches",
            "learning_rate",
            "backbone_learning_rate",
            "epochs",
            "weight_decay",
            "warmup_fraction",
            "precision",
            "logging_steps",
            "seed",
        },
        "training",
    )
    if "output_dir" not in training_raw:
        raise ValueError("training section is missing required 'output_dir'")
    training_raw["output_dir"] = _project_path(training_raw["output_dir"])
    loss_raw = _mapping(raw.get("loss", {}), "loss")
    _reject_unknown_fields(
        loss_raw,
        {"cross_entropy_weight", "dice_weight", "ignore_index"},
        "loss",
    )
    metrics_raw = _mapping(raw.get("metrics", {}), "metrics")
    _reject_unknown_fields(metrics_raw, {"trimap_kernel_size"}, "metrics")
    distillation = _load_distillation_config(raw.get("distillation"))
    return BaselineConfig(
        model=ModelConfig(**model_raw),
        data=DataConfig(**data_raw),
        loss=LossConfig(**loss_raw),
        training=TrainingConfig(**training_raw),
        metrics=MetricsConfig(**metrics_raw),
        distillation=distillation,
    )


def resolved_config(config: BaselineConfig) -> dict[str, Any]:
    """Return a YAML/JSON-safe representation of a validated configuration."""
    return asdict(config)


def write_resolved_config(config: BaselineConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(resolved_config(config), file, sort_keys=False)
