from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

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
class BaselineConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    loss: LossConfig = field(default_factory=LossConfig)

    @property
    def num_classes(self) -> int:
        return len(self.data.class_names)


def load_config(path: str | Path) -> BaselineConfig:
    raw = load_yaml(path)
    data_raw = dict(raw["data"])
    for key in ("dataset_root", "split_manifest"):
        value = Path(data_raw[key]).expanduser()
        resolved = value if value.is_absolute() else (PROJECT_ROOT / value).resolve()
        data_raw[key] = str(resolved)
    data_raw["augmentation"] = AugmentationConfig(
        **data_raw.pop("augmentation", {})
    )
    if "class_names" in data_raw:
        data_raw["class_names"] = tuple(data_raw["class_names"])
    training_raw = dict(raw["training"])
    output_dir = Path(training_raw["output_dir"]).expanduser()
    training_raw["output_dir"] = str(
        output_dir if output_dir.is_absolute() else (PROJECT_ROOT / output_dir).resolve()
    )
    return BaselineConfig(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**data_raw),
        loss=LossConfig(**raw.get("loss", {})),
        training=TrainingConfig(**training_raw),
    )


def resolved_config(config: BaselineConfig) -> dict[str, Any]:
    """Return a YAML/JSON-safe representation of a validated configuration."""
    return asdict(config)


def write_resolved_config(config: BaselineConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(resolved_config(config), file, sort_keys=False)
