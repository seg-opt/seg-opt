from dataclasses import dataclass
from typing import Literal

import yaml


def load_yaml(path: str) -> dict[str, any]:
    with open(path, 'r') as file:
        return yaml.safe_load(file)


@dataclass
class TeacherConfig:
    name: Literal["mask2former", "dinov3", "sam3"]
    model_id: str
    freeze: bool = True


@dataclass
class StudentConfig:
    backbone: Literal["resnet", "efficientnet", "mobilenet"]
    backbone_id: str
    decoder: Literal["unet", "deeplabv3", "fpn"] = "unet"
    pretrained: bool = True


@dataclass
class ModelConfig:
    num_classes: int
    teacher: TeacherConfig
    student: StudentConfig


@dataclass
class DistillationConfig:
    temperature: float
    weight: float
    hard_label_weight: float


@dataclass
class DataConfig:
    image_dir: str
    mask_dir: str
    class_names: list[str]
    image_size: int
    train_frac: float
    val_frac: float
    test_frac: float


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    precision: Literal["fp32", "fp16", "bf16"]
    logging_steps: int
    eval_steps: int
    save_steps: int
    seed: int
    output_dir: str


@dataclass
class ExperimentConfig:
    model: ModelConfig
    distillation: DistillationConfig
    data: DataConfig
    training: TrainingConfig


def load_config(path: str) -> ExperimentConfig:
    raw = load_yaml(path)
    return ExperimentConfig(
        model=ModelConfig(
            num_classes=raw["model"]["num_classes"],
            teacher=TeacherConfig(**raw["model"]["teacher"]),
            student=StudentConfig(**raw["model"]["student"]),
        ),
        distillation=DistillationConfig(**raw["distillation"]),
        data=DataConfig(**raw["data"]),
        training=TrainingConfig(**raw["training"]),
    )
