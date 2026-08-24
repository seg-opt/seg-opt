from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataloaders import make_dataloaders
from src.data.dataset import (
    LunarDataset,
    MaskVariant,
    SplitManifest,
    load_datasets,
    load_split_manifest,
)
from src.distillation.bpkd import BPKDLoss, PixelWiseKDLoss
from src.data.processors import SegmentationImageProcessor
from src.data.transforms import EvalSegmentationTransform, TrainSegmentationTransform
from src.models.loaders import load_model
from src.models.segmentation import SemanticSegmenter
from src.train.losses import SegmentationLoss
from src.train.trainer import BaselineModule, DistillationModule
from src.utils.utils import BaselineConfig, DistillationConfig, load_config


@dataclass(frozen=True)
class BaselineComponents:
    model: SemanticSegmenter
    module: BaselineModule
    train_dataset: LunarDataset
    validation_dataset: LunarDataset
    test_dataset: LunarDataset
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    split_manifest: SplitManifest


def build_baseline(
    config: BaselineConfig,
    *,
    sample_limits: tuple[int | None, int | None, int | None] = (None, None, None),
    mask_variant: MaskVariant | None = None,
    include_distillation: bool = True,
) -> BaselineComponents:
    """Build the common model, data, loss, and Lightning module for one baseline."""
    selected_mask_variant = (
        config.data.mask_variant if mask_variant is None else mask_variant
    )
    processor = SegmentationImageProcessor()
    train_transform = TrainSegmentationTransform(
        height=config.data.image_height,
        width=config.data.image_width,
        **asdict(config.data.augmentation),
    )
    eval_transform = EvalSegmentationTransform(
        height=config.data.image_height,
        width=config.data.image_width,
    )
    train_dataset, validation_dataset, test_dataset = load_datasets(
        processor,
        dataset_root=config.data.dataset_root,
        split_manifest=config.data.split_manifest,
        mask_variant=selected_mask_variant,
        train_transform=train_transform,
        eval_transform=eval_transform,
        max_train_samples=sample_limits[0],
        max_val_samples=sample_limits[1],
        max_test_samples=sample_limits[2],
    )
    manifest = train_dataset.split_manifest
    if manifest is None:
        raise RuntimeError("canonical datasets must retain their split manifest")

    loaders = make_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        seed=config.training.seed,
    )
    model = load_model(
        config.model.name,
        model_id=config.model.model_id,
        pretrained=config.model.pretrained,
        class_names=config.data.class_names,
        ignore_index=config.loss.ignore_index,
    )
    # Transformers returns pretrained task models in evaluation mode. Baseline
    # training starts from train mode; DINOv3's override keeps its frozen
    # backbone in evaluation mode while leaving the probe trainable.
    model.train()
    criterion = SegmentationLoss(
        num_classes=config.num_classes,
        class_weights=None,
        cross_entropy_weight=config.loss.cross_entropy_weight,
        dice_weight=config.loss.dice_weight,
        ignore_index=config.loss.ignore_index,
    )
    module_args = {
        "head_learning_rate": config.training.learning_rate,
        "backbone_learning_rate": config.training.backbone_learning_rate,
        "weight_decay": config.training.weight_decay,
        "warmup_fraction": config.training.warmup_fraction,
        "class_names": config.data.class_names,
        "ignore_index": config.loss.ignore_index,
        "trimap_kernel_size": config.metrics.trimap_kernel_size,
    }
    if config.distillation is not None and include_distillation:
        teacher = load_distillation_teacher(config)
        module: BaselineModule = DistillationModule(
            model,
            criterion,
            teacher=teacher,
            distillation_loss=build_distillation_loss(
                config.distillation,
                num_classes=config.num_classes,
                ignore_index=config.loss.ignore_index,
            ),
            **module_args,
        )
    else:
        module = BaselineModule(model, criterion, **module_args)
    return BaselineComponents(
        model=model,
        module=module,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=loaders[0],
        validation_loader=loaders[1],
        test_loader=loaders[2],
        split_manifest=manifest,
    )


def build_distillation_loss(
    config: DistillationConfig,
    *,
    num_classes: int,
    ignore_index: int,
) -> BPKDLoss | PixelWiseKDLoss:
    if config.method == "vanilla":
        return PixelWiseKDLoss(
            num_classes=num_classes,
            weight=config.vanilla_weight,
            temperature=config.vanilla_temperature,
            spatial_stride=config.spatial_stride,
            ignore_index=ignore_index,
        )
    return BPKDLoss(
        num_classes=num_classes,
        body_weight=config.body_weight,
        edge_weight=config.edge_weight,
        edge_alpha=config.edge_alpha,
        edge_kernel_size=config.edge_kernel_size,
        spatial_stride=config.spatial_stride,
        body_temperature=config.body_temperature,
        edge_temperature=config.edge_temperature,
        prm=config.prm,
        pom=config.pom,
        ignore_index=ignore_index,
    )


def _teacher_model_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("teacher checkpoint does not contain a state_dict mapping")
    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    if not model_state:
        raise ValueError("teacher checkpoint contains no model.* weights")
    if not all(isinstance(value, torch.Tensor) for value in model_state.values()):
        raise ValueError("teacher checkpoint model weights must be tensors")
    return model_state


def load_distillation_teacher(config: BaselineConfig) -> SemanticSegmenter:
    """Restore a supervised canonical teacher and validate its data provenance."""
    if config.distillation is None:
        raise ValueError("a distillation configuration is required to load a teacher")
    teacher_reference = config.distillation.teacher
    teacher_config_path = Path(teacher_reference.config_path)
    checkpoint_path = Path(teacher_reference.checkpoint_path)
    if not teacher_config_path.is_file():
        raise FileNotFoundError(f"teacher config not found: {teacher_config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {checkpoint_path}")

    teacher_config = load_config(teacher_config_path)
    if teacher_config.distillation is not None:
        raise ValueError("a teacher must be trained without a nested distillation section")
    if teacher_config.data.class_names != config.data.class_names:
        raise ValueError("teacher and student class names must match exactly")
    if teacher_config.data.mask_variant != config.data.mask_variant:
        raise ValueError("teacher and student mask variants must match")
    teacher_manifest = load_split_manifest(teacher_config.data.split_manifest)
    student_manifest = load_split_manifest(config.data.split_manifest)
    if teacher_manifest.sha256 != student_manifest.sha256:
        raise ValueError("teacher and student split manifests must match")

    teacher = load_model(
        teacher_config.model.name,
        model_id=teacher_config.model.model_id,
        pretrained=teacher_config.model.pretrained,
        class_names=teacher_config.data.class_names,
        ignore_index=teacher_config.loss.ignore_index,
    )
    try:
        teacher.load_state_dict(_teacher_model_state(checkpoint_path), strict=True)
    except RuntimeError as error:
        raise ValueError(
            "teacher checkpoint is incompatible with its configured model"
        ) from error
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher
