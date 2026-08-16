from dataclasses import asdict, dataclass
from torch.utils.data import DataLoader

from src.data.dataloaders import make_dataloaders
from src.data.dataset import LunarDataset, SplitManifest, load_datasets
from src.data.processors import SegmentationImageProcessor
from src.data.transforms import EvalSegmentationTransform, TrainSegmentationTransform
from src.models.loaders import load_model
from src.models.segmentation import SemanticSegmenter
from src.train.losses import SegmentationLoss
from src.train.trainer import BaselineModule
from src.utils.utils import BaselineConfig


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
) -> BaselineComponents:
    """Build the common model, data, loss, and Lightning module for one baseline."""
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
    module = BaselineModule(
        model,
        criterion,
        head_learning_rate=config.training.learning_rate,
        backbone_learning_rate=config.training.backbone_learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_fraction=config.training.warmup_fraction,
        class_names=config.data.class_names,
        ignore_index=config.loss.ignore_index,
    )
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
