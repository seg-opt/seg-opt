import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.data.transforms import EvalSegmentationTransform, TrainSegmentationTransform


DATASET_ROOT = "../../datasets/artificial_lunar_landscape"
QUALITY_EXCLUDED_ID_FILES = (
    "cam_anomaly_IDs.txt",
    "ground_facing_IDs.txt",
    "mismatch_IDs.txt",
    "shadow_IDs.txt",
)
# Kept as a compatibility name. The canonical baseline excludes only quality failures.
EXCLUDED_ID_FILES = QUALITY_EXCLUDED_ID_FILES
INCLUDED_ID_FILES = ("top200_largerocks_IDs.txt",)
DEFAULT_SPLIT_MANIFEST = (
    Path(__file__).resolve().parents[2] / "experiments" / "baselines" / "split_seed42.json"
)
CANONICAL_SPLIT_SHA256 = "accaa362c675bfdd65b1a44c8c9cd19dba70ba41642fb23c67c544655914d79f"
ALIGNED_SPLIT_MANIFEST = DEFAULT_SPLIT_MANIFEST.with_name(
    "split_seed42_aligned.json"
)
ALIGNED_SPLIT_SHA256 = "e7486b858ee776695b9e5fbdd561ac072e37999d59e801e406c8a299d778c8ab"
BACKGROUND_THRESHOLD = 10

Pair = tuple[Path, Path]
PairedTransform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
MaskVariant = Literal["ground", "clean"]
MASK_VARIANTS = ("ground", "clean")
_FRAME_ID_PATTERN = re.compile(r"\d{4}")


def _frame_id(path: Path) -> str:
    frame_id = path.stem[-4:]
    if _FRAME_ID_PATTERN.fullmatch(frame_id) is None:
        raise ValueError(f"cannot extract a four-digit frame ID from {path.name!r}")
    return frame_id


def _index_by_frame_id(directory: str | Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in Path(directory).glob("*.png"):
        frame_id = _frame_id(path)
        if frame_id in indexed:
            raise ValueError(f"duplicate frame ID {frame_id!r} in {directory}")
        indexed[frame_id] = path
    return indexed


def find_pairs(image_dir: str | Path, mask_dir: str | Path) -> list[Pair]:
    images = _index_by_frame_id(image_dir)
    masks = _index_by_frame_id(mask_dir)
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        raise ValueError(
            "image/mask frame IDs do not match: "
            f"{len(missing_masks)} missing masks, {len(missing_images)} missing images"
        )
    return [(images[frame_id], masks[frame_id]) for frame_id in sorted(images)]


def load_excluded_ids(id_files: Iterable[str | Path]) -> set[str]:
    excluded: set[str] = set()
    for id_file in id_files:
        with Path(id_file).open(encoding="utf-8") as file:
            excluded.update(line.strip() for line in file if line.strip())
    return excluded


def filter_pairs(pairs: Sequence[Pair], excluded_ids: set[str]) -> list[Pair]:
    return [pair for pair in pairs if _frame_id(pair[0]) not in excluded_ids]


def split_pairs(
    pairs: Sequence[Pair],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[Pair], list[Pair], list[Pair]]:
    """Create an ad-hoc split; canonical baselines use the checked-in manifest."""
    fractions = np.asarray([train_frac, val_frac, test_frac], dtype=np.float64)
    if not np.isclose(fractions.sum(), 1.0) or np.any(fractions <= 0):
        raise ValueError("train/validation/test fractions must be positive and sum to 1")
    trainval, test = train_test_split(
        list(pairs),
        test_size=test_frac,
        random_state=seed,
    )
    val_frac_of_trainval = val_frac / (train_frac + val_frac)
    train, validation = train_test_split(
        trainval,
        test_size=val_frac_of_trainval,
        random_state=seed,
    )
    return train, validation, test


def _validate_manifest_ids(split_name: str, values: Any) -> tuple[str, ...]:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"manifest split {split_name!r} must be a list of frame IDs")
    invalid = [value for value in values if _FRAME_ID_PATTERN.fullmatch(value) is None]
    if invalid:
        raise ValueError(
            f"manifest split {split_name!r} contains invalid frame IDs: {invalid[:3]}"
        )
    if len(values) != len(set(values)):
        raise ValueError(f"manifest split {split_name!r} contains duplicate frame IDs")
    return tuple(values)


def _validate_manifest_file_names(field: str, values: Any) -> tuple[str, ...]:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"manifest field {field!r} must be a list of file names")
    for value in values:
        if Path(value).name != value:
            raise ValueError(f"manifest field {field!r} accepts file names, not paths")
    return tuple(values)


@dataclass(frozen=True)
class SplitManifest:
    path: Path
    sha256: str
    seed: int
    excluded_id_files: tuple[str, ...]
    included_id_files: tuple[str, ...]
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    @property
    def val_ids(self) -> tuple[str, ...]:
        return self.validation_ids

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.train_ids + self.validation_ids + self.test_ids

    @property
    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train_ids),
            "validation": len(self.validation_ids),
            "test": len(self.test_ids),
            "total": len(self.all_ids),
        }

    def ids_for(self, split: str) -> tuple[str, ...]:
        if split == "train":
            return self.train_ids
        if split in {"validation", "val"}:
            return self.validation_ids
        if split == "test":
            return self.test_ids
        raise ValueError(f"unknown split {split!r}")


def _validate_manifest_against_dataset(
    manifest: SplitManifest,
    dataset_root: Path,
) -> None:
    pairs = find_pairs(
        dataset_root / "images" / "render",
        dataset_root / "images" / "ground",
    )
    paired_ids = {_frame_id(image_path) for image_path, _ in pairs}
    manifest_ids = set(manifest.all_ids)

    missing = sorted(manifest_ids - paired_ids)
    if missing:
        raise ValueError(
            f"split manifest references {len(missing)} frame IDs missing from the dataset: "
            f"{missing[:5]}"
        )

    excluded = load_excluded_ids(dataset_root / name for name in manifest.excluded_id_files)
    wrongly_included = sorted(manifest_ids & excluded)
    if wrongly_included:
        raise ValueError(
            "split manifest contains quality-excluded frame IDs: "
            f"{wrongly_included[:5]}"
        )

    eligible = paired_ids - excluded
    omitted = sorted(eligible - manifest_ids)
    if omitted:
        raise ValueError(
            f"split manifest omits {len(omitted)} eligible dataset frame IDs: {omitted[:5]}"
        )

    explicitly_included = load_excluded_ids(
        dataset_root / name for name in manifest.included_id_files
    )
    absent_included = sorted(explicitly_included - manifest_ids)
    if absent_included:
        raise ValueError(
            "split manifest omits explicitly included frame IDs: "
            f"{absent_included[:5]}"
        )


def load_split_manifest(
    path: str | Path = DEFAULT_SPLIT_MANIFEST,
    dataset_root: str | Path | None = None,
) -> SplitManifest:
    """Load and validate an immutable train/validation/test frame-ID manifest."""
    path = Path(path)
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    expected_sha256 = {
        DEFAULT_SPLIT_MANIFEST.resolve(): CANONICAL_SPLIT_SHA256,
        ALIGNED_SPLIT_MANIFEST.resolve(): ALIGNED_SPLIT_SHA256,
    }.get(path.resolve())
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError(
            "known split manifest checksum mismatch: "
            f"expected {expected_sha256}, got {sha256}"
        )

    payload = json.loads(raw)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported split manifest schema version")
    if not isinstance(payload.get("seed"), int):
        raise ValueError("manifest seed must be an integer")

    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("manifest must contain a splits object")
    train_ids = _validate_manifest_ids("train", splits.get("train"))
    validation_ids = _validate_manifest_ids("validation", splits.get("validation"))
    test_ids = _validate_manifest_ids("test", splits.get("test"))

    train_set = set(train_ids)
    validation_set = set(validation_ids)
    test_set = set(test_ids)
    if train_set & validation_set or train_set & test_set or validation_set & test_set:
        raise ValueError("manifest train, validation, and test splits must be disjoint")

    manifest = SplitManifest(
        path=path,
        sha256=sha256,
        seed=payload["seed"],
        excluded_id_files=_validate_manifest_file_names(
            "excluded_id_files", payload.get("excluded_id_files")
        ),
        included_id_files=_validate_manifest_file_names(
            "included_id_files", payload.get("included_id_files")
        ),
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
    )

    declared_counts = payload.get("counts")
    if declared_counts != manifest.counts:
        raise ValueError(
            f"manifest counts do not match split contents: {declared_counts!r} "
            f"!= {manifest.counts!r}"
        )
    if dataset_root is not None:
        _validate_manifest_against_dataset(manifest, Path(dataset_root))
    return manifest


def _mask_from_ground(ground_rgb: np.ndarray) -> np.ndarray:
    """Decode black/red/green/blue masks to background/sky/small/large rock IDs."""
    channel_max = ground_rgb.max(axis=-1)
    dominant = ground_rgb.argmax(axis=-1)  # 0=red, 1=green, 2=blue

    class_map = np.zeros(ground_rgb.shape[:2], dtype=np.int64)
    class_map[dominant == 0] = 1
    class_map[dominant == 1] = 2
    class_map[dominant == 2] = 3
    class_map[channel_max < BACKGROUND_THRESHOLD] = 0
    return class_map


def _decode_pair(pair: Pair) -> tuple[np.ndarray, np.ndarray]:
    image_path, mask_path = pair
    with Image.open(image_path) as image_file:
        image = np.asarray(image_file.convert("RGB"))
    with Image.open(mask_path) as mask_file:
        ground = np.asarray(mask_file.convert("RGB"))
    return image, _mask_from_ground(ground)


class LunarDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Pair],
        processor: Any,
        transform: PairedTransform | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        *,
        split_name: str | None = None,
        split_manifest: SplitManifest | None = None,
    ) -> None:
        self.pairs = list(pairs)
        self.processor = processor
        self.transform = transform
        self.processor_kwargs = dict(processor_kwargs or {})
        self.split_name = split_name
        self.split_manifest = split_manifest

    @property
    def frame_ids(self) -> tuple[str, ...]:
        return tuple(_frame_id(pair[0]) for pair in self.pairs)

    @property
    def manifest_sha256(self) -> str | None:
        return self.split_manifest.sha256 if self.split_manifest is not None else None

    @property
    def manifest_counts(self) -> dict[str, int] | None:
        return self.split_manifest.counts if self.split_manifest is not None else None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Decode and preprocess one batch per worker rather than per item."""
        decoded: list[tuple[np.ndarray, np.ndarray]] = []
        for index in indices:
            image, mask = _decode_pair(self.pairs[index])
            if self.transform is not None:
                image, mask = self.transform(image, mask)
            decoded.append((image, mask))

        images, masks = zip(*decoded)
        encoded = self.processor(
            images=list(images),
            return_tensors="pt",
            **self.processor_kwargs,
        )
        pixel_values = encoded.get("pixel_values")
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
            raise ValueError("processor must return a BCHW pixel_values tensor")
        if pixel_values.shape[0] != len(masks):
            raise ValueError("processor changed the batch size")

        labels = torch.stack([torch.from_numpy(mask.copy()).long() for mask in masks])
        if pixel_values.shape[-2:] != labels.shape[-2:]:
            raise ValueError(
                "processor must not resize images; transformed image and label shapes are "
                f"{tuple(pixel_values.shape[-2:])} and {tuple(labels.shape[-2:])}"
            )
        return [
            {"pixel_values": pixels, "labels": target}
            for pixels, target in zip(pixel_values, labels)
        ]


def _limit(items: list[Pair], count: int | None) -> list[Pair]:
    if count is not None and count < 0:
        raise ValueError("sample limits must be non-negative")
    return items if count is None else items[:count]


def load_datasets(
    processor: Any,
    dataset_root: str | Path = DATASET_ROOT,
    split_manifest: str | Path = DEFAULT_SPLIT_MANIFEST,
    mask_variant: MaskVariant = "ground",
    train_transform: PairedTransform | None = TrainSegmentationTransform(),
    eval_transform: PairedTransform | None = EvalSegmentationTransform(),
    processor_kwargs: dict[str, Any] | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[LunarDataset, LunarDataset, LunarDataset]:
    """Build canonical datasets from explicit manifest membership."""
    if mask_variant not in MASK_VARIANTS:
        raise ValueError(
            f"mask_variant must be one of {list(MASK_VARIANTS)}, got {mask_variant!r}"
        )
    dataset_root = Path(dataset_root)
    manifest = load_split_manifest(split_manifest, dataset_root=dataset_root)
    pairs = find_pairs(
        dataset_root / "images" / "render",
        dataset_root / "images" / mask_variant,
    )
    indexed_pairs = {_frame_id(pair[0]): pair for pair in pairs}

    def make_dataset(
        split_name: str,
        transform: PairedTransform | None,
        limit: int | None,
    ) -> LunarDataset:
        split_pairs = [indexed_pairs[frame_id] for frame_id in manifest.ids_for(split_name)]
        return LunarDataset(
            _limit(split_pairs, limit),
            processor,
            transform,
            processor_kwargs,
            split_name=split_name,
            split_manifest=manifest,
        )

    return (
        make_dataset("train", train_transform, max_train_samples),
        make_dataset("validation", eval_transform, max_val_samples),
        make_dataset("test", eval_transform, max_test_samples),
    )
