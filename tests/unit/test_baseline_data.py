import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import (
    CANONICAL_SPLIT_SHA256,
    DATASET_ROOT,
    DEFAULT_SPLIT_MANIFEST,
    LunarDataset,
    load_datasets,
    load_split_manifest,
)
from src.data.processors import IMAGENET_MEAN, IMAGENET_STD, SegmentationImageProcessor
from src.data.transforms import EvalSegmentationTransform, TrainSegmentationTransform


def _write_pair(root: Path, frame_id: str) -> None:
    image_dir = root / "images" / "render"
    mask_dir = root / "images" / "ground"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image = np.full((4, 6, 3), int(frame_id), dtype=np.uint8)
    ground = np.zeros((4, 6, 3), dtype=np.uint8)
    ground[:, :3, 1] = 255
    Image.fromarray(image).save(image_dir / f"render{frame_id}.png")
    Image.fromarray(ground).save(mask_dir / f"ground{frame_id}.png")


def _write_manifest(
    path: Path,
    train: list[str],
    validation: list[str],
    test: list[str],
    excluded_id_files: list[str] | None = None,
    included_id_files: list[str] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "dataset": "test",
        "seed": 42,
        "excluded_id_files": excluded_id_files or [],
        "included_id_files": included_id_files or [],
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "total": len(train) + len(validation) + len(test),
        },
        "splits": {"train": train, "validation": validation, "test": test},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_full_frame_transforms_have_fixed_shape_and_keep_alignment() -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    mask = np.zeros((4, 6), dtype=np.int64)
    image[1:3, 3:6, 0] = 255
    mask[1:3, 3:6] = 3
    mask[0, 0] = 255

    eval_transform = EvalSegmentationTransform()
    first_image, first_mask = eval_transform(image, mask)
    second_image, second_mask = eval_transform(image, mask)

    assert first_image.shape == (512, 768, 3)
    assert first_mask.shape == (512, 768)
    np.testing.assert_array_equal(first_image, second_image)
    np.testing.assert_array_equal(first_mask, second_mask)
    assert 255 in first_mask
    assert first_image[..., 0][first_mask == 3].mean() > 200

    random.seed(7)
    train_image, train_mask = TrainSegmentationTransform(
        flip_probability=1.0,
        brightness=0.0,
        contrast=0.0,
        gamma_min=1.0,
        gamma_max=1.0,
    )(image, mask)
    assert train_image.shape == (512, 768, 3)
    assert train_mask.shape == (512, 768)
    assert train_image[..., 0][train_mask == 3].mean() > 200
    assert np.count_nonzero(train_mask[:, :384] == 3) > np.count_nonzero(
        train_mask[:, 384:] == 3
    )


def test_imagenet_processor_normalizes_without_resizing() -> None:
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    processor = SegmentationImageProcessor()

    encoded = processor(images=[image], return_tensors="pt")

    assert encoded["pixel_values"].shape == (1, 3, 5, 7)
    expected = -torch.tensor(IMAGENET_MEAN) / torch.tensor(IMAGENET_STD)
    torch.testing.assert_close(encoded["pixel_values"][0, :, 0, 0], expected)

    with pytest.raises(TypeError):
        SegmentationImageProcessor(mean=(0.5, 0.5, 0.5))


def test_manifest_validates_counts_disjointness_and_dataset_membership(tmp_path: Path) -> None:
    for index in range(1, 9):
        _write_pair(tmp_path, f"{index:04d}")
    (tmp_path / "bad.txt").write_text("0008\n", encoding="utf-8")
    (tmp_path / "included.txt").write_text("0007\n", encoding="utf-8")
    path = tmp_path / "split.json"
    _write_manifest(
        path,
        train=["0001", "0002", "0003", "0004"],
        validation=["0005"],
        test=["0006", "0007"],
        excluded_id_files=["bad.txt"],
        included_id_files=["included.txt"],
    )

    manifest = load_split_manifest(path, dataset_root=tmp_path)

    assert manifest.counts == {"train": 4, "validation": 1, "test": 2, "total": 7}
    assert manifest.val_ids == ("0005",)
    assert manifest.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    _write_manifest(
        path,
        train=["0001", "0002"],
        validation=["0002"],
        test=["0003"],
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        load_split_manifest(path)


def test_dataset_loader_uses_manifest_order_and_exposes_metadata(tmp_path: Path) -> None:
    for index in range(1, 5):
        _write_pair(tmp_path, f"{index:04d}")
    path = tmp_path / "split.json"
    _write_manifest(
        path,
        train=["0002", "0001"],
        validation=["0003"],
        test=["0004"],
    )
    transform = EvalSegmentationTransform(height=8, width=12)

    train, validation, test = load_datasets(
        processor=SegmentationImageProcessor(),
        dataset_root=tmp_path,
        split_manifest=path,
        train_transform=transform,
        eval_transform=transform,
    )

    assert (len(train), len(validation), len(test)) == (2, 1, 1)
    assert train.frame_ids == ("0002", "0001")
    assert train.split_name == "train"
    assert validation.split_name == "validation"
    assert train.manifest_sha256 == load_split_manifest(path).sha256
    assert train.manifest_counts == {
        "train": 2,
        "validation": 1,
        "test": 1,
        "total": 4,
    }
    assert train[0]["pixel_values"].shape == (3, 8, 12)
    assert train[0]["labels"].shape == (8, 12)


def test_dataset_rejects_processors_that_change_geometry(tmp_path: Path) -> None:
    _write_pair(tmp_path, "0001")

    class ResizingProcessor:
        def __call__(self, images, return_tensors):
            return {"pixel_values": torch.zeros(len(images), 3, 2, 2)}

    dataset = LunarDataset(
        [(tmp_path / "images/render/render0001.png", tmp_path / "images/ground/ground0001.png")],
        ResizingProcessor(),
    )

    with pytest.raises(ValueError, match="must not resize"):
        dataset[0]


def test_canonical_manifest_is_frozen_and_includes_top200() -> None:
    dataset_root = Path(DATASET_ROOT)
    if not dataset_root.is_dir():
        pytest.skip("artificial lunar landscape dataset is not available")

    manifest = load_split_manifest(DEFAULT_SPLIT_MANIFEST, dataset_root=dataset_root)
    top200 = set((dataset_root / "top200_largerocks_IDs.txt").read_text().split())

    assert manifest.seed == 42
    assert manifest.sha256 == CANONICAL_SPLIT_SHA256
    assert manifest.counts == {
        "train": 7_353,
        "validation": 920,
        "test": 920,
        "total": 9_193,
    }
    assert top200 <= set(manifest.all_ids)
