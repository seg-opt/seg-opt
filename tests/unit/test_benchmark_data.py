import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.dataset import load_datasets, load_split_manifest
from src.data.processors import SegmentationImageProcessor
from src.data.transforms import EvalSegmentationTransform


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _manifest(path: Path, dataset: str, train: list, validation: list, test: list) -> Path:
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "seed": 42,
        "excluded_id_files": [],
        "included_id_files": [],
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "total": len(train) + len(validation) + len(test),
        },
        "splits": {"train": train, "validation": validation, "test": test},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _load(root: Path, manifest: Path, dataset: str):
    transform = EvalSegmentationTransform(height=32, width=64)
    return load_datasets(
        SegmentationImageProcessor(),
        dataset_root=root,
        split_manifest=manifest,
        train_transform=transform,
        eval_transform=transform,
        dataset=dataset,
    )


def test_cityscapes_label_ids_map_to_train_ids(tmp_path: Path) -> None:
    ids = ["train/aachen/aachen_000000_000019", "val/frankfurt/frankfurt_000000_000294"]
    label_ids = np.zeros((32, 64), dtype=np.uint8)
    label_ids[:, :16] = 7  # road -> 0
    label_ids[:, 16:32] = 26  # car -> 13
    label_ids[:, 32:48] = 33  # bicycle -> 18
    for sample_id in ids:
        _save(tmp_path / "leftImg8bit" / f"{sample_id}_leftImg8bit.png",
              np.zeros((32, 64, 3), dtype=np.uint8))
        _save(tmp_path / "gtFine" / f"{sample_id}_gtFine_labelIds.png", label_ids)
    manifest = _manifest(tmp_path / "split.json", "cityscapes", [ids[0]], [], [ids[1]])

    train, _, test = _load(tmp_path, manifest, "cityscapes")

    labels = test[0]["labels"]
    assert len(train) == len(test) == 1
    assert labels[0, 0] == 0 and labels[0, 20] == 13 and labels[0, 40] == 18
    assert labels[0, 60] == 255


def test_ade20k_labels_shift_and_ignore_unlabelled(tmp_path: Path) -> None:
    root = tmp_path / "ADEChallengeData2016"
    ids = ["training/ADE_train_00000001", "validation/ADE_val_00000001"]
    raw = np.zeros((32, 64), dtype=np.uint8)
    raw[:, 32:] = 150
    raw[:, 16:32] = 1
    for sample_id in ids:
        _save(root / "images" / f"{sample_id}.jpg", np.zeros((32, 64, 3), dtype=np.uint8))
        _save(root / "annotations" / f"{sample_id}.png", raw)
    manifest = _manifest(tmp_path / "split.json", "ade20k", [ids[0]], [], [ids[1]])

    _, _, test = _load(tmp_path, manifest, "ade20k")

    labels = test[0]["labels"]
    assert labels[0, 0] == 255 and labels[0, 20] == 0 and labels[0, 40] == 149


def test_benchmark_manifest_checks_files_and_dataset(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path / "split.json", "ade20k", ["training/a"], ["training/b"], ["validation/c"]
    )
    assert load_split_manifest(manifest).dataset == "ade20k"
    with pytest.raises(ValueError, match="missing from the dataset"):
        load_split_manifest(manifest, dataset_root=tmp_path)
    with pytest.raises(ValueError, match="split manifest is for 'ade20k'"):
        _load(tmp_path, manifest, "cityscapes")
