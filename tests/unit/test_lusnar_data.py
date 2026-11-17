import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataloaders import make_dataloader
from src.data.dataset import load_datasets, load_split_manifest
from src.data.processors import SegmentationImageProcessor
from src.data.transforms import EvalSegmentationTransform


PALETTE = [187, 70, 156, 120, 0, 200, 232, 250, 80, 173, 69, 31, 34, 201, 248]


def _fixture(root):
    splits = {
        "train": ["Moon_1/1693274724752036608", "Moon_1/1693274725693423360"],
        "validation": ["Moon_8/1693274724752036608"],
        "test": ["Moon_9/1693274724752036608"],
    }
    labels = np.tile(np.arange(5, dtype=np.uint8), (32, 7))[:, :32]
    for ids in splits.values():
        for sample_id in ids:
            scene, timestamp = sample_id.split("/")
            directory = root / "scenes" / scene / scene / "image0"
            for name in ("color", "label"):
                (directory / name).mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(
                directory / "color" / f"{timestamp}.png"
            )
            mask = Image.fromarray(labels).convert("P")
            mask.putpalette(PALETTE + [0] * (768 - len(PALETTE)))
            mask.save(directory / "label" / f"{timestamp}.png")
    payload = {
        "schema_version": 1,
        "dataset": "lusnar",
        "seed": 42,
        "excluded_id_files": [],
        "included_id_files": [],
        "counts": {"train": 2, "validation": 1, "test": 1, "total": 4},
        "splits": splits,
    }
    path = root / "split.json"
    path.write_text(json.dumps(payload))
    return path, payload, labels


def _load(root, manifest, **kwargs):
    transform = EvalSegmentationTransform(height=32, width=32)
    return load_datasets(
        SegmentationImageProcessor(), root, manifest,
        dataset="lusnar", train_transform=transform, eval_transform=transform, **kwargs,
    )


def test_lusnar_preserves_palette_indices_and_full_sample_ids(tmp_path):
    path, payload, labels = _fixture(tmp_path)
    train, validation, test = _load(tmp_path, path)
    assert train.frame_ids == tuple(payload["splits"]["train"])
    assert validation.frame_ids == tuple(payload["splits"]["validation"])
    assert test.frame_ids == tuple(payload["splits"]["test"])
    assert train.manifest_counts == payload["counts"]
    batch = next(iter(make_dataloader(train, batch_size=2, shuffle=False, num_workers=0)))
    assert batch["pixel_values"].shape == (2, 3, 32, 32)
    assert batch["labels"].dtype == torch.int64
    torch.testing.assert_close(batch["labels"], torch.from_numpy(np.stack([labels, labels])).long())
    expected = (128 / 255 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(batch["pixel_values"][0, :, 0, 0], expected)


def test_lusnar_rejects_scene_leakage(tmp_path):
    path, payload, _ = _fixture(tmp_path)
    payload["splits"]["validation"] = [payload["splits"]["train"].pop()]
    payload["counts"]["train"] = 1
    payload["counts"]["total"] = 3
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="scenes must not overlap"):
        _load(tmp_path, path)


def test_lusnar_rejects_missing_pairs_and_omitted_samples(tmp_path):
    path, payload, _ = _fixture(tmp_path)
    payload["splits"]["train"].pop()
    payload["counts"]["train"] = 1
    payload["counts"]["total"] = 3
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cover all image0 pairs"):
        _load(tmp_path, path)
    mask = next((tmp_path / "scenes/Moon_9/Moon_9/image0/label").glob("*.png"))
    mask.unlink()
    with pytest.raises(ValueError, match="missing from the dataset"):
        _load(tmp_path, path)


def test_lusnar_rejects_unpaired_extra_image(tmp_path):
    path, _, _ = _fixture(tmp_path)
    directory = tmp_path / "scenes/Moon_1/Moon_1/image0/color"
    Image.new("RGB", (32, 32)).save(directory / "9999999999999999999.png")
    with pytest.raises(ValueError, match="timestamps do not match"):
        _load(tmp_path, path)


def test_lusnar_rejects_invalid_mask_and_clean_variant(tmp_path):
    path, _, _ = _fixture(tmp_path)
    train, _, _ = _load(tmp_path, path)
    Image.fromarray(np.full((32, 32), 5, dtype=np.uint8)).save(train.pairs[0][1])
    with pytest.raises(ValueError, match="class indices 0..4"):
        train[0]
    with pytest.raises(ValueError, match="only supports mask_variant"):
        _load(tmp_path, path, mask_variant="clean")


def test_checked_in_lusnar_manifest_has_expected_scenes_and_counts():
    manifest = load_split_manifest("experiments/lusnar/split_scenes.json")
    assert manifest.dataset == "lusnar"
    assert manifest.counts == {"train": 9568, "validation": 1351, "test": 2087, "total": 13006}
    for split, scenes in (
        ("train", {f"Moon_{i}" for i in range(1, 8)}),
        ("validation", {"Moon_8"}),
        ("test", {"Moon_9"}),
    ):
        assert {sample_id.split("/")[0] for sample_id in manifest.ids_for(split)} == scenes


@pytest.mark.parametrize("sample_id", ["Moon_1/../123", "Moon_10/123", "Moon_1/123/456"])
def test_lusnar_rejects_invalid_sample_ids(tmp_path, sample_id):
    path, payload, _ = _fixture(tmp_path)
    payload["splits"]["test"] = [sample_id]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid frame IDs"):
        load_split_manifest(path)
