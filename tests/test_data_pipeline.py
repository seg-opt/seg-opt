from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from src.data.dataloaders import make_dataloader, make_dataloaders, move_batch_to_device
from src.data.dataset import (
    LunarDataset,
    _decode_pair,
    _mask_from_ground,
    filter_pairs,
    find_pairs,
    load_excluded_ids,
    split_pairs,
)
from src.utils.utils import load_config


class RecordingProcessor:
    """Minimal processor double, avoiding a transformers dependency in data tests."""

    def __init__(self):
        self.batch_sizes = []

    def __call__(self, images, return_tensors):
        assert return_tensors == "pt"
        self.batch_sizes.append(len(images))
        return {
            "pixel_values": torch.stack(
                [torch.from_numpy(image.copy()).permute(2, 0, 1).float() for image in images]
            ),
        }


def _write_pair(root, frame_id, image_prefix="render", mask_prefix="ground"):
    image_dir = root / "render"
    mask_dir = root / "ground"
    image_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    image = np.full((2, 3, 3), int(frame_id), dtype=np.uint8)
    mask = np.zeros((2, 3, 3), dtype=np.uint8)
    mask[..., int(frame_id) % 3] = 255
    Image.fromarray(image).save(image_dir / f"{image_prefix}{frame_id}.png")
    Image.fromarray(mask).save(mask_dir / f"{mask_prefix}{frame_id}.png")


def test_pairing_uses_frame_ids_and_rejects_missing_pairs(tmp_path):
    _write_pair(tmp_path, "0002")
    _write_pair(tmp_path, "0001")

    pairs = find_pairs(tmp_path / "render", tmp_path / "ground")

    assert [(image.name, mask.name) for image, mask in pairs] == [
        ("render0001.png", "ground0001.png"),
        ("render0002.png", "ground0002.png"),
    ]

    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(
        tmp_path / "render" / "render0003.png"
    )
    with pytest.raises(ValueError, match="1 missing masks"):
        find_pairs(tmp_path / "render", tmp_path / "ground")


def test_exclusions_and_mask_decoding(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("0002\n0004\n")
    second.write_text("0004\n")
    pairs = [
        (Path(f"render{index:04d}.png"), Path(f"ground{index:04d}.png"))
        for index in range(1, 5)
    ]

    excluded = load_excluded_ids([first, second])
    filtered = filter_pairs(pairs, excluded)
    ground = np.array(
        [[[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0], [9, 1, 1]]],
        dtype=np.uint8,
    )

    assert excluded == {"0002", "0004"}
    assert [image.stem for image, _ in filtered] == ["render0001", "render0003"]
    np.testing.assert_array_equal(_mask_from_ground(ground), [[0, 1, 2, 3, 0]])


def test_dataset_batches_processing_and_returns_expected_tensors(tmp_path):
    for index in range(4):
        _write_pair(tmp_path, f"{index:04d}")
    processor = RecordingProcessor()
    dataset = LunarDataset(find_pairs(tmp_path / "render", tmp_path / "ground"), processor)

    batch = next(iter(make_dataloader(dataset, 4, False, num_workers=0)))

    assert processor.batch_sizes == [4]
    assert batch["pixel_values"].shape == (4, 3, 2, 3)
    assert batch["pixel_values"].dtype == torch.float32
    assert batch["labels"].shape == (4, 2, 3)
    assert batch["labels"].dtype == torch.int64


def test_split_is_deterministic_disjoint_and_complete():
    pairs = [(Path(f"image{i}.png"), Path(f"mask{i}.png")) for i in range(20)]

    first = split_pairs(pairs, seed=123)
    second = split_pairs(pairs, seed=123)
    train, val, test = first

    assert first == second
    assert (len(train), len(val), len(test)) == (16, 2, 2)
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert set(train + val + test) == set(pairs)


def test_dataloader_policies_seed_and_device_transfer():
    dataset = TensorDataset(torch.arange(5))
    train, val, test = make_dataloaders(dataset, dataset, dataset, 2, num_workers=0, seed=17)
    repeat = make_dataloader(dataset, 2, True, num_workers=0, drop_last=True, seed=17)

    assert train.drop_last and not val.drop_last and not test.drop_last
    assert (len(train), len(val), len(test)) == (2, 3, 3)
    torch.testing.assert_close(
        torch.cat([batch[0] for batch in train]),
        torch.cat([batch[0] for batch in repeat]),
    )

    batch = {"pixel_values": torch.ones(2, 3), "labels": torch.zeros(2, dtype=torch.long)}
    moved = move_batch_to_device(batch, torch.device("cpu"))
    assert all(tensor.device.type == "cpu" for tensor in moved.values())


def test_configured_real_dataset_and_print_entries():
    """Validate the real dataset; run pytest with -s to display sample summaries."""
    config = load_config("experiments/baseline_resnet34/config.yaml").data
    root = Path(config.dataset_root)

    assert root.is_dir()
    assert Path(config.image_dir) == root / "images" / "render"
    assert Path(config.mask_dir) == root / "images" / "ground"

    pairs = find_pairs(config.image_dir, config.mask_dir)
    excluded = load_excluded_ids(root / name for name in config.excluded_id_files)
    filtered = filter_pairs(pairs, excluded)

    assert (len(pairs), len(excluded), len(filtered)) == (9_766, 573, 9_193)

    for index, pair in enumerate(filtered[:3]):
        image, labels = _decode_pair(pair)
        classes, counts = np.unique(labels, return_counts=True)
        assert image.shape == (480, 720, 3)
        assert labels.shape == (480, 720)
        assert set(classes).issubset({0, 1, 2, 3})
        print(
            f"entry {index}: image={pair[0].name}, mask={pair[1].name}, "
            f"image_shape={image.shape}, image_range=({image.min()}, {image.max()}), "
            f"label_shape={labels.shape}, "
            f"class_counts={dict(zip(classes.tolist(), counts.tolist()))}"
        )
