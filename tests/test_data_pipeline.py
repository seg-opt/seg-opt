from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataloaders import make_dataloader
from src.data.dataset import LunarDataset, find_pairs


class RecordingProcessor:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, images, segmentation_maps, return_tensors):
        images = images if isinstance(images, list) else [images]
        masks = segmentation_maps if isinstance(segmentation_maps, list) else [segmentation_maps]
        self.batch_sizes.append(len(images))
        return {
            "pixel_values": torch.stack(
                [torch.from_numpy(image.copy()).permute(2, 0, 1).float() for image in images]
            ),
            "labels": torch.stack([torch.from_numpy(mask.copy()) for mask in masks]),
        }


def _write_pair(root: Path, name: str):
    image_dir = root / "render"
    mask_dir = root / "ground"
    image_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(image_dir / name)
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(mask_dir / name)


def test_dataset_processor_is_called_once_per_batch(tmp_path):
    for index in range(4):
        _write_pair(tmp_path, f"{index}.png")
    processor = RecordingProcessor()
    dataset = LunarDataset(
        find_pairs(tmp_path / "render", tmp_path / "ground"),
        processor,
    )

    batches = list(make_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0))

    assert processor.batch_sizes == [4]
    assert batches[0]["pixel_values"].shape == (4, 3, 4, 5)
    assert batches[0]["labels"].shape == (4, 4, 5)


def test_find_pairs_rejects_mismatched_filenames(tmp_path):
    _write_pair(tmp_path, "a.png")
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(tmp_path / "render" / "b.png")

    with pytest.raises(ValueError, match="1 missing masks"):
        find_pairs(tmp_path / "render", tmp_path / "ground")
