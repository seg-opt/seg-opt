from pathlib import Path

import numpy as np

from src.data.dataset import _decode_pair, find_pairs, load_split_manifest
from src.utils.utils import load_config


def test_configured_real_dataset_and_print_entries():
    """Validate real dataset counts, split manifest, decoded shapes, and classes."""
    config = load_config("experiments/baselines_benchmark/resnet34_unet.yaml").data
    root = Path(config.dataset_root)

    assert root.is_dir()
    image_dir = root / "images" / "render"
    mask_dir = root / "images" / config.mask_variant
    pairs = find_pairs(image_dir, mask_dir)
    manifest = load_split_manifest(config.split_manifest, dataset_root=root)

    assert len(pairs) == 9_766
    assert manifest.counts == {
        "train": 7_200,
        "validation": 896,
        "test": 897,
        "total": 8_993,
    }
    indexed = {image.stem[-4:]: (image, mask) for image, mask in pairs}

    for frame_id in manifest.train_ids[:3]:
        pair = indexed[frame_id]
        image, labels = _decode_pair(pair)
        classes = np.unique(labels)
        assert image.shape == (480, 720, 3)
        assert labels.shape == (480, 720)
        assert set(classes).issubset({0, 1, 2, 3})
