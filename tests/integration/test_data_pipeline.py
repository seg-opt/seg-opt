from pathlib import Path

import numpy as np

from src.data.dataset import _decode_pair, filter_pairs, find_pairs, load_excluded_ids
from src.utils.utils import load_config


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
