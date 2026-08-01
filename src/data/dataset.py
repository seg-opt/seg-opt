from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


DATASET_ROOT = "../../datasets/artificial_lunar_landscape"

# Known-bad frames flagged by the dataset authors (camera glitches, masks that
# don't match their render, no-sky/ground-facing shots, heavy shadow).
EXCLUDED_ID_FILES = [
    "cam_anomaly_IDs.txt",
    "ground_facing_IDs.txt",
    "mismatch_IDs.txt",
    "shadow_IDs.txt",
]


def find_pairs(image_dir, mask_dir):
    def index_by_frame_id(directory):
        indexed = {}
        for path in Path(directory).glob("*.png"):
            frame_id = path.stem[-4:]
            if frame_id in indexed:
                raise ValueError(f"duplicate frame ID {frame_id!r} in {directory}")
            indexed[frame_id] = path
        return indexed

    images = index_by_frame_id(image_dir)
    masks = index_by_frame_id(mask_dir)
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        raise ValueError(
            "image/mask frame IDs do not match: "
            f"{len(missing_masks)} missing masks, {len(missing_images)} missing images"
        )
    return [(images[frame_id], masks[frame_id]) for frame_id in sorted(images)]


def load_excluded_ids(id_files):
    excluded = set()
    for id_file in id_files:
        with open(id_file) as f:
            excluded.update(line.strip() for line in f if line.strip())
    return excluded


def filter_pairs(pairs, excluded_ids):
    return [(img, mask) for img, mask in pairs if img.stem[-4:] not in excluded_ids]


def split_pairs(pairs, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    trainval, test = train_test_split(pairs, test_size=test_frac, random_state=seed)
    val_frac_of_trainval = val_frac / (train_frac + val_frac)
    train, val = train_test_split(trainval, test_size=val_frac_of_trainval, random_state=seed)
    return train, val, test


BACKGROUND_THRESHOLD = 10

def _mask_from_ground(ground_rgb: np.ndarray) -> np.ndarray:
    channel_max = ground_rgb.max(axis=-1)
    dominant = ground_rgb.argmax(axis=-1)  # 0=r, 1=g, 2=b

    class_map = np.zeros(ground_rgb.shape[:2], dtype=np.int64)
    class_map[dominant == 0] = 3  # large_rock (red)
    class_map[dominant == 1] = 2  # small_rock (green)
    class_map[dominant == 2] = 1  # sky (blue)
    class_map[channel_max < BACKGROUND_THRESHOLD] = 0  # background (near-black)
    return class_map


def _decode_pair(pair):
    image_path, mask_path = pair
    with Image.open(image_path) as image_file:
        image = np.asarray(image_file.convert("RGB"))
    with Image.open(mask_path) as mask_file:
        ground = np.asarray(mask_file.convert("RGB"))
    return image, _mask_from_ground(ground)


class LunarDataset(Dataset):
    def __init__(self, pairs, processor):
        self.pairs = pairs
        self.processor = processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices):
        """Fetch and preprocess one batch per worker instead of one item at a time."""
        decoded = [_decode_pair(self.pairs[idx]) for idx in indices]
        images, masks = zip(*decoded)
        encoded = self.processor(
            images=list(images),
            segmentation_maps=list(masks),
            return_tensors="pt",
        )
        return [
            {"pixel_values": pixels, "labels": labels}
            for pixels, labels in zip(encoded["pixel_values"], encoded["labels"])
        ]


def load_datasets(
    processor,
    root=DATASET_ROOT,
    excluded_id_files=EXCLUDED_ID_FILES,
    train_frac=0.8,
    val_frac=0.1,
    test_frac=0.1,
    seed=42,
):
    root = Path(root)
    pairs = find_pairs(root / "images" / "render", root / "images" / "ground")
    excluded = load_excluded_ids(root / f for f in excluded_id_files)
    pairs = filter_pairs(pairs, excluded)
    train, val, test = split_pairs(pairs, train_frac, val_frac, test_frac, seed)

    return (
        LunarDataset(train, processor),
        LunarDataset(val, processor),
        LunarDataset(test, processor),
    )
