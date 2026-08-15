from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

SamplePair = tuple[Path, Path]
_FRAME_ID = re.compile(r"^(?:render|ground)(.+)$")


def _frame_id(path: Path) -> str:
    match = _FRAME_ID.match(path.stem)
    if match is None:
        raise ValueError(f"unsupported lunar dataset filename: {path.name}")
    return match.group(1)


def find_pairs(image_dir: str | Path, mask_dir: str | Path) -> list[SamplePair]:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    images = {_frame_id(path): path for path in image_dir.glob("render*.png")}
    masks = {_frame_id(path): path for path in mask_dir.glob("ground*.png")}

    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        details = []
        if missing_masks:
            details.append(f"{len(missing_masks)} missing masks")
        if missing_images:
            details.append(f"{len(missing_images)} missing images")
        raise ValueError(", ".join(details))

    return [(images[frame_id], masks[frame_id]) for frame_id in sorted(images)]


def load_excluded_ids(paths: Iterable[str | Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        excluded.update(
            line.strip()
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return excluded


def filter_pairs(
    pairs: Sequence[SamplePair], excluded_ids: set[str]
) -> list[SamplePair]:
    return [pair for pair in pairs if _frame_id(pair[0]) not in excluded_ids]


def _mask_from_ground(ground: np.ndarray) -> np.ndarray:
    if ground.ndim != 3 or ground.shape[-1] < 3:
        raise ValueError("ground masks must be RGB images")

    rgb = ground[..., :3]
    labels = np.zeros(ground.shape[:2], dtype=np.int64)
    labels[np.all(rgb == (0, 0, 255), axis=-1)] = 1
    labels[np.all(rgb == (0, 255, 0), axis=-1)] = 2
    labels[np.all(rgb == (255, 0, 0), axis=-1)] = 3
    return labels


def _decode_pair(pair: SamplePair) -> tuple[np.ndarray, np.ndarray]:
    image_path, mask_path = pair
    with Image.open(image_path) as image_file:
        image = np.asarray(image_file.convert("RGB")).copy()
    with Image.open(mask_path) as mask_file:
        labels = _mask_from_ground(np.asarray(mask_file.convert("RGB")))
    return image, labels


def split_pairs(
    pairs: Sequence[SamplePair],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[SamplePair], list[SamplePair], list[SamplePair]]:
    if min(train_frac, val_frac, test_frac) < 0:
        raise ValueError("dataset split fractions cannot be negative")
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("dataset split fractions must sum to 1")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_frac)
    val_end = train_end + int(len(shuffled) * val_frac)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


class LunarDataset(Dataset[SamplePair]):
    def __init__(self, pairs: Sequence[SamplePair], processor: object):
        self.pairs = list(pairs)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> SamplePair:
        return self.pairs[index]


def load_datasets(
    processor: object,
    root: str | Path,
    excluded_id_files: Sequence[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[LunarDataset, LunarDataset, LunarDataset]:
    root = Path(root)
    pairs = find_pairs(root / "images" / "render", root / "images" / "ground")
    excluded = load_excluded_ids(root / filename for filename in excluded_id_files)
    splits = split_pairs(
        filter_pairs(pairs, excluded),
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    return tuple(LunarDataset(split, processor) for split in splits)  # type: ignore[return-value]