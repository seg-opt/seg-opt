from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from functools import partial

import numpy as np
import torch
from src.data.dataset import LunarDataset, SamplePair, _decode_pair
from torch.utils.data import DataLoader, Dataset


def _collate_lunar(
    dataset: LunarDataset, pairs: Sequence[SamplePair]
) -> dict[str, torch.Tensor]:
    decoded = [_decode_pair(pair) for pair in pairs]
    images = [image for image, _ in decoded]
    labels = [label for _, label in decoded]

    try:
        encoded = dataset.processor(
            images=images,
            segmentation_maps=labels,
            return_tensors="pt",
        )
    except TypeError:
        encoded = dataset.processor(images=images, return_tensors="pt")

    batch = dict(encoded)
    if "labels" not in batch:
        batch["labels"] = torch.from_numpy(np.stack(labels)).long()
    return batch


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _pin_memory_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.get_device_properties(0)
    except RuntimeError:
        return False
    return True


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool = False,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    collate_fn = None
    if isinstance(dataset, LunarDataset):
        collate_fn = partial(_collate_lunar, dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=_seed_worker,
        pin_memory=_pin_memory_available(),
        persistent_workers=num_workers > 0,
    )


def make_dataloaders(
    train: Dataset,
    val: Dataset,
    test: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    return (
        make_dataloader(
            train, batch_size, True, num_workers, drop_last=True, seed=seed
        ),
        make_dataloader(val, batch_size, False, num_workers, seed=seed),
        make_dataloader(test, batch_size, False, num_workers, seed=seed),
    )


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}