import os

import torch
from torch.utils.data import DataLoader

DEFAULT_NUM_WORKERS = 2


def make_dataloader(
    dataset,
    batch_size,
    shuffle,
    num_workers=DEFAULT_NUM_WORKERS,
    drop_last=False,
    seed=None,
):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def make_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size,
    num_workers=DEFAULT_NUM_WORKERS,
    seed=42,
):
    train_loader = make_dataloader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        seed=seed,
    )
    val_loader = make_dataloader(val_dataset, batch_size, False, num_workers)
    test_loader = make_dataloader(test_dataset, batch_size, False, num_workers)
    return train_loader, val_loader, test_loader


def move_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
