import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_NUM_WORKERS = 2


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = DEFAULT_NUM_WORKERS,
    drop_last: bool = False,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor= 4 if num_workers > 0 else None,
        generator=generator,
    )


def make_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = make_dataloader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        seed=seed,
    )
    val_loader = make_dataloader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = make_dataloader(
        test_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}
