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
    return tuple(
        make_dataloader(
            dataset,
            batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            seed=seed if shuffle else None,
        )
        for dataset, shuffle in (
            (train_dataset, True), (val_dataset, False), (test_dataset, False)
        )
    )
