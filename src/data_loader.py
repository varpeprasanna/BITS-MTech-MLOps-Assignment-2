from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import (
    CatsDogsDataset,
    get_evaluation_transform,
    get_train_transform,
)


DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0


def create_dataloaders(
    processed_dir: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    processed_dir = Path(processed_dir)

    train_dataset = CatsDogsDataset(
        processed_dir / "train.csv",
        transform=get_train_transform(),
    )

    validation_dataset = CatsDogsDataset(
        processed_dir / "validation.csv",
        transform=get_evaluation_transform(),
    )

    test_dataset = CatsDogsDataset(
        processed_dir / "test.csv",
        transform=get_evaluation_transform(),
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )