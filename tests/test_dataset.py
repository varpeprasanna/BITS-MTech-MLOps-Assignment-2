from pathlib import Path

import torch

from src.data_loader import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def test_dataset_sample_shape() -> None:
    train_loader, _, _ = create_dataloaders(
        PROCESSED_DIR,
        batch_size=4,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    assert images.shape == (4, 3, 224, 224)

    assert labels.shape == (4,)

    assert images.dtype == torch.float32

    assert labels.dtype == torch.float32


def test_dataset_contains_both_classes() -> None:
    train_loader, _, _ = create_dataloaders(
        PROCESSED_DIR,
        batch_size=32,
        num_workers=0,
    )

    labels_seen = set()

    for _, labels in train_loader:
        labels_seen.update(
            labels.tolist()
        )

        if labels_seen == {0.0, 1.0}:
            break

    assert labels_seen == {0.0, 1.0}