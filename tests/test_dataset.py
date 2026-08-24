from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from src.data_loader import create_dataloaders


@pytest.fixture
def mini_dataset(tmp_path: Path) -> Path:
    """
    Create a tiny self-contained dataset for CI tests.

    The real DVC-managed dataset is intentionally not required.
    """

    processed_dir = tmp_path / "processed"
    image_dir = tmp_path / "images"

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cat_dir = image_dir / "Cat"
    dog_dir = image_dir / "Dog"

    cat_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dog_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    # Create deterministic test images.
    for index in range(4):

        cat_path = (
            cat_dir / f"cat_{index}.jpg"
        )

        dog_path = (
            dog_dir / f"dog_{index}.jpg"
        )

        cat_image = Image.new(
            "RGB",
            (300, 300),
            (255, 100, 100),
        )

        dog_image = Image.new(
            "RGB",
            (300, 300),
            (100, 100, 255),
        )

        cat_image.save(
            cat_path,
            format="JPEG",
        )

        dog_image.save(
            dog_path,
            format="JPEG",
        )

        records.append(
            {
                "filepath": str(cat_path),
                "label": 0,
                "class_name": "cat",
            }
        )

        records.append(
            {
                "filepath": str(dog_path),
                "label": 1,
                "class_name": "dog",
            }
        )

    dataframe = pd.DataFrame(records)

    # create_dataloaders() expects all three manifests.
    dataframe.to_csv(
        processed_dir / "train.csv",
        index=False,
    )

    dataframe.to_csv(
        processed_dir / "validation.csv",
        index=False,
    )

    dataframe.to_csv(
        processed_dir / "test.csv",
        index=False,
    )

    return processed_dir


def test_dataset_sample_shape(
    mini_dataset: Path,
) -> None:

    train_loader, _, _ = create_dataloaders(
        mini_dataset,
        batch_size=4,
        num_workers=0,
    )

    images, labels = next(
        iter(train_loader)
    )

    assert images.shape == (
        4,
        3,
        224,
        224,
    )

    assert labels.shape == (
        4,
    )

    assert images.dtype == torch.float32

    assert labels.dtype == torch.float32


def test_dataset_contains_both_classes(
    mini_dataset: Path,
) -> None:

    train_loader, _, _ = create_dataloaders(
        mini_dataset,
        batch_size=8,
        num_workers=0,
    )

    labels_seen = set()

    for _, labels in train_loader:

        labels_seen.update(
            labels.tolist()
        )

        if labels_seen == {
            0.0,
            1.0,
        }:
            break

    assert labels_seen == {
        0.0,
        1.0,
    }