from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SIZE = 224

# Standard ImageNet normalization.
# We will use this consistently for the baseline CNN.
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def get_train_transform() -> transforms.Compose:
    """
    Transform pipeline used only for training data.

    Includes data augmentation to improve generalization.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.RandomRotation(
                degrees=10
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGE_MEAN,
                std=IMAGE_STD,
            ),
        ]
    )


def get_evaluation_transform() -> transforms.Compose:
    """
    Transform pipeline used for validation and test data.

    No random augmentation is applied.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGE_MEAN,
                std=IMAGE_STD,
            ),
        ]
    )


class CatsDogsDataset(Dataset):
    """
    PyTorch Dataset for the Cats vs Dogs dataset.

    The CSV manifest must contain:
        filepath
        label
        class_name
    """

    def __init__(
        self,
        csv_file: str | Path,
        transform: transforms.Compose | None = None,
    ) -> None:

        self.csv_file = Path(csv_file)
        self.transform = transform

        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"Dataset manifest not found: {self.csv_file}"
            )

        self.data = pd.read_csv(self.csv_file)

        required_columns = {
            "filepath",
            "label",
            "class_name",
        }

        missing_columns = required_columns - set(
            self.data.columns
        )

        if missing_columns:
            raise ValueError(
                f"Missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if len(self.data) == 0:
            raise ValueError(
                f"Dataset manifest is empty: {self.csv_file}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        row = self.data.iloc[index]

        image_path = Path(row["filepath"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                if self.transform is not None:
                    image = self.transform(image)

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image: {image_path}"
            ) from exc

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.float32,
        )

        return image, label