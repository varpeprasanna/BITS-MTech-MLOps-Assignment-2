from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

CLASS_MAPPING = {
    "Cat": 0,
    "Dog": 1,
}


def discover_images(raw_dir: Path) -> pd.DataFrame:
    records = []

    for class_name, label in CLASS_MAPPING.items():
        class_dir = raw_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected class directory does not exist: {class_dir}"
            )

        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }:
                continue

            records.append(
                {
                    "filepath": str(image_path),
                    "label": label,
                    "class_name": class_name.lower(),
                }
            )

    if not records:
        raise RuntimeError("No image files were discovered.")

    return pd.DataFrame(records)


def create_stratified_splits(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_df, temp_df = train_test_split(
        dataframe,
        test_size=VALIDATION_RATIO + TEST_RATIO,
        stratify=dataframe["label"],
        random_state=RANDOM_SEED,
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO),
        stratify=temp_df["label"],
        random_state=RANDOM_SEED,
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def create_metadata(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:

    return {
        "random_seed": RANDOM_SEED,
        "image_size": [224, 224],
        "color_mode": "RGB",
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },
        "class_mapping": {
            "cat": 0,
            "dog": 1,
        },
        "total_images": int(
            len(train_df) + len(validation_df) + len(test_df)
        ),
        "splits": {
            "train": {
                "count": int(len(train_df)),
                "cat": int((train_df["label"] == 0).sum()),
                "dog": int((train_df["label"] == 1).sum()),
            },
            "validation": {
                "count": int(len(validation_df)),
                "cat": int((validation_df["label"] == 0).sum()),
                "dog": int((validation_df["label"] == 1).sum()),
            },
            "test": {
                "count": int(len(test_df)),
                "cat": int((test_df["label"] == 0).sum()),
                "dog": int((test_df["label"] == 1).sum()),
            },
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "raw" / "PetImages"
    processed_dir = project_root / "data" / "processed"

    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CREATING DATASET SPLITS")
    print("=" * 70)

    print(f"Raw dataset : {raw_dir}")
    print(f"Output dir  : {processed_dir}")
    print(f"Random seed : {RANDOM_SEED}")
    print()

    dataframe = discover_images(raw_dir)

    print(f"Discovered images: {len(dataframe)}")

    train_df, validation_df, test_df = create_stratified_splits(
        dataframe
    )

    train_df.to_csv(
        processed_dir / "train.csv",
        index=False,
    )

    validation_df.to_csv(
        processed_dir / "validation.csv",
        index=False,
    )

    test_df.to_csv(
        processed_dir / "test.csv",
        index=False,
    )

    metadata = create_metadata(
        train_df,
        validation_df,
        test_df,
    )

    with open(
        processed_dir / "dataset_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\nSplit results")
    print("-" * 70)

    for split_name, split_data in metadata["splits"].items():
        print(
            f"{split_name.capitalize():12} : "
            f"{split_data['count']:5} "
            f"(Cat={split_data['cat']}, "
            f"Dog={split_data['dog']})"
        )

    print("\nFiles created:")
    print(f"  {processed_dir / 'train.csv'}")
    print(f"  {processed_dir / 'validation.csv'}")
    print(f"  {processed_dir / 'test.csv'}")
    print(f"  {processed_dir / 'dataset_metadata.json'}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()