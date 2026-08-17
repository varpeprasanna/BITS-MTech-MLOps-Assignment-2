from pathlib import Path
from collections import Counter

from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def inspect_dataset(data_dir: str) -> None:
    root = Path(data_dir)

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    image_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print("=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)

    print(f"Dataset root       : {root.resolve()}")
    print(f"Total image files  : {len(image_files)}")

    extensions = Counter(path.suffix.lower() for path in image_files)

    print("\nFile extensions:")
    for extension, count in sorted(extensions.items()):
        print(f"  {extension:8} : {count}")

    # Directory-level distribution
    parent_dirs = Counter(path.parent.name for path in image_files)

    print("\nImmediate parent directory distribution:")
    for directory, count in parent_dirs.most_common():
        print(f"  {directory:30} : {count}")

    # Image properties
    dimensions = Counter()
    modes = Counter()

    corrupt_images = []

    print("\nInspecting image properties...")

    for image_path in image_files:
        try:
            with Image.open(image_path) as image:
                dimensions[image.size] += 1
                modes[image.mode] += 1

        except Exception as exc:
            corrupt_images.append((str(image_path), str(exc)))

    print("\nTop image dimensions:")
    for dimension, count in dimensions.most_common(20):
        print(f"  {dimension!s:15} : {count}")

    print("\nImage modes:")
    for mode, count in modes.most_common():
        print(f"  {mode:10} : {count}")

    print("\nCorrupt/unreadable images:")
    print(f"  Count: {len(corrupt_images)}")

    if corrupt_images:
        print("\nFirst 10 problematic files:")
        for path, error in corrupt_images[:10]:
            print(f"  {path}")
            print(f"    Error: {error}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    inspect_dataset(r"E:\BITS Sem 3\MLOPS\Assignment 2\data\raw\PetImages")