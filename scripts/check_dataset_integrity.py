from pathlib import Path
from PIL import Image
import warnings


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def check_dataset(root_dir: str) -> None:
    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(f"Dataset not found: {root}")

    image_files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    warnings_found = []
    errors_found = []

    print("=" * 70)
    print("DATASET INTEGRITY CHECK")
    print("=" * 70)
    print(f"Dataset root : {root.resolve()}")
    print(f"Images       : {len(image_files)}")
    print()

    for image_path in image_files:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                with Image.open(image_path) as image:
                    image.verify()

                for warning in caught:
                    warnings_found.append(
                        {
                            "path": str(image_path),
                            "warning": str(warning.message),
                        }
                    )

        except Exception as exc:
            errors_found.append(
                {
                    "path": str(image_path),
                    "error": str(exc),
                }
            )

    print(f"Warnings : {len(warnings_found)}")
    print(f"Errors   : {len(errors_found)}")

    if warnings_found:
        print("\nWARNINGS")
        print("-" * 70)

        for item in warnings_found:
            print(f"File    : {item['path']}")
            print(f"Warning : {item['warning']}")
            print()

    if errors_found:
        print("\nERRORS")
        print("-" * 70)

        for item in errors_found:
            print(f"File  : {item['path']}")
            print(f"Error : {item['error']}")
            print()

    print("=" * 70)

    if errors_found:
        raise RuntimeError(
            f"Dataset integrity check failed with "
            f"{len(errors_found)} unreadable image(s)."
        )


if __name__ == "__main__":
    check_dataset("data/raw/PetImages")