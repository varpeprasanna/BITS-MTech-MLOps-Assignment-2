from pathlib import Path

import torch

from src.data_loader import create_dataloaders


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("DATA PIPELINE GPU TEST")
    print("=" * 70)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    train_loader, validation_loader, test_loader = (
        create_dataloaders(
            processed_dir=processed_dir,
            batch_size=16,
            num_workers=0,
        )
    )

    images, labels = next(iter(train_loader))

    print(f"\nCPU batch:")
    print(f"  Images : {images.shape}")
    print(f"  Labels : {labels.shape}")
    print(f"  dtype  : {images.dtype}")

    images = images.to(
        device,
        non_blocking=True,
    )

    labels = labels.to(
        device,
        non_blocking=True,
    )

    print("\nGPU batch:")
    print(f"  Images device : {images.device}")
    print(f"  Labels device : {labels.device}")

    if torch.cuda.is_available():
        print(
            f"  GPU memory allocated: "
            f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        )

    print("\nLoader sizes:")
    print(f"  Train      : {len(train_loader)} batches")
    print(
        f"  Validation : "
        f"{len(validation_loader)} batches"
    )
    print(f"  Test       : {len(test_loader)} batches")

    print("\n" + "=" * 70)
    print("DATA PIPELINE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()