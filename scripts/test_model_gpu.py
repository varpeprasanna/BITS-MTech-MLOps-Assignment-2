from pathlib import Path

import torch

from src.data_loader import create_dataloaders
from src.model import create_model


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("MODEL GPU SMOKE TEST")
    print("=" * 70)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    model = create_model().to(device)

    train_loader, _, _ = create_dataloaders(
        processed_dir=processed_dir,
        batch_size=16,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    images = images.to(device)
    labels = labels.to(device)

    print(f"\nInput shape : {images.shape}")
    print(f"Labels shape: {labels.shape}")

    with torch.no_grad():
        logits = model(images)

    print(f"Output shape: {logits.shape}")

    probabilities = torch.sigmoid(logits)

    print(
        f"Probability range: "
        f"{probabilities.min().item():.4f} - "
        f"{probabilities.max().item():.4f}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU memory allocated: "
            f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        )

    print("\nMODEL GPU TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()