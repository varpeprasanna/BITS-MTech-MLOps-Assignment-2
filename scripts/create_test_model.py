from pathlib import Path

import torch

from src.model import create_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "best_model.pt"
)


def main() -> None:
    """
    Create a deterministic model checkpoint for CI Docker builds.

    This is NOT the production trained model.

    It exists only so CI can validate that the Docker image
    can be constructed from a clean Git checkout.
    """

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.manual_seed(42)

    model = create_model(
        dropout=0.3,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": "CatsDogsCNN",
        "dropout": 0.3,
        "image_size": 224,
        "class_mapping": {
            "cat": 0,
            "dog": 1,
        },
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    print(
        f"CI test model created: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()