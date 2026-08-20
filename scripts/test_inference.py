from pathlib import Path

from PIL import Image

from src.inference import CatDogPredictor


def main() -> None:

    project_root = (
        Path(__file__).resolve().parents[1]
    )

    model_path = (
        project_root
        / "artifacts"
        / "best_model.pt"
    )

    # Use one known dataset image.
    image_path = (
        project_root
        / "data"
        / "raw"
        / "PetImages"
        / "Cat"
        / "0.jpg"
    )

    print("=" * 70)
    print("INFERENCE TEST")
    print("=" * 70)

    print(f"Model : {model_path}")
    print(f"Image : {image_path}")

    predictor = CatDogPredictor(
        model_path=model_path,
        device="cpu",
    )

    with Image.open(image_path) as image:

        result = predictor.predict(
            image
        )

    print("\nPrediction:")
    print(
        f"  Label      : "
        f"{result['predicted_label']}"
    )

    print(
        f"  Confidence : "
        f"{result['confidence']}"
    )

    print(
        f"  Cat        : "
        f"{result['class_probabilities']['cat']}"
    )

    print(
        f"  Dog        : "
        f"{result['class_probabilities']['dog']}"
    )

    print("\nINFERENCE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()