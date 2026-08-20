from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.dataset import get_evaluation_transform
from src.model import create_model


CLASS_MAPPING = {
    0: "cat",
    1: "dog",
}


class CatDogPredictor:
    """
    Loads the trained CatsDogsCNN model and performs inference.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
    ) -> None:

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )

        dropout = checkpoint.get(
            "dropout",
            0.3,
        )

        self.model = create_model(
            dropout=dropout,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = (
            get_evaluation_transform()
        )

    def preprocess(
        self,
        image: Image.Image,
    ) -> torch.Tensor:

        image = image.convert("RGB")

        tensor = self.transform(image)

        return tensor.unsqueeze(0)

    def predict(
        self,
        image: Image.Image,
    ) -> dict:

        tensor = self.preprocess(image)

        tensor = tensor.to(
            self.device
        )

        with torch.no_grad():

            logits = self.model(
                tensor
            ).squeeze(1)

            dog_probability = (
                torch.sigmoid(logits)
                .item()
            )

        cat_probability = (
            1.0 - dog_probability
        )

        if dog_probability >= 0.5:
            predicted_label = "dog"
            confidence = dog_probability
        else:
            predicted_label = "cat"
            confidence = cat_probability

        return {
            "predicted_label": predicted_label,
            "class_probabilities": {
                "cat": round(
                    cat_probability,
                    4,
                ),
                "dog": round(
                    dog_probability,
                    4,
                ),
            },
            "confidence": round(
                confidence,
                4,
            ),
        }