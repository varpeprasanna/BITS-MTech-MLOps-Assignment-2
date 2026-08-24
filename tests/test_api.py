import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


@pytest.fixture
def client(test_model_path, monkeypatch):
    """
    Start the FastAPI application using the deterministic
    test checkpoint instead of the production model.
    """

    monkeypatch.setattr(
        "api.main.MODEL_PATH",
        test_model_path,
    )

    with TestClient(app) as test_client:
        yield test_client


def create_test_image(
    image_format="JPEG",
):
    """
    Create a small valid RGB image entirely in memory.
    """

    image = Image.new(
        "RGB",
        (224, 224),
        (128, 128, 128),
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format=image_format,
    )

    buffer.seek(0)

    return buffer


def test_health(client):

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model"] == "CatsDogsCNN"
    assert data["device"] == "cpu"


def test_cat_prediction(client):

    image = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_label"] in {
        "cat",
        "dog",
    }

    probabilities = data[
        "class_probabilities"
    ]

    assert 0.0 <= probabilities["cat"] <= 1.0
    assert 0.0 <= probabilities["dog"] <= 1.0

    assert (
        abs(
            probabilities["cat"]
            + probabilities["dog"]
            - 1.0
        )
        < 1e-4
    )

    assert 0.0 <= data["confidence"] <= 1.0


def test_dog_prediction(client):

    image = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_label"] in {
        "cat",
        "dog",
    }

    probabilities = data[
        "class_probabilities"
    ]

    assert (
        abs(
            probabilities["cat"]
            + probabilities["dog"]
            - 1.0
        )
        < 1e-4
    )


def test_invalid_file(client):

    response = client.post(
        "/predict",
        files={
            "file": (
                "invalid.txt",
                b"this is not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    assert (
        "Unsupported image format"
        in response.json()["detail"]
    )


def test_empty_file(client):

    response = client.post(
        "/predict",
        files={
            "file": (
                "empty.jpg",
                b"",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400

    assert (
        "empty"
        in response.json()["detail"].lower()
    )