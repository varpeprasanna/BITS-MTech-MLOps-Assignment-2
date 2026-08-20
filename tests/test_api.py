from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAT_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PetImages"
    / "Cat"
    / "0.jpg"
)

DOG_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PetImages"
    / "Dog"
    / "0.jpg"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model"] == "CatsDogsCNN"
    assert data["device"] == "cpu"


def test_cat_prediction(client):
    with open(CAT_IMAGE, "rb") as image:

        response = client.post(
            "/predict",
            files={
                "file": (
                    "cat.jpg",
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
    with open(DOG_IMAGE, "rb") as image:

        response = client.post(
            "/predict",
            files={
                "file": (
                    "dog.jpg",
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