from __future__ import annotations

import sys
from pathlib import Path

import requests

import os

BASE_URL = os.getenv(
    "SMOKE_TEST_URL",
    "http://localhost:8001",
).rstrip("/")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "cat.jpg"


def fail(message: str) -> None:
    print(f"SMOKE TEST FAILED: {message}")
    sys.exit(1)


def run_health_check() -> None:
    print("=" * 70)
    print("HEALTH CHECK")
    print("=" * 70)

    try:
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=10,
        )
    except requests.RequestException as exc:
        fail(f"Health endpoint is unreachable: {exc}")

    print(f"HTTP status: {response.status_code}")
    print(f"Response: {response.text}")

    if response.status_code != 200:
        fail(
            f"Health endpoint returned HTTP "
            f"{response.status_code}"
        )

    try:
        payload = response.json()
    except ValueError:
        fail("Health endpoint did not return valid JSON")

    if payload.get("status") != "healthy":
        fail(
            "Health endpoint did not report "
            f"healthy status: {payload}"
        )

    if payload.get("model_loaded") is not True:
        fail(
            "Model is not loaded according to "
            f"health endpoint: {payload}"
        )

    print("Health check: PASSED")
    print()


def run_prediction_smoke_test() -> None:
    print("=" * 70)
    print("PREDICTION SMOKE TEST")
    print("=" * 70)

    if not TEST_IMAGE.exists():
        fail(f"Test image not found: {TEST_IMAGE}")

    try:
        with TEST_IMAGE.open("rb") as image_file:
            response = requests.post(
                f"{BASE_URL}/predict",
                files={
                    "file": (
                        TEST_IMAGE.name,
                        image_file,
                        "image/jpeg",
                    )
                },
                timeout=30,
            )
    except requests.RequestException as exc:
        fail(f"Prediction endpoint is unreachable: {exc}")

    print(f"HTTP status: {response.status_code}")
    print(f"Response: {response.text}")

    if response.status_code != 200:
        fail(
            "Prediction endpoint returned HTTP "
            f"{response.status_code}"
        )

    try:
        payload = response.json()
    except ValueError:
        fail("Prediction endpoint did not return valid JSON")

    required_fields = {
        "predicted_label",
        "class_probabilities",
        "confidence",
    }

    missing_fields = required_fields - set(payload)

    if missing_fields:
        fail(
            "Prediction response is missing fields: "
            f"{sorted(missing_fields)}"
        )

    if payload["predicted_label"] not in {"cat", "dog"}:
        fail(
            "Unexpected predicted label: "
            f"{payload['predicted_label']}"
        )

    probabilities = payload["class_probabilities"]

    if not isinstance(probabilities, dict):
        fail("class_probabilities is not a JSON object")

    if not {"cat", "dog"}.issubset(probabilities):
        fail(
            "class_probabilities must contain "
            "cat and dog"
        )

    confidence = payload["confidence"]

    if not isinstance(confidence, (int, float)):
        fail("confidence is not numeric")

    if not 0.0 <= confidence <= 1.0:
        fail(
            f"confidence is outside [0, 1]: {confidence}"
        )

    print("Prediction smoke test: PASSED")
    print()


def main() -> None:
    print("=" * 70)
    print("CATS VS DOGS POST-DEPLOYMENT SMOKE TEST")
    print("=" * 70)
    print(f"Base URL  : {BASE_URL}")
    print(f"Test image: {TEST_IMAGE}")
    print()

    run_health_check()
    run_prediction_smoke_test()

    print("=" * 70)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()