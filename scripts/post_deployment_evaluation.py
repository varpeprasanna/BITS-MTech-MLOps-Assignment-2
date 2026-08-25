from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests


# ============================================================
# Configuration
# ============================================================

BASE_URL = (
    __import__("os").getenv(
        "SMOKE_TEST_URL",
        "http://localhost:8001",
    )
    .rstrip("/")
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"
EVALUATION_DATASET = [
    ("cat.jpg", "cat"),
    ("cat_2.jpg", "cat"),
    ("cat_3.jpg", "cat"),
    ("cat_4.jpg", "cat"),
    ("cat_5.jpg", "cat"),
    ("cat_6.jpg", "cat"),
    ("cat_7.jpg", "cat"),
    ("cat_8.jpg", "cat"),
    ("cat_9.jpg", "cat"),
    ("cat_10.jpg", "cat"),

    ("dog.jpg", "dog"),
    ("dog_2.jpg", "dog"),
    ("dog_3.jpg", "dog"),
    ("dog_4.jpg", "dog"),
    ("dog_5.jpg", "dog"),
    ("dog_6.jpg", "dog"),
    ("dog_7.jpg", "dog"),
    ("dog_8.jpg", "dog"),
    ("dog_9.jpg", "dog"),
    ("dog_10.jpg", "dog"),
]

OUTPUT_DIR = PROJECT_ROOT / "artifacts"

RESULTS_FILE = (
    OUTPUT_DIR
    / "post_deployment_evaluation.json"
)

NUM_IMAGES_PER_CLASS = 10


# ============================================================
# Helpers
# ============================================================

def fail(message: str) -> None:
    print(f"EVALUATION FAILED: {message}")
    sys.exit(1)




def predict_image(
    image_path: Path,
) -> dict:
    """
    Send one image to the deployed API and return
    prediction information plus request latency.
    """

    start = time.perf_counter()

    try:
        with image_path.open("rb") as image_file:

            response = requests.post(
                f"{BASE_URL}/predict",
                files={
                    "file": (
                        image_path.name,
                        image_file,
                        "image/jpeg",
                    )
                },
                timeout=60,
            )

    except requests.RequestException as exc:
        fail(
            f"Prediction request failed for "
            f"{image_path}: {exc}"
        )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    if response.status_code != 200:
        fail(
            f"Prediction failed for {image_path}: "
            f"HTTP {response.status_code} "
            f"{response.text}"
        )

    try:
        payload = response.json()
    except ValueError:
        fail(
            f"Prediction response was not valid JSON "
            f"for {image_path}"
        )

    return {
        "predicted_label": payload[
            "predicted_label"
        ],
        "confidence": float(
            payload["confidence"]
        ),
        "class_probabilities": payload[
            "class_probabilities"
        ],
        "latency_ms": latency_ms,
    }


# ============================================================
# Main evaluation
# ============================================================

def main() -> None:

    print("=" * 70)
    print("POST-DEPLOYMENT MODEL PERFORMANCE EVALUATION")
    print("=" * 70)

    print(f"Base URL: {BASE_URL}")
    print()

    # --------------------------------------------------------
    # Health check
    # --------------------------------------------------------

    print("Checking deployed API...")

    try:
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=10,
        )
    except requests.RequestException as exc:
        fail(
            f"Health endpoint is unreachable: {exc}"
        )

    if response.status_code != 200:
        fail(
            f"Health endpoint returned "
            f"HTTP {response.status_code}"
        )

    health = response.json()

    if health.get("status") != "healthy":
        fail(
            f"API is not healthy: {health}"
        )

    print("API health: PASSED")
    print()

    # --------------------------------------------------------
    # Collect fixed evaluation batch
    # --------------------------------------------------------

    dataset = []

    if not FIXTURE_DIR.exists():
        fail(
            f"Evaluation fixture directory not found: "
            f"{FIXTURE_DIR}"
        )

    for filename, label in EVALUATION_DATASET:
        image_path = FIXTURE_DIR / filename

        if not image_path.exists():
            fail(
                f"Evaluation fixture not found: "
                f"{image_path}"
            )

        dataset.append(
            (image_path, label)
        )

    cat_count = sum(
        1
        for _, label in dataset
        if label == "cat"
    )

    dog_count = sum(
        1
        for _, label in dataset
        if label == "dog"
    )

    print(
        f"Evaluation batch: {len(dataset)} images"
    )

    print(
        f"  Cat images: {cat_count}"
    )

    print(
        f"  Dog images: {dog_count}"
    )

    print()

    # --------------------------------------------------------
    # Run predictions
    # --------------------------------------------------------

    results = []

    for index, (
        image_path,
        true_label,
    ) in enumerate(dataset, start=1):

        prediction = predict_image(
            image_path
        )

        predicted_label = prediction[
            "predicted_label"
        ]

        confidence = prediction[
            "confidence"
        ]

        correct = (
            predicted_label == true_label
        )

        result = {
            "image": str(
                image_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": round(
                confidence,
                4,
            ),
            "latency_ms": round(
                prediction["latency_ms"],
                2,
            ),
            "correct": correct,
        }

        results.append(result)

        status = "CORRECT" if correct else "WRONG"

        print(
            f"[{index:02d}/{len(dataset)}] "
            f"{true_label:>3} -> "
            f"{predicted_label:<3} | "
            f"confidence={confidence:.4f} | "
            f"latency={prediction['latency_ms']:.2f} ms | "
            f"{status}"
        )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    total = len(results)

    correct_count = sum(
        result["correct"]
        for result in results
    )

    incorrect_count = (
        total - correct_count
    )

    accuracy = (
        correct_count / total
        if total
        else 0.0
    )

    latencies = [
        result["latency_ms"]
        for result in results
    ]

    confidences = [
        result["confidence"]
        for result in results
    ]

    average_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0.0
    )

    average_confidence = (
        sum(confidences)
        / len(confidences)
        if confidences
        else 0.0
    )

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    cat_results = [
        result
        for result in results
        if result["true_label"] == "cat"
    ]

    dog_results = [
        result
        for result in results
        if result["true_label"] == "dog"
    ]

    cat_accuracy = (
        sum(
            result["correct"]
            for result in cat_results
        )
        / len(cat_results)
        if cat_results
        else 0.0
    )

    dog_accuracy = (
        sum(
            result["correct"]
            for result in dog_results
        )
        / len(dog_results)
        if dog_results
        else 0.0
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    confusion_matrix = {
        "actual_cat_predicted_cat": 0,
        "actual_cat_predicted_dog": 0,
        "actual_dog_predicted_cat": 0,
        "actual_dog_predicted_dog": 0,
    }

    for result in results:

        true_label = result[
            "true_label"
        ]

        predicted_label = result[
            "predicted_label"
        ]

        key = (
            f"actual_{true_label}"
            f"_predicted_{predicted_label}"
        )

        if key in confusion_matrix:
            confusion_matrix[key] += 1

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    report = {
        "evaluation": {
            "base_url": BASE_URL,
            "total_images": total,
            "images_per_class": {"cat": cat_count,"dog": dog_count,},
        },
        "metrics": {
            "accuracy": round(
                accuracy,
                4,
            ),
            "correct_predictions": correct_count,
            "incorrect_predictions": incorrect_count,
            "average_latency_ms": round(
                average_latency,
                2,
            ),
            "min_latency_ms": round(
                min(latencies),
                2,
            ),
            "max_latency_ms": round(
                max(latencies),
                2,
            ),
            "average_confidence": round(
                average_confidence,
                4,
            ),
            "cat_accuracy": round(
                cat_accuracy,
                4,
            ),
            "dog_accuracy": round(
                dog_accuracy,
                4,
            ),
        },
        "confusion_matrix": confusion_matrix,
        "results": results,
    }

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("POST-DEPLOYMENT PERFORMANCE SUMMARY")
    print("=" * 70)

    print(
        f"Total images       : {total}"
    )

    print(
        f"Correct predictions: {correct_count}"
    )

    print(
        f"Incorrect predictions: {incorrect_count}"
    )

    print(
        f"Accuracy           : {accuracy:.4f}"
    )

    print(
        f"Cat accuracy       : {cat_accuracy:.4f}"
    )

    print(
        f"Dog accuracy       : {dog_accuracy:.4f}"
    )

    print(
        f"Average confidence: {average_confidence:.4f}"
    )

    print(
        f"Average latency    : "
        f"{average_latency:.2f} ms"
    )

    print(
        f"Min latency        : "
        f"{min(latencies):.2f} ms"
    )

    print(
        f"Max latency        : "
        f"{max(latencies):.2f} ms"
    )

    print()
    print("Confusion matrix:")
    print(
        f"  Actual Cat -> Cat: "
        f"{confusion_matrix['actual_cat_predicted_cat']}"
    )
    print(
        f"  Actual Cat -> Dog: "
        f"{confusion_matrix['actual_cat_predicted_dog']}"
    )
    print(
        f"  Actual Dog -> Cat: "
        f"{confusion_matrix['actual_dog_predicted_cat']}"
    )
    print(
        f"  Actual Dog -> Dog: "
        f"{confusion_matrix['actual_dog_predicted_dog']}"
    )

    print()
    print(
        f"Detailed report saved to:"
    )
    print(
        f"  {RESULTS_FILE}"
    )

    print("=" * 70)
    print("POST-DEPLOYMENT EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()