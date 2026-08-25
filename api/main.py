from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import io
import logging
import time

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from api.schemas import (
    HealthResponse,
    PredictionResponse,
)
from src.inference import CatDogPredictor


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "best_model.pt"
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("cats-dogs-api")


# ============================================================
# Monitoring Metrics
# ============================================================

metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "prediction_requests": 0,
    "cat_predictions": 0,
    "dog_predictions": 0,
    "total_latency_ms": 0.0,
}


def reset_metrics() -> None:
    """
    Reset in-memory monitoring counters.

    Primarily useful for tests and local development.
    """
    for key in metrics:
        if key == "total_latency_ms":
            metrics[key] = 0.0
        else:
            metrics[key] = 0


def get_metrics() -> dict:
    """
    Return a snapshot of the current application metrics.
    """

    total_requests = metrics["total_requests"]

    average_latency_ms = (
        metrics["total_latency_ms"] / total_requests
        if total_requests > 0
        else 0.0
    )

    return {
        "total_requests": metrics["total_requests"],
        "successful_requests": metrics[
            "successful_requests"
        ],
        "failed_requests": metrics[
            "failed_requests"
        ],
        "prediction_requests": metrics[
            "prediction_requests"
        ],
        "cat_predictions": metrics[
            "cat_predictions"
        ],
        "dog_predictions": metrics[
            "dog_predictions"
        ],
        "average_latency_ms": round(
            average_latency_ms,
            2,
        ),
    }


# ============================================================
# Model
# ============================================================

predictor: CatDogPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    predictor = CatDogPredictor(
        model_path=MODEL_PATH,
        device="cpu",
    )

    logger.info(
        "MODEL_LOADED model=CatsDogsCNN device=cpu"
    )

    yield

    logger.info("MODEL_UNLOADED")

    predictor = None


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="Cats vs Dogs Classification API",
    description=(
        "MLOps Assignment 2 inference service "
        "for binary cat/dog image classification."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Request / Response Monitoring Middleware
# ============================================================

@app.middleware("http")
async def monitoring_middleware(
    request: Request,
    call_next,
):
    """
    Monitor every HTTP request.

    Logged information intentionally excludes:
    - request body
    - uploaded image contents
    - request payload
    - sensitive information
    """

    start_time = time.perf_counter()

    metrics["total_requests"] += 1

    status_code = 500

    logger.info(
        "REQUEST method=%s path=%s",
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)

        status_code = response.status_code

        if status_code < 400:
            metrics["successful_requests"] += 1
        else:
            metrics["failed_requests"] += 1

        return response

    except Exception:
        metrics["failed_requests"] += 1

        logger.exception(
            "REQUEST_ERROR method=%s path=%s",
            request.method,
            request.url.path,
        )

        raise

    finally:
        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        metrics["total_latency_ms"] += elapsed_ms

        logger.info(
            (
                "RESPONSE method=%s path=%s "
                "status=%s latency_ms=%.2f"
            ),
            request.method,
            request.url.path,
            status_code,
            elapsed_ms,
        )


# ============================================================
# Health endpoint
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:

    model_loaded = (
        predictor is not None
    )

    device = (
        str(predictor.device)
        if predictor is not None
        else "unknown"
    )

    return HealthResponse(
        status=(
            "healthy"
            if model_loaded
            else "unhealthy"
        ),
        model_loaded=model_loaded,
        model="CatsDogsCNN",
        device=device,
    )


# ============================================================
# Metrics endpoint
# ============================================================

@app.get("/metrics")
def application_metrics() -> dict:
    """
    Return basic application monitoring metrics.

    Metrics are maintained in memory and therefore represent
    the lifetime of the current API process/pod.
    """

    return get_metrics()


# ============================================================
# Prediction endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
) -> PredictionResponse:

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    metrics["prediction_requests"] += 1

    # --------------------------------------------------------
    # Validate content type
    # --------------------------------------------------------

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="File content type is missing.",
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Supported formats: JPEG, PNG, WebP."
            ),
        )

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    try:
        image = Image.open(
            io.BytesIO(contents)
        )

        # Force image decoding now rather than later.
        image.load()

    except (
        UnidentifiedImageError,
        OSError,
    ) as exc:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        ) from exc

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    try:

        result = predictor.predict(
            image
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Model inference failed.",
        ) from exc

    # --------------------------------------------------------
    # Track prediction result
    # --------------------------------------------------------

    predicted_label = result[
        "predicted_label"
    ]

    if predicted_label == "cat":
        metrics["cat_predictions"] += 1

    elif predicted_label == "dog":
        metrics["dog_predictions"] += 1

    # --------------------------------------------------------
    # Return prediction
    # --------------------------------------------------------

    return PredictionResponse(
        predicted_label=predicted_label,
        class_probabilities=result[
            "class_probabilities"
        ],
        confidence=result[
            "confidence"
        ],
    )