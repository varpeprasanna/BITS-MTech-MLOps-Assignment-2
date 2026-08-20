from contextlib import asynccontextmanager
from pathlib import Path
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
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

    yield

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
    # Return prediction
    # --------------------------------------------------------

    return PredictionResponse(
        predicted_label=result[
            "predicted_label"
        ],
        class_probabilities=result[
            "class_probabilities"
        ],
        confidence=result[
            "confidence"
        ],
    )