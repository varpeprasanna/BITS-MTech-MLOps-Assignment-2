from pydantic import BaseModel, Field


class ClassProbabilities(BaseModel):
    cat: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    dog: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )


class PredictionResponse(BaseModel):
    predicted_label: str
    class_probabilities: ClassProbabilities
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str
    device: str