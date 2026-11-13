from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Exactly six numerical features"
    )


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    status: str