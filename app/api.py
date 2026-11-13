from fastapi import APIRouter, HTTPException

from app.schemas import PredictionRequest, PredictionResponse
from src.predict import predict
from src.monitoring import log_prediction

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "healthy", "project": "ModelGuard"}


@router.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest):
    try:
        result = predict(request.features)

        log_prediction(
            request.features,
            result["prediction"],
            result["probability"]
        )

        return result

    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))