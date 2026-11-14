from fastapi import APIRouter, HTTPException, Query

from app.schemas import PredictionRequest, PredictionResponse
from src.predict import predict
from src.monitoring import log_prediction, run_monitoring

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


@router.post("/monitor")
def monitor(min_rows: int = Query(default=20, ge=1)):
    try:
        return run_monitoring(min_rows)

    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))