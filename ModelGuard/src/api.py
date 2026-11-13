from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    tenure: int = Field(ge=0, le=100)
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)

    contract_type: str
    internet_service: str
    payment_method: str

    paperless_billing: int = Field(ge=0, le=1)
    tech_support: int = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    prediction: int
    churn_status: str
    probability: float