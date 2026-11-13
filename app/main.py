from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api import router

app = FastAPI(
    title="ModelGuard API",
    description="ML Model Monitoring and Automated Retraining System",
    version="1.0.0"
)


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")


app.include_router(router, prefix="/api", tags=["Predictions"])