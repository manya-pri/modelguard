# ModelGuard

ModelGuard is a small, runnable MLOps demo for tabular classification. It trains a baseline model, serves predictions, records labeled production data, monitors feature drift and model quality, and promotes a retrained candidate only when it beats the current model.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m modelguard.train
uvicorn modelguard.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API. In another terminal run `streamlit run dashboard.py` for the dashboard. Use `/docs` to submit predictions and labeled observations. The included demo dataset is scikit-learn Breast Cancer Wisconsin (Diagnostic); it is for demonstration, not clinical use.

## Flow

Training saves a versioned scikit-learn pipeline and a reference feature profile. `/predict` returns a prediction and confidence. `/feedback` stores a feature row and its true label. `/monitor` compares recent labeled observations with the reference data, reports feature drift and accuracy, and triggers retraining when drift or accuracy thresholds are crossed. A candidate is promoted only if its validation accuracy is no worse than the active model. Artifacts and the active version live under `artifacts/`; MLflow runs are logged locally when enabled.

Set `MODEL_GUARD_ARTIFACTS` to change the artifact directory and `MLFLOW_TRACKING_URI` to configure MLflow. Monitoring requires enough feedback records; it will report insufficient data until then.
