import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.drift_detection import check_saved_drift
from src.predict import load_model
from src.preprocessing import preprocess_data, validate_data

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
TRAINING_DATA_PATH = ROOT / "data" / "training_data.csv"
REPORT_DIR.mkdir(exist_ok=True)


def monitor_predictions(y_true, y_pred):
    accuracy = float(accuracy_score(y_true, y_pred))

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "accuracy": accuracy,
        "samples": len(y_true),
        "status": "healthy" if accuracy >= 0.80 else "warning"
    }

    report_path = REPORT_DIR / "monitoring.json"

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    return report


def log_prediction(features, prediction, probability):
    path = REPORT_DIR / "predictions.csv"

    row = {
        **{
            f"feature_{i + 1}": value
            for i, value in enumerate(features)
        },
        "prediction": prediction,
        "probability": probability,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    df = pd.DataFrame([row])
    df.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False
    )


def run_monitoring(min_rows=20):
    drift = check_saved_drift()

    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAINING_DATA_PATH.name}."
        )

    training_data = pd.read_csv(TRAINING_DATA_PATH)
    validate_data(training_data)
    features, labels = preprocess_data(training_data)
    _, validation_features, _, validation_labels = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = load_model()
    validation_predictions = model.predict(validation_features)
    performance = monitor_predictions(validation_labels, validation_predictions)

    observed_rows = int(drift.get("current_samples", 0))
    if drift.get("status") == "insufficient_data" or observed_rows < min_rows:
        status = "insufficient_data"
        message = drift.get("message") or (
            f"Need at least {min_rows} prediction rows; found {observed_rows}."
        )
    elif drift.get("drift_detected") or performance["status"] == "warning":
        status = "warning"
        message = "Model performance or input data drift needs review."
    else:
        status = "healthy"
        message = "Model performance and input data are within expected ranges."

    return {
        "status": status,
        "message": message,
        "accuracy": performance["accuracy"],
        "samples": performance["samples"],
        "performance_status": performance["status"],
        "observed_rows": observed_rows,
        "required_rows": min_rows,
        "retraining_triggered": False,
        "drift": drift,
    }