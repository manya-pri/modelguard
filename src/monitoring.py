import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
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