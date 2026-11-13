import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.train import train_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def retrain_model(reason="Drift or performance degradation"):
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "model.pkl"
    backup_path = MODEL_DIR / "previous_model.pkl"

    if model_path.exists():
        shutil.copy2(model_path, backup_path)

    try:
        model, accuracy = train_model()

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "accuracy": float(accuracy),
            "status": "success"
        }

    except Exception as error:
        if backup_path.exists():
            shutil.copy2(backup_path, model_path)
        raise error

    with open(REPORT_DIR / "retraining.json", "w") as file:
        json.dump(report, file, indent=4)

    return report


if __name__ == "__main__":
    print(retrain_model())