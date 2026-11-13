from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run python -m src.train first."
        )
    return joblib.load(MODEL_PATH)


def predict(features):
    if len(features) != 6:
        raise ValueError("Exactly 6 features are required.")

    model = load_model()
    values = np.asarray(features, dtype=float).reshape(1, -1)

    prediction = int(model.predict(values)[0])
    probability = float(model.predict_proba(values)[0][1])

    return {
        "prediction": prediction,
        "probability": probability,
        "status": "positive" if prediction == 1 else "negative"
    }