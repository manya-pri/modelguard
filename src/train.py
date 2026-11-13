from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import preprocess_data, validate_data

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "training_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"


def train_model():
    MODEL_DIR.mkdir(exist_ok=True)

    if not DATA_PATH.exists():
        from src.generate_data import save_data
        save_data()

    df = pd.read_csv(DATA_PATH)
    validate_data(df)

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        joblib.dump(pipeline, MODEL_PATH)
        mlflow.sklearn.log_model(pipeline, "model")

    print(f"Model saved: {MODEL_PATH}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")

    return pipeline, accuracy


if __name__ == "__main__":
    train_model()