import os
import joblib
import numpy as np
import pandas as pd
import mlflow

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.preprocessing import create_preprocessor, clean_data


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"


def generate_dataset(n=1000):

    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "tenure": rng.integers(1, 73, n),
        "monthly_charges": rng.uniform(20, 120, n),
        "contract_type": rng.choice(
            ["Month-to-month", "One year", "Two year"], n
        ),
        "internet_service": rng.choice(
            ["DSL", "Fiber optic", "No"], n
        ),
        "payment_method": rng.choice(
            ["Electronic check", "Credit card", "Bank transfer"], n
        ),
        "paperless_billing": rng.integers(0, 2, n),
        "tech_support": rng.integers(0, 2, n),
    })

    df["total_charges"] = (
        df["tenure"] * df["monthly_charges"]
    )

    churn_score = (
        (df["tenure"] < 15).astype(int)
        + (df["contract_type"] == "Month-to-month").astype(int)
        + (df["monthly_charges"] > 85).astype(int)
        + (df["tech_support"] == 0).astype(int)
    )

    probability = 1 / (1 + np.exp(-(churn_score - 2)))

    df["churn"] = rng.binomial(1, probability)

    return df


def train_model():

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = generate_dataset()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    df = clean_data(df)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, zero_division=0)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("Model training completed!")
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Model saved:", MODEL_PATH)

    try:
        mlflow.set_experiment("ModelGuard")

        with mlflow.start_run():

            mlflow.log_param("model", "RandomForest")
            mlflow.log_param("n_estimators", 150)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="churn-model"
            )

        print("MLflow tracking completed.")

    except Exception as error:
        print("MLflow tracking unavailable:", error)

    return accuracy, f1


if __name__ == "__main__":
    train_model()