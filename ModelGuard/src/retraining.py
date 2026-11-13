import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.train import generate_dataset
from src.preprocessing import create_preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "churn_model.joblib"


def retrain_model():

    df = generate_dataset(1500)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    new_model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    new_model.fit(X_train, y_train)

    new_accuracy = accuracy_score(
        y_test,
        new_model.predict(X_test)
    )

    old_model = joblib.load(MODEL_PATH)

    old_accuracy = accuracy_score(
        y_test,
        old_model.predict(X_test)
    )

    print("Old accuracy:", old_accuracy)
    print("New accuracy:", new_accuracy)

    if new_accuracy > old_accuracy:

        joblib.dump(new_model, MODEL_PATH)

        print("New model promoted successfully!")

    else:

        print("Old model retained.")

    return {
        "old_accuracy": old_accuracy,
        "new_accuracy": new_accuracy,
        "promoted": new_accuracy > old_accuracy
    }


if __name__ == "__main__":
    retrain_model()