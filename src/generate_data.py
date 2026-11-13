from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def generate_data(n_samples=2000, random_state=42, drift=False):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        random_state=random_state
    )

    columns = [f"feature_{i}" for i in range(1, 7)]
    df = pd.DataFrame(X, columns=columns)

    if drift:
        df["feature_1"] += 3.0

    df["target"] = y
    return df


def save_data():
    df = generate_data()
    path = DATA_DIR / "training_data.csv"
    df.to_csv(path, index=False)
    print(f"Training data saved: {path}")
    return df


if __name__ == "__main__":
    save_data()