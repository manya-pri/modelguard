import pandas as pd

FEATURES = [f"feature_{i}" for i in range(1, 7)]
TARGET = "target"


def preprocess_data(df):
    df = df.copy()
    df = df.dropna()

    X = df[FEATURES]
    y = df[TARGET]

    return X, y


def validate_data(df):
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty")

    return True