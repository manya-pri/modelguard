from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = list(load_breast_cancer().feature_names)
TARGET = "target"
ARTIFACTS = Path(os.getenv("MODEL_GUARD_ARTIFACTS", "/tmp/modelguard" if os.getenv("VERCEL") else "artifacts"))
REFERENCE_PATH = ARTIFACTS / "reference.csv"
FEEDBACK_PATH = ARTIFACTS / "feedback.csv"
ACTIVE_PATH = ARTIFACTS / "active.json"


def dataset() -> tuple[pd.DataFrame, pd.Series]:
    bunch = load_breast_cancer(as_frame=True)
    return bunch.data, bunch.target


def make_model() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("classifier", LogisticRegression(max_iter=2000, random_state=42))])


def _version_path(version: str) -> Path:
    return ARTIFACTS / f"model-{version}.joblib"


def train_and_save() -> dict[str, Any]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    x, y = dataset()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.25, stratify=y, random_state=42)
    model = make_model().fit(x_train, y_train)
    accuracy = float(accuracy_score(y_val, model.predict(x_val)))
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    joblib.dump(model, _version_path(version))
    x.to_csv(REFERENCE_PATH, index=False)
    info = {"version": version, "accuracy": accuracy, "created_at": datetime.now(timezone.utc).isoformat()}
    ACTIVE_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    try:
        import mlflow
        mlflow.set_experiment("modelguard")
        with mlflow.start_run(run_name=f"train-{version}"):
            mlflow.log_metric("validation_accuracy", accuracy)
            mlflow.log_param("model", "logistic_regression")
            mlflow.log_param("version", version)
    except Exception:
        pass  # MLflow is optional at runtime; model artifacts remain usable.
    return info


def active() -> tuple[Any, dict[str, Any]]:
    if not ACTIVE_PATH.exists():
        train_and_save()
    info = json.loads(ACTIVE_PATH.read_text(encoding="utf-8"))
    return joblib.load(_version_path(info["version"])), info


def append_feedback(features: dict[str, float], label: int) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    row = {name: float(features[name]) for name in FEATURES}
    row[TARGET] = int(label)
    row["recorded_at"] = datetime.now(timezone.utc).isoformat()
    pd.DataFrame([row]).to_csv(FEEDBACK_PATH, mode="a", header=not FEEDBACK_PATH.exists(), index=False)


def _drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    # Prefer Evidently where its installed API is compatible; retain a transparent PSI fallback.
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        report = Report([DataDriftPreset()])
        result = report.run(reference_data=reference, current_data=current)
        return {"method": "evidently", "summary": result.dict()}
    except Exception:
        values = {}
        for col in FEATURES:
            ref = reference[col].dropna().to_numpy()
            cur = current[col].dropna().to_numpy()
            if not len(cur) or not len(ref):
                continue
            edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
            if len(edges) < 3:
                continue
            edges[0], edges[-1] = -np.inf, np.inf
            a = np.histogram(ref, bins=edges)[0] / len(ref)
            b = np.histogram(cur, bins=edges)[0] / len(cur)
            eps = 1e-6
            psi = float(np.sum((b-a) * np.log((b+eps)/(a+eps))))
            values[col] = {"psi": psi, "drifted": psi >= .2}
        drifted = [k for k, v in values.items() if v["drifted"]]
        return {"method": "psi_fallback", "drifted_features": drifted, "feature_scores": values}


def monitor(min_rows: int = 20, retrain: bool = True) -> dict[str, Any]:
    model, info = active()
    if not FEEDBACK_PATH.exists():
        return {"status": "insufficient_data", "required_rows": min_rows, "observed_rows": 0, "active_model": info}
    feedback = pd.read_csv(FEEDBACK_PATH)
    if len(feedback) < min_rows:
        return {"status": "insufficient_data", "required_rows": min_rows, "observed_rows": len(feedback), "active_model": info}
    current = feedback.tail(max(min_rows, 200))
    reference = pd.read_csv(REFERENCE_PATH)
    accuracy = float(accuracy_score(current[TARGET], model.predict(current[FEATURES])))
    drift = _drift(reference[FEATURES], current[FEATURES])
    drift_count = len(drift.get("drifted_features", []))
    should_retrain = accuracy < .90 or drift_count >= 3
    result: dict[str, Any] = {"status": "ok", "observed_rows": len(feedback), "window_rows": len(current),
        "accuracy": accuracy, "drift": drift, "retraining_triggered": False, "active_model": info}
    if should_retrain and retrain:
        x_train, x_val, y_train, y_val = train_test_split(feedback[FEATURES], feedback[TARGET], test_size=.25,
                                                         stratify=feedback[TARGET], random_state=42)
        candidate = make_model().fit(x_train, y_train)
        score = float(accuracy_score(y_val, candidate.predict(x_val)))
        result["candidate_accuracy"] = score
        if score >= info["accuracy"]:
            version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            joblib.dump(candidate, _version_path(version))
            promoted = {"version": version, "accuracy": score, "created_at": datetime.now(timezone.utc).isoformat()}
            ACTIVE_PATH.write_text(json.dumps(promoted, indent=2), encoding="utf-8")
            result.update(status="candidate_promoted", active_model=promoted, retraining_triggered=True)
        else:
            result["status"] = "candidate_rejected"
    return result
