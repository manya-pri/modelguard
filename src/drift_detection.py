import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
REFERENCE_DATA_PATH = DATA_DIR / "training_data.csv"
PREDICTION_HISTORY_PATH = REPORT_DIR / "predictions.csv"
DRIFT_JSON_PATH = REPORT_DIR / "drift.json"
DRIFT_HTML_PATH = REPORT_DIR / "drift_report.html"
MINIMUM_SAMPLES = 2


def _save_result(result, html_report=None):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with DRIFT_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    if html_report is None:
        message = escape(result["message"])
        status = escape(result["status"].replace("_", " ").title())
        html_report = (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<title>ModelGuard data drift report</title></head><body>"
            f"<h1>Data drift status: {status}</h1><p>{message}</p></body></html>"
        )

    DRIFT_HTML_PATH.write_text(html_report, encoding="utf-8")


def _insufficient_result(message, reference_samples=0, current_samples=0):
    result = {
        "status": "insufficient_data",
        "message": message,
        "drift_detected": False,
        "drift_percentage": None,
        "drifted_features": 0,
        "affected_features": [],
        "feature_count": 0,
        "reference_samples": int(reference_samples),
        "current_samples": int(current_samples),
        "details": {},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_result(result)
    return result


def detect_drift(reference, current):
    if reference is None or current is None:
        return _insufficient_result("Reference or current data is unavailable.")

    if reference.empty or current.empty:
        return _insufficient_result(
            "Reference and current data must both contain rows.",
            len(reference),
            len(current),
        )

    features = [
        column
        for column in reference.columns
        if column.startswith("feature_") and column in current.columns
    ]
    features = [
        feature
        for feature in features
        if reference[feature].notna().sum() >= MINIMUM_SAMPLES
        and current[feature].notna().sum() >= MINIMUM_SAMPLES
    ]
    if not features:
        return _insufficient_result(
            "No shared feature columns have enough non-missing values to compare.",
            len(reference),
            len(current),
        )

    reference_features = reference[features].copy()
    current_features = current[features].copy()
    report = Report(
        metrics=[
            DataDriftPreset(
                columns=features,
                drift_share=max(0.0, 1 / len(features) - 1e-9),
            )
        ]
    )
    report.run(
        reference_data=reference_features,
        current_data=current_features,
    )
    report_data = report.as_dict()
    drift_table = next(
        metric["result"]
        for metric in report_data["metrics"]
        if metric["metric"] == "DataDriftTable"
    )
    column_results = drift_table["drift_by_columns"]

    details = {}
    for feature in features:
        column_result = column_results[feature]
        score = float(column_result["drift_score"])
        stattest_name = str(column_result["stattest_name"])
        details[feature] = {
            "drift_score": score,
            "stattest_name": stattest_name,
            "p_value": score if "p_value" in stattest_name else None,
            "drift_detected": bool(column_result["drift_detected"]),
        }
    affected_features = [
        feature
        for feature, details_for_feature in details.items()
        if details_for_feature["drift_detected"]
    ]
    feature_count = len(features)
    drift_count = len(affected_features)
    result = {
        "status": "drift_detected" if drift_count else "no_drift",
        "message": "",
        "drift_detected": bool(drift_count),
        "drift_percentage": drift_count / feature_count * 100,
        "drifted_features": drift_count,
        "affected_features": affected_features,
        "feature_count": feature_count,
        "reference_samples": int(len(reference_features)),
        "current_samples": int(len(current_features)),
        "details": details,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_result(result, report.get_html())
    return result


def check_saved_drift():
    if not REFERENCE_DATA_PATH.exists():
        return _insufficient_result(
            f"Reference dataset not found: {REFERENCE_DATA_PATH.name}."
        )
    if not PREDICTION_HISTORY_PATH.exists():
        return _insufficient_result(
            f"Prediction history not found: {PREDICTION_HISTORY_PATH.name}."
        )

    try:
        reference = pd.read_csv(REFERENCE_DATA_PATH)
        current = pd.read_csv(PREDICTION_HISTORY_PATH)
    except pd.errors.EmptyDataError:
        return _insufficient_result("Reference dataset or prediction history is empty.")
    return detect_drift(reference, current)


if __name__ == "__main__":
    print(json.dumps(check_saved_drift(), indent=4))
