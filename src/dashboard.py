import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.drift_detection import check_saved_drift

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"

st.set_page_config(
    page_title="ModelGuard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ ModelGuard")
st.subheader("ML Model Monitoring & Automated Retraining")

st.markdown(
    "Monitor model health, prediction activity and data drift."
)

report_path = REPORT_DIR / "monitoring.json"
prediction_path = REPORT_DIR / "predictions.csv"
drift_report_path = REPORT_DIR / "drift_report.html"
retraining_path = REPORT_DIR / "retraining.json"

col1, col2, col3 = st.columns(3)

if report_path.exists():
    with open(report_path) as file:
        report = json.load(file)

    col1.metric("Model Accuracy", f"{report['accuracy']:.2%}")
    col2.metric("Samples Evaluated", report["samples"])
    col3.metric("Model Status", report["status"].upper())
else:
    st.info("No monitoring report available yet.")

if prediction_path.exists():
    df = pd.read_csv(prediction_path)
    st.subheader("Prediction History")
    st.dataframe(df, use_container_width=True)

    st.subheader("Prediction Distribution")
    st.bar_chart(df["prediction"].value_counts())

drift = check_saved_drift()
st.subheader("Data Drift")
drift_col1, drift_col2, drift_col3 = st.columns(3)
drift_col1.metric(
    "Drift Status",
    drift["status"].replace("_", " ").title(),
)
drift_percentage = drift["drift_percentage"]
drift_col2.metric(
    "Drift Percentage",
    f"{drift_percentage:.1f}%" if drift_percentage is not None else "N/A",
)
drift_col3.metric("Affected Features", drift["drifted_features"])

affected_features = ", ".join(drift["affected_features"]) or "None"
st.write("Affected features:", affected_features)
if drift["message"]:
    st.info(drift["message"])
if drift_report_path.exists():
    st.download_button(
        "Download drift report",
        data=drift_report_path.read_bytes(),
        file_name=drift_report_path.name,
        mime="text/html",
    )

if retraining_path.exists():
    with open(retraining_path) as file:
        retraining = json.load(file)

    st.subheader("Last Retraining")
    st.json(retraining)