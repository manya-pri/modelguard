import os
import pandas as pd
import requests
import streamlit as st
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="ModelGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ ModelGuard")
st.caption("Monitor model quality and data drift, then review retraining decisions.")
default_api = os.getenv("API_URL", "https://modelguard-g8lm.onrender.com")
if default_api and not default_api.startswith(("http://", "https://")):
    default_api = "https://" + default_api
api = st.sidebar.text_input("API URL", default_api)
api = api.rstrip("/")

try:
    health_response = requests.get(f"{api}/api/health", timeout=10)
    health_response.raise_for_status()
    health = health_response.json()
    if health.get("status") == "healthy":
        st.success(f"API online · {health['status']}")
    else:
        st.warning(f"API connected · status {health.get('status', 'unknown')}")
    active_model = health.get("active_model", {})
    if "version" in active_model:
        st.caption(f"Active model {active_model['version']}")
    if "accuracy" in active_model:
        st.metric("Validation accuracy", f"{active_model['accuracy']:.1%}")
except Exception as exc:
    st.error(f"API unavailable: {exc}")
    st.info("Check the API URL and confirm the ModelGuard API is running.")
    st.stop()

left, right = st.columns([1, 2])
with left:
    st.subheader("Monitor")
    rows = st.number_input("Minimum prediction rows", 4, 1000, 20)
    if st.button("Run drift and performance check", type="primary"):
        try:
            response = requests.post(
                f"{api}/api/monitor",
                params={"min_rows": rows},
                timeout=60,
            )
            response.raise_for_status()
            st.session_state.report = response.json()
        except Exception as exc:
            st.error(str(exc))
    if "report" in st.session_state:
        report = st.session_state.report
        st.write("Status:", report.get("status"))
        st.write(
            "Prediction rows:",
            f"{report.get('observed_rows', 0)} / {report.get('required_rows')}",
        )
        if "accuracy" in report:
            st.metric("Validation accuracy", f"{report['accuracy']:.1%}")
            st.write("Retraining triggered:", report.get("retraining_triggered", False))
            drift = report.get("drift", {})
            st.write("Drift status:", drift.get("status", "unknown"))
            st.write("Drifted features:", drift.get("affected_features", []))
        if report.get("status") == "insufficient_data":
            st.info(report.get("message", "More prediction history is needed."))

with right:
    st.subheader("Try a prediction")
    data = load_breast_cancer(as_frame=True).data
    sample_id = st.slider("Demo sample", 0, len(data)-1, 0)
    sample = data.iloc[sample_id]
    st.dataframe(sample.to_frame("value"), height=320)
    if st.button("Predict selected sample"):
        try:
            response = requests.post(
                f"{api}/api/predict",
                json={"features": sample.iloc[:6].astype(float).tolist()},
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            st.write("Prediction:", result["status"], f"(class {result['prediction']})")
            st.metric("Positive-class probability", f"{result['probability']:.1%}")
            st.json(result)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()
st.caption("Demo data only. Monitoring compares prediction history with the training data.")
