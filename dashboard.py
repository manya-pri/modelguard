import os
import pandas as pd
import requests
import streamlit as st
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="ModelGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ ModelGuard")
st.caption("Monitor model quality and data drift, then review retraining decisions.")
default_api = os.getenv("API_URL", "http://127.0.0.1:8000")
if default_api and not default_api.startswith(("http://", "https://")):
    default_api = "https://" + default_api
api = st.sidebar.text_input("API URL", default_api)

try:
    health = requests.get(f"{api}/health", timeout=3).json()
    st.success(f"API online · active model {health['active_model']['version']}")
    st.metric("Validation accuracy", f"{health['active_model']['accuracy']:.1%}")
except Exception as exc:
    st.error(f"API unavailable: {exc}")
    st.info("Start the API with `uvicorn modelguard.api:app --reload`.")
    st.stop()

left, right = st.columns([1, 2])
with left:
    st.subheader("Monitor")
    rows = st.number_input("Minimum labeled rows", 4, 1000, 20)
    if st.button("Run drift and performance check", type="primary"):
        try:
            st.session_state.report = requests.post(f"{api}/monitor", params={"min_rows": rows}, timeout=60).json()
        except Exception as exc:
            st.error(str(exc))
    if "report" in st.session_state:
        report = st.session_state.report
        st.write("Status:", report.get("status"))
        if "accuracy" in report:
            st.metric("Recent accuracy", f"{report['accuracy']:.1%}")
            st.write("Retraining triggered:", report["retraining_triggered"])
            drift = report.get("drift", {})
            st.write("Drift method:", drift.get("method"))
            st.write("Drifted features:", drift.get("drifted_features", []))
        else:
            st.write(f"Labeled rows: {report.get('observed_rows', 0)} / {report.get('required_rows')}")

with right:
    st.subheader("Try a prediction")
    data = load_breast_cancer(as_frame=True).data
    sample_id = st.slider("Demo sample", 0, len(data)-1, 0)
    sample = data.iloc[sample_id]
    st.dataframe(sample.to_frame("value"), height=320)
    if st.button("Predict selected sample"):
        response = requests.post(f"{api}/predict", json={"features": sample.to_dict()}, timeout=10)
        if response.ok:
            st.json(response.json())
        else:
            st.error(response.text)

st.divider()
st.caption("Demo data only. Monitoring requires labeled feedback submitted to POST /feedback.")
