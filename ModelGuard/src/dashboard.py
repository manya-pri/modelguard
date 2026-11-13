import streamlit as st
import pandas as pd
import plotly.express as px

from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

from src.train import generate_dataset


st.set_page_config(
    page_title="ModelGuard Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ ModelGuard")
st.subheader("ML Model Monitoring Dashboard")

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "churn.csv"

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    df = generate_dataset()

st.markdown("### Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Total Features", len(df.columns) - 1)
col3.metric(
    "Churn Rate",
    f"{df['churn'].mean() * 100:.2f}%"
)

st.divider()

st.markdown("### Churn Distribution")

fig = px.pie(
    df,
    names="churn",
    title="Customer Churn Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.markdown("### Feature Analysis")

feature = st.selectbox(
    "Select Feature",
    ["tenure", "monthly_charges", "total_charges"]
)

fig2 = px.histogram(
    df,
    x=feature,
    color="churn",
    title=f"{feature} Distribution"
)

st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.markdown("### Data Drift Monitoring")

reference = df.sample(
    frac=0.5,
    random_state=42
)

current = df.sample(
    frac=0.5,
    random_state=24
)

drift_results = []

for column in [
    "tenure",
    "monthly_charges",
    "total_charges"
]:

    ref_mean = reference[column].mean()
    cur_mean = current[column].mean()

    drift = abs(cur_mean - ref_mean) / (
        abs(ref_mean) + 1e-8
    )

    drift_results.append({
        "Feature": column,
        "Reference Mean": round(ref_mean, 2),
        "Current Mean": round(cur_mean, 2),
        "Relative Difference": round(drift, 4),
        "Drift Detected": drift > 0.20
    })

drift_df = pd.DataFrame(drift_results)

st.dataframe(
    drift_df,
    use_container_width=True
)

drift_count = int(
    drift_df["Drift Detected"].sum()
)

if drift_count > 0:
    st.warning(
        f"Potential drift detected in {drift_count} features."
    )
else:
    st.success("No significant drift detected.")

st.caption(
    "ModelGuard | ML Monitoring and Auto-Retraining System"
)