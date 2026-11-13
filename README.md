# ModelGuard

ModelGuard is an intelligent machine learning
monitoring and auto-retraining system.

## Features

- Customer churn prediction
- REST API using FastAPI
- Machine learning using Random Forest
- Model performance tracking
- Data drift detection
- Automatic model retraining
- Model comparison
- MLflow experiment tracking
- Interactive Streamlit dashboard

## Technology Stack

- Python
- Scikit-learn
- FastAPI
- MLflow
- Evidently AI
- Streamlit
- Pandas
- Plotly
- Vercel

## Run the Project

Install dependencies:

pip install -r requirements.txt

Train the model:

python -m src.train

Start API:

uvicorn app.main:app --reload

Open:

http://127.0.0.1:8000/docs

Run dashboard:

streamlit run src/dashboard.py

Run data drift detection manually:

python -m src.drift_detection

Drift detection compares `data/training_data.csv` with feature columns in
`reports/predictions.csv`. It writes `reports/drift.json` and
`reports/drift_report.html`; the dashboard refreshes these reports when it
runs. At least two non-missing observations per feature are required.

Run tests:

pytest -v