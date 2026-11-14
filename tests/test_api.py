from fastapi.testclient import TestClient

from app.main import app
import app.api as api_module

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction(monkeypatch):
    def fake_predict(features):
        return {
            "prediction": 1,
            "probability": 0.85,
            "status": "positive"
        }

    monkeypatch.setattr(api_module, "predict", fake_predict)
    monkeypatch.setattr(api_module, "log_prediction", lambda *args: None)

    response = client.post(
        "/api/predict",
        json={"features": [1, 2, 3, 4, 5, 6]}
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 1


def test_monitor_without_prediction_history(monkeypatch, tmp_path):
    import src.drift_detection as drift_module
    import src.monitoring as monitoring_module
    from src.generate_data import generate_data

    training_path = tmp_path / "training_data.csv"
    generate_data(n_samples=100).to_csv(training_path, index=False)
    monkeypatch.setattr(monitoring_module, "TRAINING_DATA_PATH", training_path)
    monkeypatch.setattr(monitoring_module, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(drift_module, "REFERENCE_DATA_PATH", training_path)
    monkeypatch.setattr(
        drift_module,
        "PREDICTION_HISTORY_PATH",
        tmp_path / "missing_predictions.csv",
    )
    monkeypatch.setattr(drift_module, "REPORT_DIR", tmp_path / "drift_reports")
    monkeypatch.setattr(
        drift_module,
        "DRIFT_JSON_PATH",
        tmp_path / "drift_reports" / "drift.json",
    )
    monkeypatch.setattr(
        drift_module,
        "DRIFT_HTML_PATH",
        tmp_path / "drift_reports" / "drift_report.html",
    )

    class FakeModel:
        def predict(self, features):
            return [0] * len(features)

    monkeypatch.setattr(monitoring_module, "load_model", FakeModel)

    response = client.post("/api/monitor", params={"min_rows": 4})

    assert response.status_code == 200
    report = response.json()
    assert report["status"] == "insufficient_data"
    assert report["observed_rows"] == 0
    assert report["required_rows"] == 4
    assert 0 <= report["accuracy"] <= 1
    assert report["drift"]["status"] == "insufficient_data"


def test_monitor_with_prediction_history(monkeypatch, tmp_path):
    import src.drift_detection as drift_module
    import src.monitoring as monitoring_module
    from src.generate_data import generate_data

    training_path = tmp_path / "training_data.csv"
    prediction_path = tmp_path / "predictions.csv"
    training_data = generate_data(n_samples=100)
    training_data.to_csv(training_path, index=False)
    history = generate_data(n_samples=20).drop(columns="target")
    history["prediction"] = 0
    history["probability"] = 0.5
    history["timestamp"] = "2026-01-01T00:00:00+00:00"
    history.to_csv(prediction_path, index=False)

    monkeypatch.setattr(monitoring_module, "TRAINING_DATA_PATH", training_path)
    monkeypatch.setattr(monitoring_module, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(drift_module, "REFERENCE_DATA_PATH", training_path)
    monkeypatch.setattr(drift_module, "PREDICTION_HISTORY_PATH", prediction_path)
    monkeypatch.setattr(drift_module, "REPORT_DIR", tmp_path / "drift_reports")
    monkeypatch.setattr(
        drift_module,
        "DRIFT_JSON_PATH",
        tmp_path / "drift_reports" / "drift.json",
    )
    monkeypatch.setattr(
        drift_module,
        "DRIFT_HTML_PATH",
        tmp_path / "drift_reports" / "drift_report.html",
    )

    class FakeModel:
        def predict(self, features):
            return [0] * len(features)

    monkeypatch.setattr(monitoring_module, "load_model", FakeModel)

    response = client.post("/api/monitor", params={"min_rows": 4})

    assert response.status_code == 200
    report = response.json()
    assert report["status"] in {"healthy", "warning"}
    assert report["observed_rows"] == 20
    assert report["drift"]["current_samples"] == 20