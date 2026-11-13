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

    response = client.post(
        "/api/predict",
        json={"features": [1, 2, 3, 4, 5, 6]}
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 1