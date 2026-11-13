from src.generate_data import generate_data
from src.drift_detection import DRIFT_HTML_PATH, detect_drift


def test_drift_detection():
    reference = generate_data(n_samples=500)
    current = generate_data(n_samples=500, drift=True)

    result = detect_drift(reference, current)

    assert "drift_detected" in result
    assert "details" in result
    assert result["drift_detected"] is True
    assert all(
        0 <= feature["p_value"] <= 1
        for feature in result["details"].values()
    )


def test_insufficient_data_is_reported():
    reference = generate_data(n_samples=10).iloc[:1]
    current = reference.copy()

    result = detect_drift(reference, current)

    assert result["status"] == "insufficient_data"
    assert result["drift_percentage"] is None
    assert result["affected_features"] == []
    assert DRIFT_HTML_PATH.exists()