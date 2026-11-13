from src.generate_data import generate_data
from src.preprocessing import preprocess_data, validate_data


def test_data_generation():
    df = generate_data(n_samples=100)
    assert len(df) == 100
    assert "target" in df.columns


def test_preprocessing():
    df = generate_data(n_samples=100)
    validate_data(df)

    X, y = preprocess_data(df)

    assert X.shape[1] == 6
    assert len(X) == len(y)