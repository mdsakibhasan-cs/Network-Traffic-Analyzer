import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import (
    prepare_numeric_features,
    train_anomaly_detector,
    detect_anomalies,
)


def make_sample_data():
    return pd.DataFrame({
        "Flow Duration": [100, 105, 98, 102, 101, 99, 50000],
        "Total Fwd Packets": [10, 11, 9, 10, 12, 10, 5000],
        "Total Backward Packets": [8, 9, 7, 8, 8, 9, 4000],
        "Total Length of Fwd Packets": [1000, 1100, 900, 1050, 1200, 980, 500000],
        "Total Length of Bwd Packets": [800, 850, 750, 820, 900, 780, 400000],
    })


def test_prepare_numeric_features_removes_infinity():
    df = pd.DataFrame({
        "Feature A": [1, 2, np.inf],
        "Feature B": [10, -np.inf, 30],
    })

    result = prepare_numeric_features(df)

    assert result.shape == (3, 2)
    assert result.isna().sum().sum() == 2


def test_train_anomaly_detector_returns_model():
    df = make_sample_data()

    model = train_anomaly_detector(
        df,
        contamination=1 / 7,
        random_state=42,
    )

    assert hasattr(model, "predict")
    assert hasattr(model, "decision_function")


def test_detect_anomalies_returns_expected_columns():
    df = make_sample_data()

    model = train_anomaly_detector(
        df,
        contamination=1 / 7,
        random_state=42,
    )

    result = detect_anomalies(model, df)

    assert list(result.columns) == [
        "status",
        "is_anomaly",
        "anomaly_score",
    ]
    assert len(result) == len(df)
    assert result["anomaly_score"].notna().all()


def test_extreme_flow_is_detected_as_anomaly():
    df = make_sample_data()

    model = train_anomaly_detector(
        df,
        contamination=1 / 7,
        random_state=42,
    )

    result = detect_anomalies(model, df)

    assert result["is_anomaly"].sum() >= 1
    assert result.iloc[-1]["is_anomaly"]


def test_invalid_contamination_is_rejected():
    df = make_sample_data()

    with pytest.raises(ValueError):
        train_anomaly_detector(df, contamination=0)

    with pytest.raises(ValueError):
        train_anomaly_detector(df, contamination=0.6)
