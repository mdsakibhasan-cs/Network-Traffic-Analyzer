import numpy as np
import pandas as pd

from src.preprocessing import (
    normalize_column_names,
    normalize_labels,
    split_features_and_labels,
)


def test_normalize_column_names():
    df = pd.DataFrame({" Flow Duration ": [100], " Label ": ["BENIGN"]})
    result = normalize_column_names(df)

    assert result.columns.tolist() == ["Flow Duration", "Label"]


def test_normalize_labels():
    labels = pd.Series([" BENIGN ", "Web Attack \ufffd Brute Force"])
    result = normalize_labels(labels)

    assert result.tolist() == ["BENIGN", "Web Attack - Brute Force"]


def test_split_features_and_labels():
    df = pd.DataFrame({
        " Flow Duration ": ["100", "200"],
        " Total Fwd Packets": ["5", "10"],
        " Label ": ["BENIGN", "DDoS"],
    })

    features, labels = split_features_and_labels(df)

    assert features.columns.tolist() == [
        "Flow Duration", "Total Fwd Packets"
    ]
    assert labels.tolist() == ["BENIGN", "DDoS"]
    assert all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in features.dtypes
    )


def test_invalid_feature_values_become_missing():
    df = pd.DataFrame({
        "Feature": ["10", "invalid", "inf"],
        "Label": ["BENIGN", "DDoS", "BENIGN"],
    })

    features, _ = split_features_and_labels(df)

    assert features["Feature"].iloc[0] == 10
    assert np.isnan(features["Feature"].iloc[1])
    assert np.isnan(features["Feature"].iloc[2])
