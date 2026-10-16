import numpy as np
import pandas as pd
import pytest

from src.classification import (
    prepare_classification_features,
    train_traffic_classifier,
    predict_traffic,
)


def make_sample_data():
    features = pd.DataFrame({
        "Flow Duration": [100, 105, 98, 50000, 51000, 49000],
        "Total Fwd Packets": [10, 11, 9, 5000, 5100, 4900],
        "Total Backward Packets": [8, 9, 7, 4000, 4100, 3900],
        "Total Length of Fwd Packets": [
            1000, 1100, 900, 500000, 510000, 490000
        ],
        "Total Length of Bwd Packets": [
            800, 850, 750, 400000, 410000, 390000
        ],
    })

    labels = pd.Series([
        "BENIGN",
        "BENIGN",
        "BENIGN",
        "DDoS",
        "DDoS",
        "DDoS",
    ])

    return features, labels


def test_prepare_classification_features():
    features, _ = make_sample_data()

    result = prepare_classification_features(features)

    assert result.shape == features.shape
    assert result.isna().sum().sum() == 0


def test_prepare_features_replaces_infinity():
    features = pd.DataFrame({
        "Feature A": [1, 2, np.inf],
        "Feature B": [10, -np.inf, 30],
    })

    result = prepare_classification_features(features)

    assert result.shape == (3, 2)
    assert result.isna().sum().sum() == 2


def test_train_classifier():
    features, labels = make_sample_data()

    model = train_traffic_classifier(
        features,
        labels,
        random_state=42,
    )

    assert hasattr(model, "predict")


def test_predict_traffic():
    features, labels = make_sample_data()

    model = train_traffic_classifier(
        features,
        labels,
        random_state=42,
    )

    predictions = predict_traffic(model, features)

    assert len(predictions) == len(features)
    assert predictions.notna().all()
    assert set(predictions.unique()).issubset({"BENIGN", "DDoS"})


def test_mismatched_labels_are_rejected():
    features, labels = make_sample_data()

    with pytest.raises(ValueError):
        train_traffic_classifier(
            features,
            labels.iloc[:3],
        )


def test_single_class_is_rejected():
    features, _ = make_sample_data()

    labels = pd.Series([
        "BENIGN",
        "BENIGN",
        "BENIGN",
        "BENIGN",
        "BENIGN",
        "BENIGN",
    ])

    with pytest.raises(ValueError):
        train_traffic_classifier(features, labels)
