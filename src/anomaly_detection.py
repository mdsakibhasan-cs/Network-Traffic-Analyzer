"""Unsupervised anomaly detection for network traffic features."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.preprocessing import normalize_column_names


def prepare_numeric_features(features: pd.DataFrame) -> pd.DataFrame:
    """Clean feature names and prepare numeric values for detection."""
    result = normalize_column_names(features)
    result = result.apply(pd.to_numeric, errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)

    # Remove columns that contain no usable numeric values.
    result = result.dropna(axis=1, how="all")

    if result.empty or result.shape[1] == 0:
        raise ValueError("No usable numeric features were provided.")

    if len(result) < 2:
        raise ValueError("At least two records are required.")

    return result


def train_anomaly_detector(
    features: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> Pipeline:
    """Train an Isolation Forest model on numeric traffic features."""
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be greater than 0 and at most 0.5.")

    numeric_features = prepare_numeric_features(features)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("detector", IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    model.fit(numeric_features)
    return model


def detect_anomalies(
    model: Pipeline,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Return anomaly predictions and scores for network traffic records."""
    numeric_features = prepare_numeric_features(features)

    # Align incoming features to the columns used during training.
    expected_columns = list(model.feature_names_in_)
    numeric_features = numeric_features.reindex(columns=expected_columns)

    predictions = model.predict(numeric_features)
    decision_scores = model.decision_function(numeric_features)

    return pd.DataFrame(
        {
            "status": np.where(predictions == -1, "Anomaly", "Normal"),
            "is_anomaly": predictions == -1,
            "anomaly_score": -decision_scores,
        },
        index=features.index,
    )
