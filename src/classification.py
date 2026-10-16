"""Supervised traffic classification using Random Forest."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.preprocessing import normalize_column_names


def prepare_classification_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare numeric features for supervised classification."""
    result = normalize_column_names(features)

    result = result.apply(pd.to_numeric, errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)

    result = result.dropna(axis=1, how="all")

    if result.empty or result.shape[1] == 0:
        raise ValueError("No usable numeric features were provided.")

    if len(result) < 2:
        raise ValueError("At least two records are required.")

    return result


def train_traffic_classifier(
    features: pd.DataFrame,
    labels: pd.Series,
    random_state: int = 42,
) -> Pipeline:
    """Train a Random Forest classifier for network traffic labels."""
    numeric_features = prepare_classification_features(features)

    if len(labels) != len(numeric_features):
        raise ValueError("Features and labels must have the same number of rows.")

    if labels.nunique() < 2:
        raise ValueError("At least two classes are required.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    model.fit(numeric_features, labels.astype(str).str.strip())

    return model


def predict_traffic(
    model: Pipeline,
    features: pd.DataFrame,
) -> pd.Series:
    """Predict traffic labels using a trained classifier."""
    numeric_features = prepare_classification_features(features)

    expected_columns = list(model.feature_names_in_)
    numeric_features = numeric_features.reindex(columns=expected_columns)

    predictions = model.predict(numeric_features)

    return pd.Series(predictions, index=features.index, name="predicted_label")
