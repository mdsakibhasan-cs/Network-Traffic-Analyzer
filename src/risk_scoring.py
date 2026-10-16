"""Risk scoring for suspicious network traffic."""

import numpy as np
import pandas as pd


def calculate_risk_score(
    anomaly_score: pd.Series,
    predicted_label: pd.Series,
) -> pd.DataFrame:
    """Convert anomaly scores and predicted labels into 0-100 risk scores."""

    if len(anomaly_score) != len(predicted_label):
        raise ValueError(
            "Anomaly scores and predicted labels must have the same length."
        )

    scores = pd.to_numeric(anomaly_score, errors="coerce")

    if scores.isna().any():
        raise ValueError("Anomaly scores must contain valid numeric values.")

    labels = predicted_label.astype(str).str.strip()

    # Normalize anomaly scores to a 0-100 range.
    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        anomaly_component = pd.Series(
            np.zeros(len(scores)),
            index=scores.index,
        )
    else:
        anomaly_component = (
            (scores - min_score)
            / (max_score - min_score)
            * 100
        )

    # Known attack classifications receive additional risk weight.
    attack_component = labels.str.upper().ne("BENIGN").astype(float) * 40

    risk_score = (
        anomaly_component * 0.6
        + attack_component
    ).clip(0, 100)

    risk_level = pd.cut(
        risk_score,
        bins=[-1, 30, 70, 100],
        labels=["Low", "Medium", "High"],
    )

    return pd.DataFrame(
        {
            "risk_score": risk_score.round(2),
            "risk_level": risk_level.astype(str),
        },
        index=anomaly_score.index,
    )
