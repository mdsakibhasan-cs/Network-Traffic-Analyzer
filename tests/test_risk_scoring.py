import pandas as pd
import pytest

from src.risk_scoring import calculate_risk_score


def test_risk_score_range():
    anomaly_scores = pd.Series([0.0, 0.5, 1.0])
    labels = pd.Series(["BENIGN", "BENIGN", "DDoS"])

    result = calculate_risk_score(anomaly_scores, labels)

    assert len(result) == 3
    assert result["risk_score"].between(0, 100).all()


def test_attack_receives_additional_risk():
    anomaly_scores = pd.Series([0.5, 0.5])
    labels = pd.Series(["BENIGN", "DDoS"])

    result = calculate_risk_score(anomaly_scores, labels)

    assert result.iloc[1]["risk_score"] > result.iloc[0]["risk_score"]


def test_risk_levels_are_valid():
    anomaly_scores = pd.Series([0.0, 0.5, 1.0])
    labels = pd.Series(["BENIGN", "BENIGN", "DDoS"])

    result = calculate_risk_score(anomaly_scores, labels)

    assert set(result["risk_level"]).issubset({
        "Low",
        "Medium",
        "High",
    })


def test_constant_scores_are_supported():
    anomaly_scores = pd.Series([0.5, 0.5, 0.5])
    labels = pd.Series(["BENIGN", "BENIGN", "DDoS"])

    result = calculate_risk_score(anomaly_scores, labels)

    assert result["risk_score"].notna().all()


def test_mismatched_lengths_are_rejected():
    anomaly_scores = pd.Series([0.1, 0.2])
    labels = pd.Series(["BENIGN"])

    with pytest.raises(ValueError):
        calculate_risk_score(anomaly_scores, labels)


def test_invalid_scores_are_rejected():
    anomaly_scores = pd.Series([0.1, "invalid"])
    labels = pd.Series(["BENIGN", "DDoS"])

    with pytest.raises(ValueError):
        calculate_risk_score(anomaly_scores, labels)
