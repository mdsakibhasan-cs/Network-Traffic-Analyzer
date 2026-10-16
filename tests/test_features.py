import numpy as np
import pandas as pd

from src.feature_engineering import add_derived_features


def test_derived_packet_and_byte_features():
    df = pd.DataFrame({
        "Total Fwd Packets": [10],
        "Total Backward Packets": [5],
        "Total Length of Fwd Packets": [70],
        "Total Length of Bwd Packets": [30],
        "Flow Duration": [1_000_000],
    })

    result = add_derived_features(df)

    assert result.loc[0, "Total Packets"] == 15
    assert np.isclose(result.loc[0, "Forward Packet Ratio"], 10 / 15)
    assert result.loc[0, "Total Bytes"] == 100
    assert np.isclose(result.loc[0, "Forward Byte Ratio"], 0.7)
    assert np.isclose(result.loc[0, "Average Packet Size"], 100 / 15)
    assert result.loc[0, "Duration Seconds"] == 1


def test_zero_packets_produce_missing_ratios():
    df = pd.DataFrame({
        "Total Fwd Packets": [0],
        "Total Backward Packets": [0],
        "Total Length of Fwd Packets": [0],
        "Total Length of Bwd Packets": [0],
        "Flow Duration": [0],
    })

    result = add_derived_features(df)

    assert pd.isna(result.loc[0, "Forward Packet Ratio"])
    assert pd.isna(result.loc[0, "Forward Byte Ratio"])
    assert pd.isna(result.loc[0, "Average Packet Size"])


def test_original_dataframe_is_not_modified():
    df = pd.DataFrame({
        "Total Fwd Packets": [4],
        "Total Backward Packets": [6],
    })
    original = df.copy(deep=True)

    add_derived_features(df)

    pd.testing.assert_frame_equal(df, original)


def test_duration_and_label_are_preserved():
    df = pd.DataFrame({
        "Flow Duration": [2_000_000],
        "Label": ["BENIGN"],
    })

    result = add_derived_features(df)

    assert result.loc[0, "Duration Seconds"] == 2
    assert result.loc[0, "Label"] == "BENIGN"
def test_existing_average_packet_size_is_preserved():
    df = pd.DataFrame({
        " Total Fwd Packets": [10],
        " Total Backward Packets": [5],
        "Total Length of Fwd Packets": [70],
        "Total Length of Bwd Packets": [30],
        " Average Packet Size": [9.0],
    })

    result = add_derived_features(df)

    assert result.loc[0, "Average Packet Size"] == 9.0
