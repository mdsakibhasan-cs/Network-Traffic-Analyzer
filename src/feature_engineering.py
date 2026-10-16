"""Create additional features from network traffic measurements."""

import numpy as np
import pandas as pd

from src.preprocessing import normalize_column_names


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add traffic features without modifying existing input columns."""
    result = normalize_column_names(df)

    def numeric_column(name):
        if name not in result.columns:
            return None

        values = pd.to_numeric(result[name], errors="coerce")
        return values.replace([np.inf, -np.inf], np.nan)

    fwd_packets = numeric_column("Total Fwd Packets")
    bwd_packets = numeric_column("Total Backward Packets")

    if fwd_packets is not None and bwd_packets is not None:
        total_packets = fwd_packets + bwd_packets
        result["Total Packets"] = total_packets

        result["Forward Packet Ratio"] = (
            fwd_packets / total_packets.where(total_packets > 0)
        )

    fwd_bytes = numeric_column("Total Length of Fwd Packets")
    bwd_bytes = numeric_column("Total Length of Bwd Packets")

    if fwd_bytes is not None and bwd_bytes is not None:
        total_bytes = fwd_bytes + bwd_bytes
        result["Total Bytes"] = total_bytes

        result["Forward Byte Ratio"] = (
            fwd_bytes / total_bytes.where(total_bytes > 0)
        )

        if "Total Packets" in result.columns:
            packets = result["Total Packets"]

            # Preserve the original dataset's existing feature.
            if "Average Packet Size" not in result.columns:
                result["Average Packet Size"] = (
                    total_bytes / packets.where(packets > 0)
                )

    duration = numeric_column("Flow Duration")

    if duration is not None:
        # CIC-IDS2017 records Flow Duration in microseconds.
        result["Duration Seconds"] = duration / 1_000_000

    return result
