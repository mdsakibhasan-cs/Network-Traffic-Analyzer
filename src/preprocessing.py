"""Utilities for cleaning CIC-IDS2017 network traffic data."""

import re

import numpy as np
import pandas as pd


def normalize_column_names(df):
    """Remove surrounding whitespace and make column names unique."""
    result = df.copy()
    cleaned_names = []
    seen = {}

    for column in result.columns:
        name = str(column).replace("\ufeff", "").strip()
        name = re.sub(r"\s+", " ", name)

        if not name:
            name = "unnamed"

        count = seen.get(name, 0) + 1
        seen[name] = count

        if count > 1:
            name = f"{name}_{count}"

        cleaned_names.append(name)

    result.columns = cleaned_names
    return result


def normalize_labels(labels):
    """Standardize whitespace and replacement characters in labels."""
    result = labels.astype("string").str.strip()
    result = result.str.replace("\ufffd", "-", regex=False)
    result = result.str.replace(r"\s+", " ", regex=True)
    result = result.str.replace(r"\s*-\s*", " - ", regex=True)
    return result


def split_features_and_labels(df, label_column="Label"):
    """Return numeric feature columns and a normalized label series."""
    result = normalize_column_names(df)

    matches = [
        column
        for column in result.columns
        if column.casefold() == label_column.casefold()
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one '{label_column}' column; "
            f"found {len(matches)}."
        )

    actual_label_column = matches[0]
    labels = normalize_labels(result[actual_label_column])

    features = result.drop(columns=[actual_label_column])
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)

    return features, labels
