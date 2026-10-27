"""Chronological holdout split for the synthetic supervised incident baseline."""

import pandas as pd


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.25):
    """Split by *unique timestamp* so no observation time crosses the boundary.

    Returns independent, time-ordered train and holdout dataframes. The holdout
    is strictly later than the training period, including for multi-site data.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required")

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    if data["timestamp"].isna().any():
        raise ValueError("timestamp contains missing values")

    unique_times = sorted(data["timestamp"].unique())
    if len(unique_times) < 2:
        raise ValueError("at least two distinct timestamps are required")
    cut = max(1, min(len(unique_times) - 1, int(len(unique_times) * (1 - test_fraction))))
    boundary = unique_times[cut]
    train = data.loc[data["timestamp"] < boundary].sort_values("timestamp").reset_index(drop=True)
    holdout = data.loc[data["timestamp"] >= boundary].sort_values("timestamp").reset_index(drop=True)
    return train, holdout
