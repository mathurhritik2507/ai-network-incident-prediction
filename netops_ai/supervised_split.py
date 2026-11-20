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


def chronological_train_validation_test_split(
    df: pd.DataFrame, validation_fraction: float = 0.20, test_fraction: float = 0.20
):
    """Return disjoint, increasing train/validation/test windows by unique time.

    Fractions refer to distinct timestamps, not rows. Multiple sites observed
    at the same instant stay together. At least one timestamp goes in each
    window, even for small synthetic fixtures.
    """
    if not (
        0 < validation_fraction < 1
        and 0 < test_fraction < 1
        and validation_fraction + test_fraction < 1
    ):
        raise ValueError("validation_fraction and test_fraction must be positive and sum to < 1")
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required")

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    if data["timestamp"].isna().any():
        raise ValueError("timestamp contains missing values")

    times = sorted(data["timestamp"].unique())
    n_times = len(times)
    if n_times < 3:
        raise ValueError("at least three distinct timestamps are required")

    train_end = max(1, min(n_times - 2, int(n_times * (1 - validation_fraction - test_fraction))))
    validation_end = max(train_end + 1, min(n_times - 1, int(n_times * (1 - test_fraction))))
    train = data.loc[data["timestamp"] < times[train_end]]
    validation = data.loc[
        (data["timestamp"] >= times[train_end]) & (data["timestamp"] < times[validation_end])
    ]
    test = data.loc[data["timestamp"] >= times[validation_end]]
    return tuple(part.sort_values("timestamp").reset_index(drop=True) for part in (train, validation, test))
