"""Regression checks on the repository's checked-in *synthetic* telemetry.

These are data-contract checks, not evidence of real-world predictive performance.
"""
from pathlib import Path

import pandas as pd

from netops_ai.supervised_split import chronological_train_validation_test_split


DATA = Path(__file__).resolve().parents[1] / "network_kpi_logs.csv"


def test_checked_in_synthetic_data_has_disjoint_chronological_windows():
    source = pd.read_csv(DATA)
    train, validation, test = chronological_train_validation_test_split(source)

    assert len(source) > 0
    assert sum(map(len, (train, validation, test))) == len(source)
    assert train.timestamp.max() < validation.timestamp.min()
    assert validation.timestamp.max() < test.timestamp.min()
    assert set(train.timestamp).isdisjoint(validation.timestamp)
    assert set(train.timestamp).isdisjoint(test.timestamp)
    assert set(validation.timestamp).isdisjoint(test.timestamp)

    # Fail early if a synthetic dataset revision makes model selection undefined.
    assert set(train.incident.unique()) == {0, 1}
    assert set(validation.incident.unique()) == {0, 1}
    assert set(test.incident.unique()).issubset({0, 1})


def test_supervised_features_exclude_timestamp_and_target():
    source = pd.read_csv(DATA)
    train, validation, test = chronological_train_validation_test_split(source)
    expected = set(source.columns) - {"timestamp", "incident"}
    assert expected
    for window in (train, validation, test):
        features = window.drop(columns=["timestamp", "incident"])
        assert set(features.columns) == expected
        assert not features.isna().any().any()
