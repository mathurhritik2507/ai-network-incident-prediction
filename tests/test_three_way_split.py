"""Offline tests for time-isolated model selection on simulated telemetry."""

import pandas as pd
import pytest

from netops_ai.supervised_split import chronological_train_validation_test_split


def _multi_site_rows():
    rows = [
        {"timestamp": f"2026-01-01T{hour:02d}:00:00Z", "site_id": site,
         "latency_ms": hour * 10, "incident": int(hour in (3, 9))}
        for hour in range(12) for site in ("A", "B")
    ]
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


def test_three_windows_strictly_ordered_and_complete():
    source = _multi_site_rows()
    train, validation, test = chronological_train_validation_test_split(source)
    assert [len(part) for part in (train, validation, test)] == [14, 4, 6]
    assert train.timestamp.max() < validation.timestamp.min() < test.timestamp.min()
    assert validation.timestamp.max() < test.timestamp.min()
    assert set(train.timestamp).isdisjoint(validation.timestamp)
    assert set(validation.timestamp).isdisjoint(test.timestamp)
    assert set(train.site_id) == set(validation.site_id) == set(test.site_id) == {"A", "B"}
    assert sum(map(len, (train, validation, test))) == len(source)


def test_future_test_feature_edits_cannot_change_earlier_windows():
    source = _multi_site_rows()
    train, validation, test = chronological_train_validation_test_split(source)
    changed = source.copy()
    changed.loc[pd.to_datetime(changed.timestamp, utc=True) >= test.timestamp.min(), "latency_ms"] = 999999
    next_train, next_validation, next_test = chronological_train_validation_test_split(changed)
    pd.testing.assert_frame_equal(train, next_train)
    pd.testing.assert_frame_equal(validation, next_validation)
    assert next_test.latency_ms.min() == 999999


@pytest.mark.parametrize("validation,test", [(0, 0.2), (0.2, 0), (-0.1, 0.2), (0.5, 0.5), (0.9, 0.2), (float('nan'), 0.2)])
def test_invalid_fractions_fail(validation, test):
    with pytest.raises(ValueError, match="fraction"):
        chronological_train_validation_test_split(_multi_site_rows(), validation, test)


def test_invalid_timestamps_fail():
    with pytest.raises(ValueError, match="timestamp column"):
        chronological_train_validation_test_split(pd.DataFrame({"incident": [0, 1, 0]}))
    with pytest.raises(ValueError, match="missing"):
        chronological_train_validation_test_split(pd.DataFrame({"timestamp": [None, "2026-01-01", "2026-01-02"]}))
    with pytest.raises(ValueError, match="three distinct"):
        chronological_train_validation_test_split(pd.DataFrame({"timestamp": ["2026-01-01", "2026-01-02"]}))


def test_small_three_timestamp_fixture_still_has_three_windows():
    source = pd.DataFrame({"timestamp": ["2026-01-03", "2026-01-01", "2026-01-02"]})
    parts = chronological_train_validation_test_split(source, validation_fraction=0.01, test_fraction=0.8)
    assert [len(part) for part in parts] == [1, 1, 1]
