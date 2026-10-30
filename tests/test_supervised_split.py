"""Offline tests for chronological supervised evaluation using synthetic rows."""

import pandas as pd
import pytest

from netops_ai.supervised_split import chronological_split


def _sample():
    rows = [
        {"timestamp": f"2026-01-01T{hour:02d}:00:00Z", "site_id": site,
         "latency_ms": hour * 10, "incident": int(hour in (2, 7))}
        for hour in range(8) for site in ("A", "B")
    ]
    return pd.DataFrame(rows).sample(frac=1, random_state=23).reset_index(drop=True)


def test_holdout_is_strictly_later_with_no_shared_site_timestamps():
    train, holdout = chronological_split(_sample())
    assert len(train) == 12 and len(holdout) == 4
    assert train.timestamp.max() < holdout.timestamp.min()
    assert set(train.timestamp).isdisjoint(set(holdout.timestamp))
    assert set(train.site_id) == set(holdout.site_id) == {"A", "B"}


def test_future_changes_do_not_change_training_rows():
    original = _sample()
    train, holdout = chronological_split(original)
    modified = original.copy()
    boundary = holdout.timestamp.min()
    future_mask = pd.to_datetime(modified.timestamp, utc=True) >= boundary
    modified.loc[future_mask, "latency_ms"] = 1_000_000
    changed_train, changed_holdout = chronological_split(modified)
    pd.testing.assert_frame_equal(train, changed_train)
    assert changed_holdout.latency_ms.max() == 1_000_000


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_invalid_fraction_rejected(fraction):
    with pytest.raises(ValueError, match="test_fraction"):
        chronological_split(_sample(), test_fraction=fraction)


def test_missing_or_single_timestamp_rejected():
    with pytest.raises(ValueError, match="missing"):
        chronological_split(pd.DataFrame({"timestamp": [None, "2026-01-01"]}))
    with pytest.raises(ValueError, match="two distinct"):
        chronological_split(pd.DataFrame({"timestamp": ["2026-01-01", "2026-01-01"]}))
