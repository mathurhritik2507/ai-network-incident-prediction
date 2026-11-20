"""DST regression: repeated wall-clock hours must remain distinct UTC instants.

Uses synthetic timestamps only; no network or production telemetry.
"""

import pandas as pd

from netops_ai.supervised_split import chronological_train_validation_test_split


def test_dst_fall_back_repeated_hour_is_not_collapsed():
    # 01:30 occurs twice in New York when daylight saving time ends.
    # Their explicit offsets represent distinct UTC instants, one hour apart.
    instants = [
        "2026-11-01T00:30:00-04:00",
        "2026-11-01T01:30:00-04:00",
        "2026-11-01T01:30:00-05:00",
        "2026-11-01T02:30:00-05:00",
        "2026-11-01T03:30:00-05:00",
        "2026-11-01T04:30:00-05:00",
    ]
    source = pd.DataFrame(
        [
            {"timestamp": instant, "site_id": site, "incident": 0}
            for instant in instants
            for site in ("A", "B")
        ]
    ).sample(frac=1, random_state=17).reset_index(drop=True)
    original = source.copy(deep=True)

    train, validation, test = chronological_train_validation_test_split(source)
    pd.testing.assert_frame_equal(source, original)
    assert [len(part) for part in (train, validation, test)] == [6, 2, 4]

    expected = pd.to_datetime(instants, utc=True)
    observed = pd.concat([train, validation, test], ignore_index=True)
    assert observed["timestamp"].nunique() == 6
    assert set(observed["timestamp"]) == set(expected)
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()
    for part in (train, validation, test):
        assert part.groupby("timestamp").size().eq(2).all()
