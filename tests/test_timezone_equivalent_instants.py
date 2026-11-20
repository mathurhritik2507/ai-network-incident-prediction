"""Regression checks for UTC-equivalent timestamps in synthetic multi-site data."""

import pandas as pd

from netops_ai.supervised_split import chronological_train_validation_test_split


def test_equivalent_timezone_instants_stay_in_same_window_and_input_is_unchanged():
    # Each instant is represented twice using different UTC offsets.
    source = pd.DataFrame(
        [
            {
                "timestamp": (f"2026-01-01T{hour:02d}:00:00+00:00"
                              if site == "A" else
                              f"2025-12-31T{(hour + 19) % 24:02d}:00:00-05:00"),
                "site_id": site,
                "latency_ms": hour * 10 + (site == "B"),
                "incident": int(hour in (2, 8, 10)),
            }
            for hour in range(12)
            for site in ("A", "B")
        ]
    )
    # For site B, 00:00Z is 19:00-05:00 on the prior day,
    # while 05:00Z and later are on January 1 in that offset.
    source.loc[source.site_id == "B", "timestamp"] = [
        (pd.Timestamp(f"2026-01-01T{hour:02d}:00:00Z")
         .tz_convert("America/New_York").isoformat())
        for hour in range(12)
    ]
    original = source.copy(deep=True)

    train, validation, test = chronological_train_validation_test_split(source)
    pd.testing.assert_frame_equal(source, original)
    assert [len(part) for part in (train, validation, test)] == [14, 4, 6]
    assert train.timestamp.max() < validation.timestamp.min()
    assert validation.timestamp.max() < test.timestamp.min()
    for part in (train, validation, test):
        assert part.groupby("timestamp").size().eq(2).all()
        assert set(part.site_id) == {"A", "B"}
