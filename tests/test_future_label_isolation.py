"""Synthetic regression: future incident labels cannot affect earlier split rows."""
import pandas as pd
from netops_ai.supervised_split import chronological_train_validation_test_split


def test_future_labels_cannot_change_train_or_validation_windows():
    source = pd.DataFrame([
        {"timestamp": f"2026-01-01T{hour:02d}:00:00Z", "site_id": site,
         "latency_ms": hour * 10, "incident": int(hour in (2, 8, 11))}
        for hour in range(15) for site in ("A", "B")
    ]).sample(frac=1, random_state=19).reset_index(drop=True)
    train, validation, test = chronological_train_validation_test_split(source)
    changed = source.copy()
    future = pd.to_datetime(changed["timestamp"], utc=True) >= test["timestamp"].min()
    changed.loc[future, "incident"] = 1 - changed.loc[future, "incident"]
    next_train, next_validation, next_test = chronological_train_validation_test_split(changed)
    pd.testing.assert_frame_equal(train, next_train)
    pd.testing.assert_frame_equal(validation, next_validation)
    assert next_test["incident"].tolist() == (1 - test["incident"]).tolist()
    assert len(train) + len(validation) + len(test) == len(source)
