"""Guard train-only preprocessing and seeded baseline reproducibility.

Uses tiny simulated telemetry; does not measure production model quality.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from netops_ai.supervised_split import chronological_train_validation_test_split


def test_scaler_fits_training_window_only_and_predictions_repeat():
    rows = [
        {
            "timestamp": f"2026-01-01T{hour:02d}:00:00Z",
            "latency_ms": float(hour if hour < 7 else 1000 + hour),
            "incident": hour % 2,
        }
        for hour in range(12)
    ]
    source = pd.DataFrame(rows)
    train, validation, test = chronological_train_validation_test_split(source)
    features = ["latency_ms"]

    def fit_baseline():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42, max_iter=1000)),
        ])
        pipeline.fit(train[features], train["incident"])
        return pipeline

    first = fit_baseline()
    second = fit_baseline()
    expected_mean = train["latency_ms"].mean()
    assert np.isclose(first.named_steps["scaler"].mean_[0], expected_mean)
    assert not np.isclose(expected_mean, source["latency_ms"].mean())
    np.testing.assert_allclose(
        first.predict_proba(validation[features]),
        second.predict_proba(validation[features]),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        first.predict_proba(test[features]),
        second.predict_proba(test[features]),
        rtol=0,
        atol=1e-12,
    )
