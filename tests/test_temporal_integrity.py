"""Regression checks for chronological isolation in the synthetic anomaly baseline."""

import numpy as np

from netops_ai.analytics import FEATURES, fit_evaluate, generate_data


def _boundary(df):
    timestamps = sorted(df["timestamp"].unique())
    return timestamps[int(0.7 * len(timestamps))]


def test_holdout_is_later_than_every_training_timestamp():
    df = generate_data(sites=3, hours=60, seed=17)
    boundary = _boundary(df)
    _, holdout, metrics = fit_evaluate(df)

    training_normals = df.loc[(df["timestamp"] < boundary) & (df["incident"] == 0)]
    assert not training_normals.empty
    assert training_normals["timestamp"].max() < holdout["timestamp"].min()
    assert holdout["timestamp"].min() == boundary
    assert metrics["train_rows"] == len(training_normals)
    assert metrics["test_rows"] == 3 * (60 - 42)


def test_preprocessing_uses_only_normal_training_observations():
    df = generate_data(sites=3, hours=60, seed=17)
    boundary = _boundary(df)
    training_normals = df.loc[(df["timestamp"] < boundary) & (df["incident"] == 0)]
    model, _, _ = fit_evaluate(df)

    np.testing.assert_allclose(
        model.named_steps["simpleimputer"].statistics_,
        training_normals[FEATURES].median().to_numpy(),
    )
    np.testing.assert_allclose(
        model.named_steps["standardscaler"].mean_,
        training_normals[FEATURES].mean().to_numpy(),
    )


def test_changing_future_rows_cannot_change_fitted_preprocessing():
    original = generate_data(sites=3, hours=60, seed=17)
    modified = original.copy()
    future_mask = modified["timestamp"] >= _boundary(modified)
    modified.loc[future_mask, FEATURES] = 1_000_000.0
    modified.loc[future_mask, "incident"] = 1

    baseline_model, _, baseline_metrics = fit_evaluate(original)
    modified_model, _, modified_metrics = fit_evaluate(modified)
    assert baseline_metrics["train_rows"] == modified_metrics["train_rows"]
    for step, attribute in [("simpleimputer", "statistics_"), ("standardscaler", "mean_")]:
        np.testing.assert_array_equal(
            getattr(baseline_model.named_steps[step], attribute),
            getattr(modified_model.named_steps[step], attribute),
        )
    np.testing.assert_array_equal(
        baseline_model.named_steps["isolationforest"].estimators_[0].tree_.threshold,
        modified_model.named_steps["isolationforest"].estimators_[0].tree_.threshold,
    )
