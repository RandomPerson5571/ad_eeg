"""Tests for feature selection pipeline."""

import numpy as np
import pandas as pd
import pytest

from eeg.feature_selection import select_features


def _synthetic_df(n=60, n_features=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    # Make two highly correlated columns
    X[:, 1] = X[:, 0] + rng.normal(0, 0.01, n)
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["participant_id"] = [f"sub-{i:03d}" for i in range(n)]
    df["label"] = rng.choice(["A", "F", "C"], size=n)
    return df


def test_select_features_drops_correlated():
    df = _synthetic_df()
    cols = [c for c in df.columns if c.startswith("f")]
    result = select_features(df, cols, config={"seed": 42, "feature_selection": {"correlation_threshold": 0.95}})
    assert result.n_after_correlation < result.n_after_variance
    assert len(result.selected_columns) >= 1
    assert "mi_score" in result.importance.columns


def test_select_features_top_k():
    df = _synthetic_df(n_features=8)
    cols = [c for c in df.columns if c.startswith("f")]
    result = select_features(
        df, cols, config={"seed": 42, "feature_selection": {"top_k": 3, "correlation_threshold": 0.99}}
    )
    assert len(result.selected_columns) == 3
