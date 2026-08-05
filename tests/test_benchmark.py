"""Tests for benchmark runner."""

import numpy as np
import pandas as pd
import pytest

from eeg.training.benchmark import MODEL_REGISTRY, run_benchmark
from eeg.training.evaluation import bootstrap_ci, compute_benchmark_metrics


def test_model_registry_keys():
    assert "logistic_regression" in MODEL_REGISTRY
    assert "random_forest" in MODEL_REGISTRY
    assert "xgboost" in MODEL_REGISTRY
    assert "mlp" in MODEL_REGISTRY


def test_compute_benchmark_metrics_multiclass():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.4, 0.4, 0.2],
            [0.2, 0.6, 0.2],
            [0.1, 0.7, 0.2],
            [0.1, 0.1, 0.8],
            [0.5, 0.3, 0.2],
        ]
    )
    m = compute_benchmark_metrics(y_true, y_pred, y_proba, labels=[0, 1, 2])
    assert "balanced_accuracy" in m
    assert "mcc" in m
    assert "per_class" in m


def test_bootstrap_ci():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])

    def fn(yt, yp, ypr):
        return float((yt == yp).mean())

    mean, lo, hi = bootstrap_ci(fn, y_true, y_pred, n=50, seed=0)
    assert lo <= mean <= hi


def test_run_benchmark_smoke(tmp_path, monkeypatch):
    """Minimal feature frame with 6 subjects for 3-fold CV."""
    rng = np.random.default_rng(0)
    n = 30
    rows = []
    for sid in range(6):
        for _ in range(5):
            rows.append(
                {
                    "participant_id": f"sub-{sid:03d}",
                    "label": ["A", "F", "C"][sid % 3],
                    "dataset_id": 2,
                    "dataset_name": "eyesclosed",
                    "lzc_posterior": rng.random(),
                    "mse_posterior": rng.random(),
                    "rel_alpha": rng.random(),
                    "rel_beta": rng.random(),
                    "rel_theta": rng.random(),
                    "rel_delta": rng.random(),
                    "alpha_peak_freq": 10.0,
                    "theta_alpha_ratio": rng.random(),
                    "theta_beta_ratio": rng.random(),
                    "slow_fast_ratio": rng.random(),
                    "theta_wpli": rng.random(),
                    "alpha_wpli": rng.random(),
                }
            )
    df = pd.DataFrame(rows)

    import eeg.io as io_mod
    import eeg.paths as paths_mod
    import eeg.training.benchmark as bench_mod

    feat_dir = tmp_path / "features" / "eyesclosed" / "baseline"
    feat_dir.mkdir(parents=True)
    parquet = feat_dir / "subject_features.parquet"
    df.to_parquet(parquet, engine="pyarrow")

    def _results(d, e):
        return tmp_path / "results" / d / e

    monkeypatch.setattr(io_mod, "resolve_features_path", lambda d, e, i=None: parquet)
    monkeypatch.setattr(io_mod, "load_features_df", lambda d, e="baseline", i=None: pd.read_parquet(parquet))
    monkeypatch.setattr(paths_mod, "results_dir", _results)
    monkeypatch.setattr(paths_mod, "figures_dir", lambda d, e: _results(d, e) / "figures")
    monkeypatch.setattr(paths_mod, "models_dir", lambda d, e: tmp_path / "models" / d / e)
    monkeypatch.setattr(bench_mod, "results_dir", _results)
    monkeypatch.setattr(bench_mod, "figures_dir", lambda d, e: _results(d, e) / "figures")
    monkeypatch.setattr(bench_mod, "models_dir", lambda d, e: tmp_path / "models" / d / e)

    result = run_benchmark(
        dataset="eyesclosed",
        experiment="baseline",
        models=["logistic_regression"],
        config={"seed": 42, "cv_folds": 3},
        cv_folds=3,
    )
    from pathlib import Path

    assert Path(result["benchmark_csv"]).exists()
    meta_path = tmp_path / "results" / "eyesclosed" / "baseline" / "benchmark_metadata.json"
    assert meta_path.exists()
