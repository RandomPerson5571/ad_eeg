"""Tests for benchmark runner."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from eeg.training.benchmark import (
    MODEL_REGISTRY,
    _optimize_thresholds,
    _predict_with_threshold,
    _selection_settings,
    run_benchmark,
)
from eeg.training.evaluation import bootstrap_ci, compute_benchmark_metrics


def test_model_registry_keys():
    assert "logistic_regression" in MODEL_REGISTRY
    assert "random_forest" in MODEL_REGISTRY
    assert "xgboost" in MODEL_REGISTRY
    assert "mlp" in MODEL_REGISTRY


def test_selection_settings_accepts_notebook_experiment_name():
    settings = _selection_settings(
        {"experiment": "baseline", "feature_selection": {"top_k": 8}},
        seed=42,
    )

    assert settings["top_k"] == 8
    assert settings["random_state"] == 42


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
    expected_ap = average_precision_score(
        np.eye(3, dtype=int)[y_true], y_proba, average="macro"
    )
    assert m["macro_pr_auc"] == pytest.approx(expected_ap)


def test_bootstrap_ci():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])

    def fn(yt, yp, ypr):
        return float((yt == yp).mean())

    mean, lo, hi = bootstrap_ci(fn, y_true, y_pred, n=50, seed=0)
    assert lo <= mean <= hi


def test_threshold_helpers_support_binary_and_multiclass():
    binary = np.array([[0.8, 0.2], [0.4, 0.6]])
    assert _predict_with_threshold(binary, [0.7]).tolist() == [0, 0]
    assert _predict_with_threshold(binary, [0.5]).tolist() == [0, 1]

    multiclass = np.array(
        [[0.55, 0.40, 0.05], [0.20, 0.45, 0.35], [0.10, 0.20, 0.70]]
    )
    thresholds = _optimize_thresholds(np.array([0, 1, 2]), multiclass, 3)
    assert len(thresholds) == 3
    assert _predict_with_threshold(multiclass, thresholds).shape == (3,)


def test_run_benchmark_smoke(tmp_path, monkeypatch):
    """Minimal feature frame with enough subjects for nested 3x2 grouped CV."""
    rng = np.random.default_rng(0)
    n = 30
    rows = []
    for sid in range(12):
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
        config={
            "seed": 42,
            "cv_folds": 3,
            "inner_cv_folds": 2,
            "bootstrap_iterations": 30,
        },
        cv_folds=3,
    )
    from pathlib import Path

    assert Path(result["benchmark_csv"]).exists()
    assert Path(result["fold_metrics_csv"]).exists()
    meta_path = tmp_path / "results" / "eyesclosed" / "baseline" / "benchmark_metadata.json"
    assert meta_path.exists()
    predictions = pd.read_csv(
        tmp_path / "results" / "eyesclosed" / "baseline" / "predictions.csv"
    )
    epoch_predictions = pd.read_csv(
        tmp_path / "results" / "eyesclosed" / "baseline" / "epoch_predictions.csv"
    )
    assert len(predictions) == 12
    assert predictions["participant_id"].nunique() == 12
    assert len(epoch_predictions) == len(df)
    details = result["benchmark_detail"]["logistic_regression"]
    assert len(details["fold_details"]) == 3
    assert all(fold["inner_cv_folds"] == 2 for fold in details["fold_details"])
    assert all(fold["selected_features"] for fold in details["fold_details"])


def test_run_benchmark_threshold_calibration_and_lr_variants(tmp_path, monkeypatch):
    """Calibration, threshold tuning, L1 selection, and PCA remain grouped."""
    rng = np.random.default_rng(1)
    rows = []
    for sid in range(12):
        label = ["A", "F", "C"][sid % 3]
        for epoch in range(3):
            rows.append(
                {
                    "participant_id": f"sub-{sid:03d}",
                    "label": label,
                    "dataset_id": 2,
                    "dataset_name": "eyesclosed",
                    "epoch_id": epoch,
                    **{f"f{i}": rng.normal(loc=(sid % 3) * 0.2, scale=1.0) for i in range(6)},
                }
            )
    df = pd.DataFrame(rows)
    parquet = tmp_path / "subject_features.parquet"
    df.to_parquet(parquet, engine="pyarrow")

    import eeg.io as io_mod
    import eeg.paths as paths_mod
    import eeg.training.benchmark as bench_mod

    def _results(d, e):
        return tmp_path / "results" / d / e

    monkeypatch.setattr(io_mod, "load_features_df", lambda d, e="baseline", i=None: df.copy())
    monkeypatch.setattr(bench_mod, "load_features_df", lambda d, e="baseline", i=None: df.copy())
    monkeypatch.setattr(paths_mod, "results_dir", _results)
    monkeypatch.setattr(paths_mod, "figures_dir", lambda d, e: _results(d, e) / "figures")
    monkeypatch.setattr(paths_mod, "models_dir", lambda d, e: tmp_path / "models" / d / e)
    monkeypatch.setattr(bench_mod, "results_dir", _results)
    monkeypatch.setattr(bench_mod, "figures_dir", lambda d, e: _results(d, e) / "figures")
    monkeypatch.setattr(bench_mod, "models_dir", lambda d, e: tmp_path / "models" / d / e)

    base_config = {
        "seed": 42,
        "cv_folds": 3,
        "inner_cv_folds": 2,
        "bootstrap_iterations": 10,
        "features": {"feature_columns": [f"f{i}" for i in range(6)]},
        "feature_selection": {"correlation_threshold": 0.99},
    }
    result = run_benchmark(
        "eyesclosed",
        "baseline",
        models=["logistic_regression"],
        config=base_config,
        cv_folds=3,
        threshold_optimization=True,
        calibration_method="sigmoid",
        output_tag="threshold_test",
    )
    row = result["rows"][0]
    assert row["threshold_optimization"] is True
    assert row["calibration_method"] == "sigmoid"
    assert row["expected_calibration_error"] is not None
    assert all(
        fold["decision_thresholds"] and fold["calibration_method"] == "sigmoid"
        for fold in result["benchmark_detail"]["logistic_regression"]["fold_details"]
    )

    for model_key in ("logistic_regression_l1_select", "logistic_regression_pca"):
        variant = run_benchmark(
            "eyesclosed",
            "baseline",
            models=[model_key],
            config=base_config,
            cv_folds=3,
            output_tag=model_key,
        )
        assert variant["rows"][0]["model"] == model_key
        assert variant["benchmark_detail"][model_key]["final_model"]["selected_features"]
