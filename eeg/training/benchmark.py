"""Central benchmark runner — single source of truth for model evaluation."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from eeg.config import load_experiment, resolve_dataset
from eeg.io import load_features_df, write_json
from eeg.paths import figures_dir, models_dir, results_dir
from eeg.repro import attach_repro_metadata, init_repro, snapshot_environment
from eeg.training.datasets import feature_columns, load_selected_features
from eeg.training.evaluation import (
    bootstrap_ci,
    compute_benchmark_metrics,
    plot_calibration,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr,
    plot_roc,
)

MODEL_REGISTRY: dict[str, Any] = {
    "logistic_regression": lambda seed: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            ),
        ]
    ),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1
    ),
    "xgboost": lambda seed: XGBClassifier(
        random_state=seed,
        eval_metric="mlogloss",
        objective="multi:softprob",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
    ),
    "mlp": lambda seed: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=500,
                    random_state=seed,
                ),
            ),
        ]
    ),
}


def _model_size_bytes(estimator) -> int:
    buf = io.BytesIO()
    joblib.dump(estimator, buf)
    return buf.tell()


def _get_feature_frame(
    dataset: str,
    experiment: str,
    dataset_id: int,
    feature_cols: list[str] | None,
):
    if feature_cols is not None:
        try:
            return load_selected_features(dataset, experiment), feature_cols
        except FileNotFoundError:
            pass
    df = load_features_df(dataset, experiment, dataset_id)
    cols = feature_cols or feature_columns()
    return df, cols


def run_benchmark(
    dataset: str,
    experiment: str,
    models: list[str] | None = None,
    feature_cols: list[str] | None = None,
    config: dict[str, Any] | None = None,
    cv_folds: int | None = None,
) -> dict[str, Any]:
    """Run GroupKFold benchmark for registered models and write artifacts."""
    cfg = config or load_experiment(experiment)
    seed = cfg.get("seed", cfg.get("training", {}).get("random_state", 42))
    init_repro(seed)

    spec = resolve_dataset(dataset)[0]
    df, cols = _get_feature_frame(dataset, experiment, spec.id, feature_cols)
    meta_cols = {"participant_id", "label", "dataset_id", "dataset_name"}
    feature_cols_resolved = [c for c in cols if c in df.columns and c not in meta_cols]
    if not feature_cols_resolved:
        feature_cols_resolved = [c for c in feature_columns() if c in df.columns]

    X = df[feature_cols_resolved].values
    groups = df["participant_id"].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    class_names = list(le.classes_)

    n_folds = cv_folds or cfg.get("cv_folds", cfg.get("training", {}).get("cv_folds", 5))
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    model_keys = models or list(MODEL_REGISTRY.keys())
    res_dir = results_dir(dataset, experiment)
    fig_dir = figures_dir(dataset, experiment)
    model_dir = models_dir(dataset, experiment)
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    detail: dict[str, Any] = {}
    predictions_rows: list[dict] = []

    for model_key in model_keys:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_key}. Known: {list(MODEL_REGISTRY)}")

        fold_metrics: list[dict] = []
        fold_preds: list[tuple] = []
        train_times: list[float] = []
        predict_times: list[float] = []
        final_estimator = None

        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
            estimator = MODEL_REGISTRY[model_key](seed)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            t0 = time.perf_counter()
            estimator.fit(X_train, y_train)
            train_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            y_pred = estimator.predict(X_test)
            y_proba = estimator.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None
            predict_times.append((time.perf_counter() - t1) / max(len(X_test), 1))

            m = compute_benchmark_metrics(y_test, y_pred, y_proba, labels=list(range(len(class_names))))
            m["fold"] = fold_idx
            fold_metrics.append(m)
            fold_preds.append((y_test, y_pred, y_proba))

            for i, idx in enumerate(test_idx):
                predictions_rows.append(
                    {
                        "model": model_key,
                        "fold": fold_idx,
                        "participant_id": df.iloc[idx]["participant_id"],
                        "y_true": int(y_test[i]) if i < len(y_test) else int(y[idx]),
                        "y_pred": int(y_pred[i]),
                    }
                )

            final_estimator = estimator

        # Aggregate holdout-style metrics from last fold for figures
        y_test_last, y_pred_last, y_proba_last = fold_preds[-1]
        agg = compute_benchmark_metrics(
            y_test_last, y_pred_last, y_proba_last, labels=list(range(len(class_names)))
        )

        def _metric_fn(yt, yp, ypr):
            return compute_benchmark_metrics(yt, yp, ypr, labels=list(range(len(class_names))))[
                "balanced_accuracy"
            ]

        ci_mean, ci_lo, ci_hi = bootstrap_ci(
            _metric_fn,
            y_test_last,
            y_pred_last,
            y_proba_last,
            seed=seed,
        )

        model_path = model_dir / f"{model_key}.joblib"
        joblib.dump({"pipeline": final_estimator, "label_encoder": le, "features": feature_cols_resolved}, model_path)
        model_bytes = _model_size_bytes(final_estimator)

        row = {
            "model": model_key,
            "dataset": dataset,
            "experiment": experiment,
            "accuracy": agg["accuracy"],
            "balanced_accuracy": agg["balanced_accuracy"],
            "macro_f1": agg["macro_f1"],
            "macro_roc_auc": agg.get("macro_roc_auc"),
            "macro_pr_auc": agg.get("macro_pr_auc"),
            "mcc": agg["mcc"],
            "cohen_kappa": agg["cohen_kappa"],
            "balanced_accuracy_ci_lo": ci_lo,
            "balanced_accuracy_ci_hi": ci_hi,
            "train_time_s": float(np.mean(train_times)),
            "predict_time_per_sample_s": float(np.mean(predict_times)),
            "model_size_bytes": model_bytes,
            "n_folds": n_folds,
        }
        all_rows.append(row)
        detail[model_key] = {"fold_metrics": fold_metrics, "aggregate": agg}

        if y_proba_last is not None:
            plot_roc(y_test_last, y_proba_last, class_names, fig_dir / f"roc_{model_key}.png")
            plot_pr(y_test_last, y_proba_last, class_names, fig_dir / f"pr_{model_key}.png")
            plot_calibration(y_test_last, y_proba_last, fig_dir / f"calibration_{model_key}.png")
        plot_confusion_matrix(
            y_test_last, y_pred_last, class_names, fig_dir / f"confusion_{model_key}.png"
        )

        if hasattr(final_estimator, "feature_importances_"):
            imp = dict(zip(feature_cols_resolved, final_estimator.feature_importances_))
            plot_feature_importance(imp, fig_dir / f"importance_{model_key}.png")
        elif isinstance(final_estimator, Pipeline) and hasattr(
            final_estimator.named_steps.get("clf", final_estimator.steps[-1][1]),
            "feature_importances_",
        ):
            clf = final_estimator.named_steps["clf"]
            imp = dict(zip(feature_cols_resolved, clf.feature_importances_))
            plot_feature_importance(imp, fig_dir / f"importance_{model_key}.png")

    benchmark_df = pd.DataFrame(all_rows)
    benchmark_path = res_dir / "benchmark.csv"
    benchmark_df.to_csv(benchmark_path, index=False)

    detail_path = res_dir / "benchmark_detail.json"
    write_json(detail_path, detail)

    metadata = attach_repro_metadata(
        {"dataset": dataset, "experiment": experiment, "models": model_keys},
        cfg if isinstance(cfg, dict) else {"experiment": experiment, "seed": seed},
    )
    write_json(res_dir / "benchmark_metadata.json", metadata)

    if predictions_rows:
        pd.DataFrame(predictions_rows).to_csv(res_dir / "predictions.csv", index=False)

    return {
        "benchmark_csv": str(benchmark_path),
        "benchmark_detail": detail,
        "rows": all_rows,
        "metadata": metadata,
    }
