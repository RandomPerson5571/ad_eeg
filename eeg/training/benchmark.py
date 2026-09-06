"""Leakage-safe nested grouped benchmark with subject-level OOF scoring."""

from __future__ import annotations

import io
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eeg.config import load_experiment, resolve_dataset
from eeg.feature_selection import FoldFeatureSelector
from eeg.io import load_features_df, write_json
from eeg.paths import figures_dir, models_dir, results_dir
from eeg.repro import attach_repro_metadata, init_repro
from eeg.training.datasets import feature_columns
from eeg.training.evaluation import (
    bootstrap_ci,
    compute_benchmark_metrics,
    plot_calibration,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr,
    plot_roc,
)

def _xgboost_classifier(seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        random_state=seed,
        eval_metric="mlogloss",
        objective="multi:softprob",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
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
    "logistic_regression_l1": lambda seed: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    penalty="l1",
                    solver="saga",
                    random_state=seed,
                ),
            ),
        ]
    ),
    "logistic_regression_elasticnet": lambda seed: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    random_state=seed,
                ),
            ),
        ]
    ),
    "logistic_regression_l1_select": lambda seed: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "l1_select",
                SelectFromModel(
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        penalty="l1",
                        solver="saga",
                        random_state=seed,
                    ),
                    threshold="median",
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            ),
        ]
    ),
    "logistic_regression_pca": lambda seed: Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, svd_solver="full", random_state=seed)),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            ),
        ]
    ),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1
    ),
    "xgboost": _xgboost_classifier,
    "mlp": lambda seed: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed
                ),
            ),
        ]
    ),
}

# Small explicit grids make the inner loop real without making the default
# benchmark intractable. They can be overridden under training.parameter_grids.
PARAMETER_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {"model__clf__C": [0.1, 1.0, 10.0]},
    "logistic_regression_l1": {"model__clf__C": [0.01, 0.1, 1.0, 10.0]},
    "logistic_regression_elasticnet": {
        "model__clf__C": [0.01, 0.1, 1.0, 10.0],
        "model__clf__l1_ratio": [0.2, 0.5, 0.8],
    },
    "logistic_regression_l1_select": {
        "model__l1_select__estimator__C": [0.01, 0.1, 1.0],
        "model__clf__C": [0.1, 1.0, 10.0],
    },
    "logistic_regression_pca": {
        "model__pca__n_components": [0.8, 0.95, None],
        "model__clf__C": [0.1, 1.0, 10.0],
    },
    "random_forest": {
        "model__max_depth": [None, 8],
        "model__min_samples_leaf": [1, 3],
    },
    "xgboost": {
        "model__max_depth": [3, 6],
        "model__learning_rate": [0.05, 0.1],
    },
    "mlp": {"model__clf__alpha": [1e-4, 1e-3]},
}

DEFAULT_MODEL_KEYS = ["logistic_regression", "random_forest", "xgboost", "mlp"]


def _model_size_bytes(estimator) -> int:
    buf = io.BytesIO()
    joblib.dump(estimator, buf)
    return buf.tell()


def _candidate_features(df: pd.DataFrame, config: dict, requested: list[str] | None) -> list[str]:
    meta_cols = {"participant_id", "label", "dataset_id", "dataset_name", "epoch_id"}
    configured = config.get("feature_columns") if isinstance(config, dict) else None
    configured_columns = (
        feature_columns(config)
        if isinstance(config, dict) and ("features" in config or "feature_columns" in config)
        else feature_columns()
    )
    candidates = requested or configured or configured_columns
    resolved = [c for c in candidates if c in df.columns and c not in meta_cols]
    if not resolved:
        resolved = [c for c in configured_columns if c in df.columns]
    if not resolved:
        raise ValueError("No configured feature columns exist in the feature parquet.")
    return resolved


def _selection_settings(config: dict, seed: int) -> dict[str, Any]:
    training = config.get("training", {})
    # Notebook configs identify the experiment by name (for example,
    # ``{"experiment": "baseline"}``), while some callers may provide a
    # nested experiment config. Only the latter can contain selection options.
    experiment = config.get("experiment", {})
    experiment_settings = experiment if isinstance(experiment, dict) else {}
    selection = (
        config.get("feature_selection")
        or training.get("feature_selection")
        or experiment_settings.get("feature_selection")
        or {}
    )
    return {
        "variance_threshold": selection.get("variance_threshold", 0.0),
        "correlation_threshold": selection.get("correlation_threshold", 0.95),
        "top_k": selection.get("top_k"),
        "random_state": seed,
    }


def _effective_grouped_splits(
    y: np.ndarray,
    groups: np.ndarray,
    requested: int,
    *,
    allow_no_inner: bool,
) -> int:
    subject_targets = pd.DataFrame({"group": groups, "target": y}).drop_duplicates()
    per_group = subject_targets.groupby("group")["target"].nunique()
    if (per_group > 1).any():
        raise ValueError("Each participant must have exactly one label.")
    class_subjects = subject_targets.groupby("target")["group"].nunique()
    possible = int(min(subject_targets["group"].nunique(), class_subjects.min()))
    if possible < 2:
        if allow_no_inner:
            return 0
        raise ValueError("Grouped CV requires at least two subjects in every class.")
    return min(int(requested), possible)


def _grouped_stratified_splits(
    y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
):
    """Stratify unique subjects, then expand each split to all of their epochs.

    Doing the stratification at subject granularity guarantees class-balanced
    folds when groups are pure-label, which ``StratifiedGroupKFold`` does not
    always achieve when every subject contributes many identical-label rows.
    """
    subjects = pd.DataFrame({"group": groups, "target": y}).drop_duplicates("group")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for subject_train, subject_test in splitter.split(subjects[["group"]], subjects["target"]):
        train_groups = set(subjects.iloc[subject_train]["group"])
        test_groups = set(subjects.iloc[subject_test]["group"])
        yield (
            np.flatnonzero(np.isin(groups, list(train_groups))),
            np.flatnonzero(np.isin(groups, list(test_groups))),
        )


def _build_pipeline(model_key: str, seed: int, selection: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("feature_selection", FoldFeatureSelector(**selection)),
            ("model", MODEL_REGISTRY[model_key](seed)),
        ]
    )


def _fit_with_inner_cv(
    model_key: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    requested_inner_folds: int,
    seed: int,
    selection: dict[str, Any],
    parameter_grid: dict[str, list[Any]],
    calibration_method: str | None = None,
):
    inner_folds = _effective_grouped_splits(
        y, groups, requested_inner_folds, allow_no_inner=True
    )
    candidate = _build_pipeline(model_key, seed, selection)
    if inner_folds == 0:
        candidate.fit(X, y)
        return candidate, {}, None, 0

    inner_cv = list(_grouped_stratified_splits(y, groups, inner_folds, seed))
    search = GridSearchCV(
        candidate,
        parameter_grid,
        scoring="balanced_accuracy",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y, groups=groups)
    estimator = search.best_estimator_
    if calibration_method:
        # Calibration folds are the same subject-grouped inner folds used to
        # select hyperparameters. No outer-fold observation enters this fit.
        calibrated = CalibratedClassifierCV(
            estimator=clone(candidate).set_params(**search.best_params_),
            method=calibration_method,
            cv=inner_cv,
            n_jobs=1,
        )
        calibrated.fit(X, y)
        estimator = calibrated
    return estimator, search.best_params_, float(search.best_score_), inner_folds


def _aligned_probabilities(estimator, X: pd.DataFrame, n_classes: int) -> np.ndarray | None:
    if not hasattr(estimator, "predict_proba"):
        return None
    raw = np.asarray(estimator.predict_proba(X))
    classes = np.asarray(estimator.classes_, dtype=int)
    aligned = np.zeros((len(X), n_classes), dtype=float)
    aligned[:, classes] = raw
    return aligned


def _aggregate_subject_predictions(
    participant_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    n_classes: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"participant_id": participant_ids, "y_true": y_true, "epoch_y_pred": y_pred}
    )
    if y_proba is not None:
        for class_idx in range(n_classes):
            frame[f"proba_{class_idx}"] = y_proba[:, class_idx]

    rows: list[dict[str, Any]] = []
    for participant_id, subject in frame.groupby("participant_id", sort=False):
        labels = subject["y_true"].unique()
        if len(labels) != 1:
            raise ValueError(f"Participant {participant_id} has multiple labels.")
        row: dict[str, Any] = {
            "participant_id": participant_id,
            "y_true": int(labels[0]),
            "n_epochs": int(len(subject)),
        }
        if y_proba is not None:
            mean_proba = np.asarray([subject[f"proba_{i}"].mean() for i in range(n_classes)])
            row["y_pred"] = int(mean_proba.argmax())
            row.update({f"proba_{i}": float(mean_proba[i]) for i in range(n_classes)})
        else:
            counts = np.bincount(subject["epoch_y_pred"].astype(int), minlength=n_classes)
            row["y_pred"] = int(counts.argmax())
        rows.append(row)
    return pd.DataFrame(rows)


def _base_pipeline(estimator):
    """Unwrap calibration while keeping feature-selection introspection generic."""
    if isinstance(estimator, CalibratedClassifierCV):
        calibrated = getattr(estimator, "calibrated_classifiers_", [])
        if calibrated:
            return getattr(calibrated[0], "estimator", calibrated[0])
        if hasattr(estimator, "estimator"):
            return estimator.estimator
    return estimator


def _selected_features(estimator) -> list[str]:
    pipeline = _base_pipeline(estimator)
    if not hasattr(pipeline, "named_steps"):
        return []
    selected = list(pipeline.named_steps["feature_selection"].selected_columns_)
    model = pipeline.named_steps.get("model")
    if not isinstance(model, Pipeline):
        return selected
    for _, transformer in model.steps:
        if isinstance(transformer, SelectFromModel):
            support = transformer.get_support()
            selected = list(np.asarray(selected, dtype=object)[support])
    return selected


def _feature_importance(estimator) -> dict[str, float] | None:
    pipeline = _base_pipeline(estimator)
    if not hasattr(pipeline, "named_steps"):
        return None
    model = pipeline.named_steps["model"]
    if isinstance(model, Pipeline):
        final_model = model.steps[-1][1]
        if hasattr(model.named_steps.get("pca"), "components_") and hasattr(final_model, "coef_"):
            coefficients = np.mean(np.abs(np.asarray(final_model.coef_, dtype=float)), axis=0)
            values = coefficients @ np.abs(model.named_steps["pca"].components_)
            return dict(zip(_selected_features(estimator), values))
        model = final_model
    if hasattr(model, "feature_importances_"):
        return dict(zip(_selected_features(estimator), model.feature_importances_))
    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_, dtype=float)
        values = np.mean(np.abs(values), axis=0)
        return dict(zip(_selected_features(estimator), values))
    if hasattr(pipeline.named_steps.get("l1_select"), "estimator_"):
        selector = pipeline.named_steps["l1_select"]
        values = np.mean(np.abs(np.asarray(selector.estimator_.coef_)), axis=0)
        return dict(zip(_selected_features(pipeline), values))
    if not _selected_features(estimator):
        return None
    return None


def _predict_with_threshold(y_proba: np.ndarray, thresholds: list[float] | np.ndarray) -> np.ndarray:
    """Convert probabilities to predictions using a fold-local threshold policy."""
    probabilities = np.asarray(y_proba, dtype=float)
    values = np.asarray(thresholds, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Thresholded prediction requires probabilities for at least two classes.")
    if probabilities.shape[1] == 2 and values.size == 1:
        return (probabilities[:, 1] >= values[0]).astype(int)
    if values.size != probabilities.shape[1]:
        raise ValueError("One threshold is required for each class in multiclass prediction.")
    passes = probabilities >= values[None, :]
    predictions = probabilities.argmax(axis=1).astype(int)
    for row_idx in np.flatnonzero(passes.any(axis=1)):
        eligible = np.flatnonzero(passes[row_idx])
        predictions[row_idx] = int(
            eligible[np.argmax(probabilities[row_idx, eligible] - values[eligible])]
        )
    return predictions


def _fixed_label_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    cm = np.asarray(
        pd.crosstab(
            pd.Series(y_true, name="true"),
            pd.Series(y_pred, name="pred"),
            dropna=False,
        ).reindex(index=range(n_classes), columns=range(n_classes), fill_value=0)
    )
    denominators = cm.sum(axis=1)
    recalls = np.divide(
        np.diag(cm),
        denominators,
        out=np.zeros(n_classes, dtype=float),
        where=denominators != 0,
    )
    return float(recalls.mean())


def _inner_oof_subject_probabilities(
    model_key: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    inner_cv,
    *,
    best_params: dict[str, Any],
    selection: dict[str, Any],
    seed: int,
    n_classes: int,
    calibration_method: str | None = None,
) -> pd.DataFrame:
    """Generate subject-level inner OOF probabilities for threshold selection."""
    rows: list[pd.DataFrame] = []
    for split_idx, (fit_idx, valid_idx) in enumerate(inner_cv):
        estimator = _build_pipeline(
            model_key,
            seed + split_idx,
            {**selection, "random_state": seed + split_idx},
        )
        estimator.set_params(**best_params)
        if calibration_method:
            calibration_folds = _effective_grouped_splits(
                y[fit_idx], groups[fit_idx], min(3, len(set(groups[fit_idx]))), allow_no_inner=True
            )
            if calibration_folds > 1:
                calibration_cv = list(
                    _grouped_stratified_splits(
                        y[fit_idx], groups[fit_idx], calibration_folds, seed + split_idx
                    )
                )
                estimator = CalibratedClassifierCV(
                    estimator=estimator,
                    method=calibration_method,
                    cv=calibration_cv,
                    n_jobs=1,
                )
        estimator.fit(X.iloc[fit_idx], y[fit_idx])
        probabilities = _aligned_probabilities(estimator, X.iloc[valid_idx], n_classes)
        if probabilities is None:
            raise ValueError(f"Model '{model_key}' does not expose predict_proba().")
        predictions = probabilities.argmax(axis=1)
        subject = _aggregate_subject_predictions(
            groups[valid_idx],
            y[valid_idx],
            predictions,
            probabilities,
            n_classes,
        )
        rows.append(subject)
    result = pd.concat(rows, ignore_index=True)
    if result["participant_id"].duplicated().any():
        raise RuntimeError("Inner CV did not produce exactly one threshold-tuning prediction per subject.")
    return result


def _optimize_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
) -> list[float]:
    """Optimize binary or class-specific multiclass thresholds on inner OOF data."""
    candidates = np.linspace(0.1, 0.9, 9)
    if n_classes == 2:
        scores = [
            _fixed_label_balanced_accuracy(
                y_true, _predict_with_threshold(y_proba, [float(value)]), n_classes
            )
            for value in candidates
        ]
        best = max(range(len(scores)), key=lambda idx: (scores[idx], -abs(candidates[idx] - 0.5)))
        return [float(candidates[best])]

    thresholds = np.full(n_classes, 1.0 / n_classes, dtype=float)
    # Coordinate ascent keeps the multiclass search small while allowing each
    # class to have its own operating point. All tuning observations are inner OOF.
    for _ in range(2):
        for class_idx in range(n_classes):
            scores = []
            for value in candidates:
                trial = thresholds.copy()
                trial[class_idx] = value
                predictions = _predict_with_threshold(y_proba, trial)
                scores.append(_fixed_label_balanced_accuracy(y_true, predictions, n_classes))
            best = max(
                range(len(scores)),
                key=lambda idx: (scores[idx], -abs(candidates[idx] - 1.0 / n_classes)),
            )
            thresholds[class_idx] = candidates[best]
    return [float(value) for value in thresholds]


def run_benchmark(
    dataset: str,
    experiment: str,
    models: list[str] | None = None,
    feature_cols: list[str] | None = None,
    config: dict[str, Any] | None = None,
    cv_folds: int | None = None,
    output_tag: str | None = None,
    threshold_optimization: bool | None = None,
    calibration_method: str | None = None,
    label_permutation_seed: int | None = None,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Run nested grouped CV and score all outer-fold OOF predictions by subject."""
    cfg = config or load_experiment(experiment)
    training_cfg = cfg.get("training", {})
    seed = cfg.get("seed", training_cfg.get("random_state", 42))
    init_repro(seed)

    spec = resolve_dataset(dataset)[0]
    # Always use the unselected epoch table. feature_cols only limits candidates.
    df = load_features_df(dataset, experiment, spec.id).reset_index(drop=True)
    candidates = _candidate_features(df, cfg, feature_cols)
    X = df[candidates].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    groups = df["participant_id"].to_numpy()
    le = LabelEncoder()
    labels_for_encoding = df["label"].to_numpy()
    if label_permutation_seed is not None:
        subject_frame = df[["participant_id", "label"]].drop_duplicates("participant_id")
        rng = np.random.default_rng(label_permutation_seed)
        shuffled = subject_frame["label"].to_numpy().copy()
        rng.shuffle(shuffled)
        permuted = dict(zip(subject_frame["participant_id"], shuffled))
        labels_for_encoding = np.asarray([permuted[participant] for participant in groups])
    y = le.fit_transform(labels_for_encoding)
    class_names = list(le.classes_)
    labels = list(range(len(class_names)))

    requested_outer = cv_folds or cfg.get("cv_folds", training_cfg.get("cv_folds", 5))
    requested_inner = cfg.get("inner_cv_folds", training_cfg.get("inner_cv_folds", 3))
    outer_folds = _effective_grouped_splits(y, groups, requested_outer, allow_no_inner=False)
    outer_cv = list(_grouped_stratified_splits(y, groups, outer_folds, seed))
    selection = _selection_settings(cfg, seed)
    configured_grids = training_cfg.get("parameter_grids", {})
    threshold_optimization = (
        training_cfg.get("threshold_optimization", False)
        if threshold_optimization is None
        else threshold_optimization
    )
    calibration_method = (
        training_cfg.get("calibration_method")
        if calibration_method is None
        else calibration_method
    )
    if calibration_method not in {None, "sigmoid", "isotonic"}:
        raise ValueError("calibration_method must be one of: None, 'sigmoid', or 'isotonic'.")

    model_keys = models or DEFAULT_MODEL_KEYS
    artifact_experiment = (
        f"{experiment}/analysis/{output_tag}" if output_tag else experiment
    )
    res_dir = results_dir(dataset, artifact_experiment)
    fig_dir = figures_dir(dataset, artifact_experiment)
    model_dir = models_dir(dataset, artifact_experiment)
    if write_artifacts:
        res_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    subject_prediction_rows: list[dict[str, Any]] = []
    epoch_prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_key}. Known: {list(MODEL_REGISTRY)}")
        parameter_grid = configured_grids.get(model_key, PARAMETER_GRIDS[model_key])
        fold_metrics: list[dict[str, Any]] = []
        fold_subjects: list[pd.DataFrame] = []
        fold_details: list[dict[str, Any]] = []
        train_times: list[float] = []
        predict_times: list[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            train_groups = groups[train_idx]

            t0 = time.perf_counter()
            estimator, best_params, inner_score, inner_folds = _fit_with_inner_cv(
                model_key,
                X_train,
                y_train,
                train_groups,
                requested_inner_folds=requested_inner,
                seed=seed + fold_idx,
                selection={**selection, "random_state": seed + fold_idx},
                parameter_grid=parameter_grid,
                calibration_method=calibration_method,
            )
            train_times.append(time.perf_counter() - t0)

            thresholds: list[float] | None = None
            if threshold_optimization and inner_folds > 1:
                threshold_cv = list(
                    _grouped_stratified_splits(
                        y_train, train_groups, inner_folds, seed + fold_idx
                    )
                )
                inner_oof = _inner_oof_subject_probabilities(
                    model_key,
                    X_train,
                    y_train,
                    train_groups,
                    threshold_cv,
                    best_params=best_params,
                    selection={**selection, "random_state": seed + fold_idx},
                    seed=seed + fold_idx,
                    n_classes=len(class_names),
                    calibration_method=calibration_method,
                )
                inner_proba = inner_oof[[f"proba_{i}" for i in labels]].to_numpy()
                thresholds = _optimize_thresholds(
                    inner_oof["y_true"].to_numpy(), inner_proba, len(class_names)
                )

            t1 = time.perf_counter()
            epoch_pred = np.asarray(estimator.predict(X_test), dtype=int)
            epoch_proba = _aligned_probabilities(estimator, X_test, len(class_names))
            predict_times.append((time.perf_counter() - t1) / max(len(X_test), 1))

            subject_pred = _aggregate_subject_predictions(
                groups[test_idx], y_test, epoch_pred, epoch_proba, len(class_names)
            )
            if thresholds is not None:
                subject_pred["y_pred"] = _predict_with_threshold(
                    subject_pred[[f"proba_{i}" for i in labels]].to_numpy(), thresholds
                )
            subject_pred.insert(0, "fold", fold_idx)
            subject_pred.insert(0, "model", model_key)
            fold_subjects.append(subject_pred)

            proba_cols = [f"proba_{i}" for i in labels]
            subject_proba = subject_pred[proba_cols].to_numpy() if epoch_proba is not None else None
            metrics = compute_benchmark_metrics(
                subject_pred["y_true"].to_numpy(),
                subject_pred["y_pred"].to_numpy(),
                subject_proba,
                labels=labels,
            )
            metrics["fold"] = fold_idx
            metrics["n_subjects"] = int(len(subject_pred))
            fold_metrics.append(metrics)
            fold_details.append(
                {
                    "fold": fold_idx,
                    "inner_cv_folds": inner_folds,
                    "inner_best_score": inner_score,
                    "best_params": best_params,
                    "selected_features": _selected_features(estimator),
                    "threshold_optimization": bool(threshold_optimization),
                    "decision_thresholds": thresholds,
                    "calibration_method": calibration_method,
                    "train_subjects": sorted(set(train_groups)),
                    "test_subjects": sorted(set(groups[test_idx])),
                }
            )

            for local_idx, source_idx in enumerate(test_idx):
                epoch_row: dict[str, Any] = {
                    "model": model_key,
                    "fold": fold_idx,
                    "participant_id": groups[source_idx],
                    "epoch_id": df.iloc[source_idx].get("epoch_id", local_idx),
                    "y_true": int(y[source_idx]),
                    "y_pred": int(epoch_pred[local_idx]),
                }
                if epoch_proba is not None:
                    epoch_row.update(
                        {f"proba_{i}": float(epoch_proba[local_idx, i]) for i in labels}
                    )
                epoch_prediction_rows.append(epoch_row)

        oof = pd.concat(fold_subjects, ignore_index=True)
        expected_subjects = set(groups)
        if set(oof["participant_id"]) != expected_subjects or oof["participant_id"].duplicated().any():
            raise RuntimeError("Outer CV did not produce exactly one OOF prediction per subject.")
        oof = oof.sort_values("participant_id").reset_index(drop=True)
        oof_proba = oof[[f"proba_{i}" for i in labels]].to_numpy() if "proba_0" in oof else None
        oof_true = oof["y_true"].to_numpy()
        oof_pred = oof["y_pred"].to_numpy()
        aggregate = compute_benchmark_metrics(oof_true, oof_pred, oof_proba, labels=labels)

        def _metric_fn(yt, yp, ypr):
            return compute_benchmark_metrics(yt, yp, ypr, labels=labels)["balanced_accuracy"]

        _, ci_lo, ci_hi = bootstrap_ci(
            _metric_fn,
            oof_true,
            oof_pred,
            oof_proba,
            n=cfg.get("bootstrap_iterations", training_cfg.get("bootstrap_iterations", 1000)),
            seed=seed,
            strata=oof_true,
        )

        if write_artifacts:
            final_estimator, final_params, final_inner_score, final_inner_folds = _fit_with_inner_cv(
                model_key,
                X,
                y,
                groups,
                requested_inner_folds=requested_inner,
                seed=seed,
                selection=selection,
                parameter_grid=parameter_grid,
                calibration_method=calibration_method,
            )
        else:
            # The final all-subject fit is a deployment artifact, not part of
            # the unbiased OOF estimate. Avoid refitting it for permutations.
            final_estimator = estimator
            final_params = best_params
            final_inner_score = inner_score
            final_inner_folds = inner_folds
        if write_artifacts:
            model_path = model_dir / f"{model_key}.joblib"
            joblib.dump(
                {
                    "pipeline": final_estimator,
                    "label_encoder": le,
                    "candidate_features": candidates,
                    "selected_features": _selected_features(final_estimator),
                    "evaluation": "nested_stratified_group_cv_subject_oof",
                    "threshold_optimization": bool(threshold_optimization),
                    "calibration_method": calibration_method,
                },
                model_path,
            )

        row = {
            "model": model_key,
            "dataset": dataset,
            "experiment": experiment,
            "accuracy": aggregate["accuracy"],
            "balanced_accuracy": aggregate["balanced_accuracy"],
            "macro_f1": aggregate["macro_f1"],
            "macro_roc_auc": aggregate.get("macro_roc_auc"),
            "macro_pr_auc": aggregate.get("macro_pr_auc"),
            "brier_score": aggregate.get("brier_score"),
            "log_loss": aggregate.get("log_loss"),
            "expected_calibration_error": aggregate.get("expected_calibration_error"),
            "mcc": aggregate["mcc"],
            "cohen_kappa": aggregate["cohen_kappa"],
            "balanced_accuracy_ci_lo": ci_lo,
            "balanced_accuracy_ci_hi": ci_hi,
            "train_time_s": float(np.mean(train_times)),
            "predict_time_per_epoch_s": float(np.mean(predict_times)),
            "model_size_bytes": _model_size_bytes(final_estimator),
            "n_folds": outer_folds,
            "inner_cv_folds": final_inner_folds,
            "n_oof_subjects": int(len(oof)),
            "threshold_optimization": bool(threshold_optimization),
            "calibration_method": calibration_method,
            "label_permutation_seed": label_permutation_seed,
        }
        all_rows.append(row)
        detail[model_key] = {
            "fold_metrics": fold_metrics,
            "fold_details": fold_details,
            "aggregate_subject_oof": aggregate,
            "final_model": {
                "inner_cv_folds": final_inner_folds,
                "inner_best_score": final_inner_score,
                "best_params": final_params,
                "selected_features": _selected_features(final_estimator),
                "calibration_method": calibration_method,
            },
        }
        fold_metric_rows.extend({"model": model_key, **metrics} for metrics in fold_metrics)

        for record in oof.to_dict(orient="records"):
            record["y_true_label"] = class_names[int(record["y_true"])]
            record["y_pred_label"] = class_names[int(record["y_pred"])]
            subject_prediction_rows.append(record)

        if write_artifacts and oof_proba is not None:
            plot_roc(oof_true, oof_proba, class_names, fig_dir / f"roc_{model_key}.png")
            plot_pr(oof_true, oof_proba, class_names, fig_dir / f"pr_{model_key}.png")
            plot_calibration(oof_true, oof_proba, fig_dir / f"calibration_{model_key}.png")
        if write_artifacts:
            plot_confusion_matrix(oof_true, oof_pred, class_names, fig_dir / f"confusion_{model_key}.png")
        importance = _feature_importance(final_estimator)
        if write_artifacts and importance:
            plot_feature_importance(importance, fig_dir / f"importance_{model_key}.png")

    benchmark_path = res_dir / "benchmark.csv"
    fold_metrics_path = res_dir / "fold_metrics.csv"
    metadata = attach_repro_metadata(
        {
            "dataset": dataset,
            "experiment": experiment,
            "artifact_experiment": artifact_experiment,
            "models": model_keys,
            "evaluation": "nested_stratified_group_cv_subject_oof",
            "outer_cv_folds": outer_folds,
            "requested_inner_cv_folds": requested_inner,
            "threshold_optimization": bool(threshold_optimization),
            "calibration_method": calibration_method,
            "label_permutation_seed": label_permutation_seed,
        },
        cfg if isinstance(cfg, dict) else {"experiment": experiment, "seed": seed},
    )
    if write_artifacts:
        pd.DataFrame(all_rows).to_csv(benchmark_path, index=False)
        pd.DataFrame(fold_metric_rows).to_csv(fold_metrics_path, index=False)
        write_json(res_dir / "benchmark_detail.json", detail)
        write_json(res_dir / "benchmark_metadata.json", metadata)
        pd.DataFrame(subject_prediction_rows).to_csv(res_dir / "predictions.csv", index=False)
        pd.DataFrame(epoch_prediction_rows).to_csv(res_dir / "epoch_predictions.csv", index=False)

    return {
        "benchmark_csv": str(benchmark_path),
        "fold_metrics_csv": str(fold_metrics_path),
        "benchmark_detail": detail,
        "rows": all_rows,
        "metadata": metadata,
    }
