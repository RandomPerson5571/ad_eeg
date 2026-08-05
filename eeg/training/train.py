"""Model training functions (XGBoost, MLP) — preserved from monolithic training.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from category_encoders import TargetEncoder
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from xgboost import XGBClassifier

from eeg.config import load_base_configs, load_experiment
from eeg.io import write_json
from eeg.paths import models_dir, results_dir
from eeg.training.datasets import feature_columns, prepare_data, subject_level_split, validate_feature_schema
from eeg.training.evaluation import plot_confusion_matrix, plot_roc


def classification_metrics(y_true, y_pred, label_encoder):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_model_artifact(pipeline, label_encoder, feature_names, split_ids, model_name, output_path, extra=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "feature_names": list(feature_names),
        "split_ids": split_ids,
        "model_name": model_name,
    }
    if extra:
        artifact.update(extra)
    joblib.dump(artifact, output_path)
    return artifact


def _plot_confusion_matrix_legacy(y_true, y_pred, label_encoder, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=label_encoder.classes_, ax=ax, cmap="Blues"
    )
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_roc_legacy(y_true, y_proba, label_encoder, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, name="model", ax=ax)
    ax.set_title("ROC curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_xgboost(dataset_name: str, experiment: str, dataset_id: int, config: dict | None = None):
    cfg = config or load_experiment(experiment)
    train_cfg = cfg["training"]
    random_state = train_cfg.get("random_state", 42)
    xgb_cfg = train_cfg.get("xgboost", {})

    x_train, x_test, y_train, y_test, label_encoder, split_ids = prepare_data(
        dataset_name, experiment, dataset_id, cfg
    )

    pipe = Pipeline(
        [
            ("encoder", TargetEncoder()),
            (
                "clf",
                XGBClassifier(
                    random_state=random_state,
                    eval_metric="mlogloss",
                    objective="multi:softprob",
                ),
            ),
        ]
    )
    search_space = {
        "clf__max_depth": Integer(2, 8),
        "clf__learning_rate": Real(0.001, 1.0, prior="log-uniform"),
        "clf__subsample": Real(0.5, 1.0),
        "clf__colsample_bytree": Real(0.5, 1.0),
        "clf__colsample_bylevel": Real(0.5, 1.0),
        "clf__colsample_bynode": Real(0.5, 1.0),
        "clf__reg_alpha": Real(0.0, 10.0),
        "clf__reg_lambda": Real(0.0, 10.0),
        "clf__gamma": Real(0.0, 10.0),
    }
    opt = BayesSearchCV(
        pipe,
        search_space,
        cv=xgb_cfg.get("cv", 3),
        n_iter=xgb_cfg.get("n_iter", 10),
        scoring="balanced_accuracy",
        random_state=random_state,
        n_jobs=1,
        verbose=0,
    )
    opt.fit(x_train, y_train)
    y_pred = opt.predict(x_test)
    metrics = classification_metrics(y_test, y_pred, label_encoder)
    metrics["cv_best_score"] = float(opt.best_score_)
    metrics["model"] = "xgboost"
    metrics["dataset_name"] = dataset_name
    metrics["dataset_id"] = dataset_id
    metrics["experiment"] = experiment

    model_dir = models_dir(dataset_name, experiment)
    res_dir = results_dir(dataset_name, experiment)
    model_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "xgboost.joblib"
    save_model_artifact(
        opt.best_estimator_, label_encoder, list(x_train.columns), split_ids, "xgboost", model_path
    )

    xgb_step = opt.best_estimator_.named_steps["clf"]
    importances = xgb_step.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = list(x_train.columns)
    metrics["feature_importances"] = {
        feature_names[i]: float(importances[i]) for i in indices[:10]
    }

    y_proba = opt.predict_proba(x_test)
    _plot_confusion_matrix_legacy(y_test, y_pred, label_encoder, res_dir / "confusion_matrix_xgboost.png")
    if y_proba.shape[1] == 2:
        _plot_roc_legacy(y_test, y_proba[:, 1], label_encoder, res_dir / "roc_xgboost.png")

    write_json(res_dir / "metrics_xgboost.json", metrics)
    write_json(res_dir / "subject_splits.json", split_ids)
    return opt, metrics


def train_mlp(dataset_name: str, experiment: str, dataset_id: int, config: dict | None = None):
    cfg = config or load_experiment(experiment)
    train_cfg = cfg["training"]
    random_state = train_cfg.get("random_state", 42)
    mlp_cfg = train_cfg.get("mlp", {})
    hidden = tuple(mlp_cfg.get("hidden_layers", [128, 64]))

    x_train, x_test, y_train, y_test, label_encoder, split_ids = prepare_data(
        dataset_name, experiment, dataset_id, cfg
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=hidden,
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=25,
                    tol=1e-4,
                    random_state=random_state,
                    verbose=False,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    metrics = classification_metrics(y_test, y_pred, label_encoder)
    metrics["model"] = "mlp"
    metrics["dataset_name"] = dataset_name
    metrics["dataset_id"] = dataset_id
    metrics["experiment"] = experiment

    model_dir = models_dir(dataset_name, experiment)
    res_dir = results_dir(dataset_name, experiment)
    model_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "mlp.joblib"
    save_model_artifact(pipeline, label_encoder, list(x_train.columns), split_ids, "mlp", model_path)

    _plot_confusion_matrix_legacy(y_test, y_pred, label_encoder, res_dir / "confusion_matrix_mlp.png")
    write_json(res_dir / "metrics_mlp.json", metrics)
    return pipeline, metrics


def train_models(
    dataset_name: str,
    experiment: str,
    dataset_id: int,
    models: list[str],
    config: dict | None = None,
) -> dict[str, Any]:
    results = {}
    for model in models:
        name = model.strip().lower()
        if name == "xgboost":
            _, metrics = train_xgboost(dataset_name, experiment, dataset_id, config)
            results["xgboost"] = metrics
        elif name == "mlp":
            _, metrics = train_mlp(dataset_name, experiment, dataset_id, config)
            results["mlp"] = metrics
        else:
            raise ValueError(f"Unknown model: {model}")

    combined_path = results_dir(dataset_name, experiment) / "metrics.json"
    write_json(combined_path, results)
    return results
