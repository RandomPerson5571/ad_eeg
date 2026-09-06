"""Metrics, bootstrap CIs, and figure writers for benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def compute_benchmark_metrics(
    y_true,
    y_pred,
    y_proba=None,
    labels: list | None = None,
) -> dict:
    """Compute classification metrics including per-class and macro averages."""
    labels = labels or sorted(set(y_true) | set(y_pred))
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    if y_proba is not None and len(labels) > 1:
        try:
            if len(labels) == 2:
                metrics["macro_roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                y_bin = label_binarize(y_true, classes=labels)
                metrics["macro_roc_auc"] = float(
                    roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
                )
            if len(labels) == 2:
                metrics["macro_pr_auc"] = float(
                    average_precision_score(
                        np.asarray(y_true) == labels[1], np.asarray(y_proba)[:, 1]
                    )
                )
            else:
                y_bin = label_binarize(y_true, classes=labels)
                metrics["macro_pr_auc"] = float(
                    average_precision_score(y_bin, y_proba, average="macro")
                )
            y_bin = label_binarize(y_true, classes=labels)
            if len(labels) == 2:
                y_bin = np.column_stack([1 - y_bin, y_bin])
            probabilities = np.asarray(y_proba, dtype=float)
            metrics["brier_score"] = float(
                np.mean(np.sum((y_bin - probabilities) ** 2, axis=1))
            )
            metrics["log_loss"] = float(log_loss(y_true, probabilities, labels=labels))
            confidence = probabilities.max(axis=1)
            correctness = (np.asarray(y_pred) == np.asarray(y_true)).astype(float)
            bin_edges = np.linspace(0.0, 1.0, 11)
            ece = 0.0
            for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
                in_bin = (confidence >= lower) & (
                    (confidence < upper) if upper < 1.0 else (confidence <= upper)
                )
                if in_bin.any():
                    ece += float(in_bin.mean()) * abs(
                        float(correctness[in_bin].mean()) - float(confidence[in_bin].mean())
                    )
            metrics["expected_calibration_error"] = float(ece)
        except (ValueError, IndexError):
            metrics["macro_roc_auc"] = None
            metrics["macro_pr_auc"] = None

    # Per-class sensitivity / specificity from confusion matrix
    per_class = {}
    for i, lbl in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        per_class[str(lbl)] = {
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        }
    metrics["per_class"] = per_class
    return metrics


def bootstrap_ci(
    metric_fn: Callable,
    y_true,
    y_pred,
    y_proba=None,
    n: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    strata=None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI, optionally resampling within each stratum."""
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    strata_array = np.asarray(strata) if strata is not None else None
    scores = []
    for _ in range(n):
        if strata_array is None:
            idx = rng.integers(0, n_samples, n_samples)
        else:
            idx = np.concatenate(
                [
                    rng.choice(class_idx, size=len(class_idx), replace=True)
                    for value in np.unique(strata_array)
                    if len(class_idx := np.flatnonzero(strata_array == value))
                ]
            )
        yt = np.asarray(y_true)[idx]
        yp = np.asarray(y_pred)[idx]
        ypr = y_proba[idx] if y_proba is not None else None
        try:
            scores.append(float(metric_fn(yt, yp, ypr)))
        except Exception:
            continue
    if not scores:
        return (float("nan"), float("nan"), float("nan"))
    lo = float(np.percentile(scores, 100 * alpha / 2))
    hi = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return float(np.mean(scores)), lo, hi


def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, ax=ax, cmap="Blues"
    )
    ax.set_title("Confusion matrix")
    _save_fig(fig, out_path)


def plot_roc(y_true, y_proba, class_names, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if y_proba.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        ax.plot(fpr, tpr, label=class_names[1] if len(class_names) > 1 else "positive")
    else:
        for i, name in enumerate(class_names):
            y_bin = (np.asarray(y_true) == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
            ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curve")
    ax.legend()
    _save_fig(fig, out_path)


def plot_pr(y_true, y_proba, class_names, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if y_proba.shape[1] == 2:
        prec, rec, _ = precision_recall_curve(y_true, y_proba[:, 1])
        ax.plot(rec, prec)
    else:
        for i, name in enumerate(class_names):
            y_bin = (np.asarray(y_true) == i).astype(int)
            prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, i])
            ax.plot(rec, prec, label=name)
        ax.legend()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    _save_fig(fig, out_path)


def plot_calibration(y_true, y_proba, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if y_proba.shape[1] == 2:
        CalibrationDisplay.from_predictions(y_true, y_proba[:, 1], n_bins=10, ax=ax)
    else:
        for class_idx in range(y_proba.shape[1]):
            class_true = (np.asarray(y_true) == class_idx).astype(int)
            CalibrationDisplay.from_predictions(
                class_true,
                y_proba[:, class_idx],
                n_bins=8,
                ax=ax,
                name=f"class {class_idx}",
            )
    ax.set_title("Calibration curve")
    _save_fig(fig, out_path)


def plot_feature_importance(importances: dict[str, float], out_path: Path, top_n: int = 15) -> None:
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, vals = zip(*items) if items else ([], [])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(list(reversed(names)), list(reversed(vals)))
    ax.set_title("Feature importance")
    _save_fig(fig, out_path)


def plot_learning_curve(train_sizes, train_scores, test_scores, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(train_sizes, train_scores, "o-", label="train")
    ax.plot(train_sizes, test_scores, "o-", label="validation")
    ax.set_xlabel("Training size")
    ax.set_ylabel("Score")
    ax.set_title("Learning curve")
    ax.legend()
    _save_fig(fig, out_path)


def plot_runtime_bar(runtimes: dict[str, float], out_path: Path, title: str = "Runtime") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(runtimes.keys()), list(runtimes.values()))
    ax.set_ylabel("seconds")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    _save_fig(fig, out_path)
