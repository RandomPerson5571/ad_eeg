"""Baseline feature selection: variance, correlation, mutual information."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

from eeg.paths import feature_importance_path, selected_features_path


@dataclass
class SelectionResult:
    selected_columns: list[str]
    importance: pd.DataFrame
    n_input: int
    n_after_variance: int
    n_after_correlation: int
    n_selected: int


def _correlation_filter(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop: set[str] = set()
    variances = df.var()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > threshold].tolist()
        for other in high_corr:
            if variances.get(col, 0) >= variances.get(other, 0):
                to_drop.add(other)
            else:
                to_drop.add(col)
    return [c for c in df.columns if c not in to_drop]


def select_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: dict | None = None,
    label_col: str = "label",
) -> SelectionResult:
    """VarianceThreshold → correlation filter → mutual information."""
    cfg = config or {}
    sel_cfg = cfg.get("feature_selection", {})
    var_threshold = sel_cfg.get("variance_threshold", 0.0)
    corr_threshold = sel_cfg.get("correlation_threshold", 0.95)
    top_k = sel_cfg.get("top_k", None)

    X = df[feature_cols].copy()
    n_input = len(feature_cols)

    vt = VarianceThreshold(threshold=var_threshold)
    vt.fit(X)
    kept = [c for c, keep in zip(feature_cols, vt.get_support()) if keep]
    X_var = X[kept]
    n_after_variance = len(kept)

    kept_corr = _correlation_filter(X_var, threshold=corr_threshold)
    X_corr = X_var[kept_corr]
    n_after_correlation = len(kept_corr)

    y = df[label_col].astype("category").cat.codes.values
    mi = mutual_info_classif(X_corr.fillna(0), y, random_state=cfg.get("seed", 42))
    importance = pd.DataFrame({"feature": kept_corr, "mi_score": mi}).sort_values(
        "mi_score", ascending=False
    )

    if top_k is not None:
        selected = importance.head(top_k)["feature"].tolist()
    else:
        selected = kept_corr

    return SelectionResult(
        selected_columns=selected,
        importance=importance,
        n_input=n_input,
        n_after_variance=n_after_variance,
        n_after_correlation=n_after_correlation,
        n_selected=len(selected),
    )


def save_selection_artifacts(
    df: pd.DataFrame,
    result: SelectionResult,
    dataset: str,
    experiment: str,
    meta_cols: list[str] | None = None,
) -> dict[str, Path]:
    """Write selected_features.parquet and feature_importance.csv."""
    meta_cols = meta_cols or ["participant_id", "label", "dataset_id", "dataset_name", "epoch_id"]
    keep = [c for c in meta_cols if c in df.columns] + result.selected_columns
    out_df = df[keep].copy()

    parquet_path = selected_features_path(dataset, experiment)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(parquet_path, engine="pyarrow")

    imp_path = feature_importance_path(dataset, experiment)
    result.importance.to_csv(imp_path, index=False)

    return {"selected_features": parquet_path, "feature_importance": imp_path}
