"""Reproducible follow-up analyses for the classical EEG benchmark.

This module deliberately keeps every comparison on the same nested,
subject-grouped evaluation protocol. It is an experiment runner, not a second
evaluation implementation: each run delegates model fitting and scoring to
``run_benchmark``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg.config import load_experiment, resolve_dataset
from eeg.io import load_features_df, write_json
from eeg.paths import results_dir
from eeg.training.benchmark import (
    _candidate_features,
    run_benchmark,
)


SPECTRAL_FEATURES = [
    "rel_alpha",
    "rel_beta",
    "rel_theta",
    "rel_delta",
    "alpha_peak_freq",
    "theta_alpha_ratio",
    "theta_beta_ratio",
    "slow_fast_ratio",
]
CONNECTIVITY_FEATURES = ["theta_wpli", "alpha_wpli"]
COMPLEXITY_FEATURES = ["lzc_posterior", "mse_posterior"]


def default_feature_groups() -> dict[str, list[str]]:
    """Return interpretable groups for the current 12-feature schema."""
    return {
        "band_delta": ["rel_delta"],
        "band_theta": ["rel_theta"],
        "band_alpha": ["rel_alpha"],
        "band_beta": ["rel_beta"],
        "spectral": SPECTRAL_FEATURES,
        "connectivity": CONNECTIVITY_FEATURES,
        "complexity": COMPLEXITY_FEATURES,
        "posterior_region": COMPLEXITY_FEATURES,
        "global_features": SPECTRAL_FEATURES + CONNECTIVITY_FEATURES,
    }


def resolve_feature_groups(
    feature_columns: list[str], config: dict[str, Any] | None = None
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Resolve configured groups and explain groups unavailable in the table."""
    cfg = config or {}
    analysis_cfg = (
        (cfg.get("analysis") or cfg.get("training", {}).get("analysis", {}))
        if isinstance(cfg, dict)
        else {}
    )
    configured = analysis_cfg.get("feature_groups", {})
    groups = default_feature_groups()
    if isinstance(configured, dict):
        groups.update(
            {str(name): list(columns) for name, columns in configured.items()}
        )

    available: dict[str, list[str]] = {}
    unavailable: dict[str, str] = {}
    feature_set = set(feature_columns)
    for name, columns in groups.items():
        resolved = [column for column in columns if column in feature_set]
        if resolved:
            available[name] = resolved
        else:
            unavailable[name] = "No columns from this group exist in the feature table."

    channel_groups = analysis_cfg.get("channel_groups", {})
    if channel_groups:
        for name, columns in channel_groups.items():
            resolved = [column for column in columns if column in feature_set]
            if resolved:
                available[str(name)] = resolved
            else:
                unavailable[str(name)] = "Configured channel group has no matching columns."
    elif not any("_" in column and column.rsplit("_", 1)[-1] in {"Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"} for column in feature_columns):
        unavailable["channel_level"] = (
            "The current feature parquet contains posterior aggregates and global "
            "connectivity, not channel-specific features; channel ablations require "
            "channel-level extraction first."
        )
    return available, unavailable


def _selection_config(config: dict[str, Any], **updates: Any) -> dict[str, Any]:
    result = deepcopy(config)
    selection = dict(result.get("feature_selection", {}))
    selection.update(updates)
    result["feature_selection"] = selection
    return result


def _record_run(
    manifest: dict[str, Any], name: str, kind: str, result: dict[str, Any]
) -> None:
    row = result["rows"][0] if result.get("rows") else {}
    manifest.setdefault("runs", []).append(
        {
            "name": name,
            "kind": kind,
            "benchmark_csv": result.get("benchmark_csv"),
            "row": row,
        }
    )


def _run_permutation_test(
    dataset: str,
    experiment: str,
    config: dict[str, Any],
    observed: float,
    *,
    n_permutations: int,
    cv_folds: int | None,
    threshold_optimization: bool,
    calibration_method: str | None,
) -> dict[str, Any]:
    seed = int(config.get("seed", config.get("training", {}).get("random_state", 42)))
    rows: list[dict[str, Any]] = []
    for permutation_idx in range(n_permutations):
        result = run_benchmark(
            dataset=dataset,
            experiment=experiment,
            models=["logistic_regression"],
            config=config,
            cv_folds=cv_folds,
            threshold_optimization=threshold_optimization,
            calibration_method=calibration_method,
            label_permutation_seed=seed + permutation_idx + 1,
            output_tag="permutation_test",
            write_artifacts=False,
        )
        row = dict(result["rows"][0])
        row["permutation"] = permutation_idx + 1
        rows.append(row)

    null_scores = np.asarray([row["balanced_accuracy"] for row in rows], dtype=float)
    p_value = (
        float((1 + np.count_nonzero(null_scores >= observed)) / (len(null_scores) + 1))
        if len(null_scores)
        else None
    )
    return {
        "n_permutations": int(n_permutations),
        "observed_balanced_accuracy": float(observed),
        "null_mean": float(null_scores.mean()) if len(null_scores) else None,
        "null_sd": float(null_scores.std(ddof=1)) if len(null_scores) > 1 else None,
        "null_q025": float(np.quantile(null_scores, 0.025)) if len(null_scores) else None,
        "null_q975": float(np.quantile(null_scores, 0.975)) if len(null_scores) else None,
        "p_value_greater_equal_observed": p_value,
        "rows": rows,
    }


def _write_markdown_report(manifest: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Extended analysis: {manifest['dataset']} / {manifest['experiment']}",
        "",
        "All reported model comparisons use subject-stratified grouped nested CV. "
        "Thresholds and model hyperparameters are selected only within inner folds.",
        "",
        "## Results",
        "",
        "| Analysis | Kind | Model | Balanced accuracy | Macro F1 | ROC AUC | PR AUC | MCC |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in manifest.get("runs", []):
        row = run.get("row", {})
        lines.append(
            "| {name} | {kind} | {model} | {ba:.3f} | {f1} | {roc} | {pr} | {mcc} |".format(
                name=run.get("name", ""),
                kind=run.get("kind", ""),
                model=row.get("model", ""),
                ba=float(row.get("balanced_accuracy", float("nan"))),
                f1=_format_metric(row.get("macro_f1")),
                roc=_format_metric(row.get("macro_roc_auc")),
                pr=_format_metric(row.get("macro_pr_auc")),
                mcc=_format_metric(row.get("mcc")),
            )
        )
    permutation = manifest.get("permutation_test", {})
    lines.extend(
        [
            "",
            "## Permutation test",
            "",
            f"- Permutations: {permutation.get('n_permutations', 0)}",
            f"- Observed balanced accuracy: {_format_metric(permutation.get('observed_balanced_accuracy'))}",
            f"- Null mean (SD): {_format_metric(permutation.get('null_mean'))} "
            f"({_format_metric(permutation.get('null_sd'))})",
            f"- Empirical p-value (null ≥ observed): {_format_metric(permutation.get('p_value_greater_equal_observed'))}",
        ]
    )
    unavailable = manifest.get("unavailable_groups", {})
    if unavailable:
        lines.extend(["", "## Unavailable groups", ""])
        lines.extend(f"- **{name}:** {reason}" for name, reason in unavailable.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metric(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{float(value):.3f}"


def run_extended_analysis(
    dataset: str,
    experiment: str = "baseline",
    config: dict[str, Any] | None = None,
    *,
    cv_folds: int | None = None,
    n_permutations: int | None = None,
) -> dict[str, Any]:
    """Run threshold/calibration, ablation, regularization, selection, and null tests."""
    cfg = deepcopy(config or load_experiment(experiment))
    cfg.setdefault("seed", cfg.get("training", {}).get("random_state", 42))
    analysis_cfg = cfg.get("analysis") or cfg.get("training", {}).get("analysis", {})
    threshold = bool(analysis_cfg.get("threshold_optimization", True))
    calibration = analysis_cfg.get("calibration_method", "sigmoid")
    permutations = (
        int(analysis_cfg.get("permutations", 1000))
        if n_permutations is None
        else int(n_permutations)
    )
    if permutations < 0:
        raise ValueError("n_permutations must be non-negative.")

    spec = resolve_dataset(dataset)[0]
    frame = load_features_df(dataset, experiment, spec.id).reset_index(drop=True)
    candidates = _candidate_features(frame, cfg, None)
    groups, unavailable = resolve_feature_groups(candidates, cfg)

    manifest: dict[str, Any] = {
        "dataset": dataset,
        "experiment": experiment,
        "candidate_features": candidates,
        "feature_groups": groups,
        "unavailable_groups": unavailable,
        "protocol": {
            "outer_cv": "subject-stratified grouped",
            "threshold_objective": "balanced_accuracy",
            "threshold_tuning": "inner out-of-fold subject probabilities",
            "calibration": calibration,
        },
    }

    threshold_result = run_benchmark(
        dataset=dataset,
        experiment=experiment,
        models=["logistic_regression"],
        config=cfg,
        cv_folds=cv_folds,
        threshold_optimization=threshold,
        calibration_method=calibration,
        output_tag="threshold_calibration",
    )
    _record_run(manifest, "logistic_regression_threshold_calibrated", "threshold_calibration", threshold_result)

    regularization_models = analysis_cfg.get(
        "regularization_models",
        [
            "logistic_regression",
            "logistic_regression_l1",
            "logistic_regression_elasticnet",
        ],
    )
    regularization = run_benchmark(
        dataset=dataset,
        experiment=experiment,
        models=list(regularization_models),
        config=cfg,
        cv_folds=cv_folds,
        output_tag="regularization",
    )
    for row in regularization.get("rows", []):
        manifest.setdefault("runs", []).append(
            {
                "name": row["model"],
                "kind": "regularization",
                "benchmark_csv": regularization.get("benchmark_csv"),
                "row": row,
            }
        )

    for name, columns in groups.items():
        result = run_benchmark(
            dataset=dataset,
            experiment=experiment,
            models=["logistic_regression"],
            feature_cols=columns,
            config=cfg,
            cv_folds=cv_folds,
            output_tag=f"ablation_{name}",
        )
        _record_run(manifest, name, "feature_ablation", result)

    top_k_values = analysis_cfg.get("univariate_top_k", [3, 5, 8, 12])
    for top_k in top_k_values:
        top_k = int(top_k)
        if not 1 <= top_k <= len(candidates):
            continue
        result = run_benchmark(
            dataset=dataset,
            experiment=experiment,
            models=["logistic_regression"],
            config=_selection_config(cfg, top_k=top_k),
            cv_folds=cv_folds,
            output_tag=f"selection_univariate_top_{top_k}",
        )
        _record_run(manifest, f"univariate_top_{top_k}", "feature_selection", result)

    for model_key, name in [
        ("logistic_regression_l1_select", "l1_selection"),
        ("logistic_regression_pca", "pca"),
    ]:
        result = run_benchmark(
            dataset=dataset,
            experiment=experiment,
            models=[model_key],
            config=_selection_config(cfg, top_k=None),
            cv_folds=cv_folds,
            output_tag=f"selection_{name}",
        )
        _record_run(manifest, name, "feature_selection", result)

    observed = threshold_result["rows"][0]["balanced_accuracy"]
    permutation = _run_permutation_test(
        dataset,
        experiment,
        cfg,
        observed,
        n_permutations=permutations,
        cv_folds=cv_folds,
        threshold_optimization=threshold,
        calibration_method=calibration,
    )
    manifest["permutation_test"] = {
        key: value for key, value in permutation.items() if key != "rows"
    }

    analysis_dir = results_dir(dataset, experiment) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [
        {**run["row"], "analysis_name": run["name"], "analysis_kind": run["kind"]}
        for run in manifest.get("runs", [])
        if run.get("row")
    ]
    pd.DataFrame(summary_rows).to_csv(analysis_dir / "analysis_summary.csv", index=False)
    pd.DataFrame(permutation["rows"]).to_csv(analysis_dir / "permutation_scores.csv", index=False)
    write_json(analysis_dir / "extended_analysis.json", manifest)
    _write_markdown_report(manifest, analysis_dir / "extended_analysis.md")
    return {
        "analysis_summary": str(analysis_dir / "analysis_summary.csv"),
        "permutation_scores": str(analysis_dir / "permutation_scores.csv"),
        "report": str(analysis_dir / "extended_analysis.md"),
        "manifest": manifest,
    }
