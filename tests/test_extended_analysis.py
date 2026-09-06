from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import eeg.training.extended_analysis as analysis


def test_resolve_feature_groups_reports_missing_channel_level_features():
    groups, unavailable = analysis.resolve_feature_groups(
        ["rel_delta", "rel_alpha", "theta_wpli", "lzc_posterior", "mse_posterior"]
    )
    assert groups["band_delta"] == ["rel_delta"]
    assert groups["connectivity"] == ["theta_wpli"]
    assert "channel_level" in unavailable


def test_extended_analysis_orchestrates_all_requested_families(tmp_path, monkeypatch):
    columns = [
        "lzc_posterior",
        "mse_posterior",
        "rel_alpha",
        "rel_beta",
        "rel_theta",
        "rel_delta",
        "alpha_peak_freq",
        "theta_alpha_ratio",
        "theta_beta_ratio",
        "slow_fast_ratio",
        "theta_wpli",
        "alpha_wpli",
    ]
    frame = pd.DataFrame([{column: 0.0 for column in columns}])
    calls = []

    monkeypatch.setattr(analysis, "resolve_dataset", lambda _: [SimpleNamespace(id=2)])
    monkeypatch.setattr(analysis, "load_features_df", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(
        analysis,
        "results_dir",
        lambda dataset, experiment: Path(tmp_path) / dataset / experiment,
    )

    def fake_run_benchmark(**kwargs):
        calls.append(kwargs)
        models = kwargs["models"]
        is_permutation = kwargs.get("label_permutation_seed") is not None
        rows = [
            {
                "model": model,
                "balanced_accuracy": 0.5 if is_permutation else 0.6,
                "dataset": kwargs["dataset"],
            }
            for model in models
        ]
        return {"benchmark_csv": "synthetic.csv", "rows": rows}

    monkeypatch.setattr(analysis, "run_benchmark", fake_run_benchmark)
    result = analysis.run_extended_analysis(
        "eyesclosed",
        config={
            "seed": 42,
            "features": {"feature_columns": columns},
            "training": {"analysis": {"permutations": 3}},
        },
        n_permutations=3,
    )

    kinds = {run["kind"] for run in result["manifest"]["runs"]}
    assert {"threshold_calibration", "regularization", "feature_ablation", "feature_selection"} <= kinds
    assert result["manifest"]["permutation_test"]["n_permutations"] == 3
    assert result["manifest"]["permutation_test"]["p_value_greater_equal_observed"] == 0.25
    assert any(call.get("label_permutation_seed") == 43 for call in calls)
    assert (tmp_path / "eyesclosed" / "baseline" / "analysis" / "analysis_summary.csv").exists()
    assert (tmp_path / "eyesclosed" / "baseline" / "analysis" / "extended_analysis.md").exists()
