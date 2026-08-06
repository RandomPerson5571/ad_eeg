"""Tests for preprocessing QC reporting (no MNE preprocessing runs)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg.preprocess_report import (
    build_summary_dataframe,
    build_summary_json,
    compare_experiments,
    compute_distribution,
    detect_outliers,
    render_dataset_report_md,
    render_html_dashboard,
    write_preprocess_report,
)
from eeg.preprocessing import _resolve_ica_n_components
from eeg.qc import backfill_spectral_qc, compute_band_power_table, log_to_summary_row
from eeg.repro import preprocessing_fingerprint


def _sample_log(participant_id: str = "sub-001", status: str = "ok") -> dict:
    return {
        "participant_id": participant_id,
        "dataset_name": "eyesclosed",
        "experiment": "baseline",
        "status": status,
        "runtime_seconds": 42.5,
        "stages": {
            "raw": {
                "duration_s": 300.0,
                "sfreq": 500.0,
                "n_channels": 19,
                "n_eeg_channels": 19,
            },
            "ica": {
                "ica_n_components_fitted": 18,
                "ica_n_removed": 2,
                "ica_eeg_rank": 18,
                "ica_n_iter": 50,
            },
            "clean": {
                "bad_channels": ["Fp1"],
                "delta_before_uv2": 1.0,
                "delta_after_uv2": 0.9,
                "delta_delta_uv2": -0.1,
                "alpha_before_uv2": 12.48,
                "alpha_after_uv2": 11.93,
                "alpha_delta_uv2": -0.55,
                "alpha_before_log10": -7.42,
                "alpha_after_log10": -7.44,
            },
            "epochs": {
                "n_epochs_before_ar": 100,
                "n_epochs_after_ar": 90,
                "n_epochs_rejected": 10,
                "pct_epochs_rejected": 10.0,
                "snr_db": 15.5,
            },
        },
    }


def test_resolve_ica_n_components_auto():
    import mne

    info = mne.create_info(["E1", "E2", "E3"], sfreq=250, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((3, 500)), info)
    n, requested = _resolve_ica_n_components(raw, "auto")
    assert n == 2
    assert requested == "auto"


def test_log_to_summary_row():
    row = log_to_summary_row(_sample_log())
    assert row["participant_id"] == "sub-001"
    assert row["ica_n_components_fitted"] == 18
    assert row["ica_n_removed"] == 2
    assert row["pct_epochs_rejected"] == 10.0
    assert row["n_bad_channels"] == 1
    assert row["alpha_before_uv2"] == 12.48
    assert row["alpha_delta_uv2"] == -0.55
    assert row["alpha_before_log10"] == -7.42


def test_log_to_summary_row_ignores_legacy_zero_band_keys():
    log = _sample_log()
    clean = log["stages"]["clean"]
    for band in ("delta", "theta", "alpha", "beta", "gamma"):
        for suffix in ("_before_uv2", "_after_uv2", "_delta_uv2"):
            clean.pop(f"{band}{suffix}", None)
        clean[f"{band}_before"] = 0.0
    row = log_to_summary_row(log)
    assert row["alpha_before_uv2"] is None
    assert row["alpha_delta_uv2"] is None


def test_v2_band_power_rounds_to_zero():
    assert round(8.6e-12, 8) == 0.0
    assert round(8.6e-12 * 1e12, 4) == 8.6


def test_compute_distribution():
    dist = compute_distribution([1.0, 2.0, 3.0, 4.0, 100.0])
    assert dist["count"] == 5
    assert dist["median"] == 3.0
    assert dist["min"] == 1.0
    assert dist["max"] == 100.0
    assert len(dist["histogram"]["counts"]) >= 5


def test_detect_outliers_rejection():
    base = [
        {"participant_id": f"sub-{i:03d}", "pct_epochs_rejected": 5.0 + i * 0.1, "ica_n_components_fitted": 18, "ica_n_removed": 1}
        for i in range(1, 8)
    ]
    base.append(
        {"participant_id": "sub-099", "pct_epochs_rejected": 50.0, "ica_n_components_fitted": 18, "ica_n_removed": 1}
    )
    df = pd.DataFrame(base)
    outliers = detect_outliers(df)
    assert any(o["participant_id"] == "sub-099" for o in outliers)


def test_preprocessing_fingerprint_keys():
    fp = preprocessing_fingerprint({"experiment": {"preprocessing": {}}})
    for key in (
        "mne_version",
        "autoreject_version",
        "asrpy_version",
        "git_commit",
        "python_version",
    ):
        assert key in fp
    assert "config_sha256" in preprocessing_fingerprint({"a": 1})


def test_write_preprocess_report(tmp_path, monkeypatch):
    dataset = "eyesclosed"
    experiment = "baseline"
    logs_dir = tmp_path / "data" / "preprocessed" / dataset / experiment / "logs"
    logs_dir.mkdir(parents=True)
    for i in range(1, 4):
        log = _sample_log(f"sub-{i:03d}")
        log["pct_epochs_rejected"] = i * 5.0
        log["stages"]["epochs"]["pct_epochs_rejected"] = i * 5.0
        (logs_dir / f"sub-{i:03d}.json").write_text(json.dumps(log), encoding="utf-8")

    import eeg.preprocess_report as pr

    monkeypatch.setattr(pr, "backfill_spectral_qc", lambda d, e, limit=None: [])
    monkeypatch.setattr(pr, "load_subject_logs", lambda d, e: [_sample_log(f"sub-{i:03d}") for i in range(1, 4)])
    monkeypatch.setattr(pr, "qc_report_dir", lambda d, e: tmp_path / "data" / "preprocessed" / d / e / "qc")

    config = {"experiment": {"preprocessing": {"ica_n_components": "auto"}}}
    paths = write_preprocess_report(dataset, experiment, config=config)
    assert paths["summary_csv"].exists()
    assert paths["summary_json"].exists()
    assert paths["dataset_report_md"].exists()
    assert paths["index_html"].exists()

    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["n_subjects"]["ok"] == 3
    assert "pct_epochs_rejected" in summary["distributions"]
    assert summary["distributions"]["pct_epochs_rejected"]["median"] is not None

    md = paths["dataset_report_md"].read_text(encoding="utf-8")
    assert "Fingerprint" in md
    assert "Metric Distributions" in md


def test_render_html_dashboard(tmp_path):
    df = build_summary_dataframe([_sample_log()])
    html = render_html_dashboard(df, tmp_path)
    assert "sub-001" in html
    assert "qc-table" in html


def test_compare_experiments(tmp_path, monkeypatch):
    import eeg.preprocess_report as pr

    for exp in ("baseline", "strict"):
        qc_dir = tmp_path / "data" / "preprocessed" / "eyesclosed" / exp / "qc"
        qc_dir.mkdir(parents=True)
        summary = {
            "distributions": {
                "pct_epochs_rejected": {"median": 10.0 if exp == "baseline" else 5.0},
                "runtime_seconds": {"median": 40.0},
                "ica_n_removed": {"median": 2.0},
                "alpha_delta_uv2": {"mean": 0.1},
            }
        }
        (qc_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    monkeypatch.setattr(
        pr,
        "qc_report_dir",
        lambda d, e: tmp_path / "data" / "preprocessed" / d / e / "qc",
    )
    out = tmp_path / "comparison.csv"
    df = compare_experiments("eyesclosed", ["baseline", "strict"], output_path=out)
    assert out.exists()
    assert "baseline" in df.columns
    assert len(df) == 4


def test_compute_band_power_table():
    import mne

    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 2000)) * 1e-6
    info = mne.create_info(["E1", "E2", "E3", "E4"], sfreq=500, ch_types="eeg")
    raw_a = mne.io.RawArray(data, info)
    raw_b = mne.io.RawArray(data * 0.5, info)
    result = compute_band_power_table(raw_a, raw_b)
    assert result["alpha_before_uv2"] > 0
    assert result["alpha_delta_uv2"] != 0
    assert np.isfinite(result["alpha_before_log10"])
    assert result["alpha_before_log10"] < 0
    assert result["band_power"]["before_uv2"]["delta"] > 0


def test_backfill_spectral_qc(tmp_path, monkeypatch):
    import mne

    from eeg.io import save_checkpoint
    import eeg.paths as paths  # noqa: F401 — ensure paths module loaded

    dataset = "eyesclosed"
    experiment = "baseline"
    participant_id = "sub-001"

    pre_dir = tmp_path / "data" / "preprocessed" / dataset / experiment
    pre_dir.mkdir(parents=True)
    logs_dir = pre_dir / "logs"
    logs_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 2000)) * 1e-6
    info = mne.create_info(["E1", "E2", "E3", "E4"], sfreq=500, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    clean = mne.io.RawArray(data * 0.8, info)

    suffix = {"raw": "_raw.fif", "clean": "_clean_raw.fif"}

    def _subject_log_path(d, e, pid, stage="preprocessed"):
        return logs_dir / f"{pid}.json"

    def _resolve_checkpoint_path(d, e, pid, stage):
        if stage not in suffix:
            return pre_dir / f"{pid}_missing_{stage}.fif"
        return pre_dir / f"{pid}{suffix[stage]}"

    monkeypatch.setattr("eeg.qc.subject_log_path", _subject_log_path)
    monkeypatch.setattr("eeg.qc.resolve_checkpoint_path", _resolve_checkpoint_path)
    monkeypatch.setattr("eeg.preprocessing.subject_log_path", _subject_log_path)

    save_checkpoint(raw, pre_dir / f"{participant_id}_raw.fif")
    save_checkpoint(clean, pre_dir / f"{participant_id}_clean_raw.fif")

    log = _sample_log(participant_id)
    log["stages"]["clean"] = {"bad_channels": [], "alpha_before": 0.0}
    (logs_dir / f"{participant_id}.json").write_text(json.dumps(log), encoding="utf-8")

    patched = backfill_spectral_qc(dataset, experiment)
    assert patched == [participant_id]

    updated = json.loads((logs_dir / f"{participant_id}.json").read_text(encoding="utf-8"))
    clean_stage = updated["stages"]["clean"]
    assert "alpha_before" not in clean_stage
    assert clean_stage["alpha_before_uv2"] > 0
    assert updated["alpha_before_uv2"] > 0
