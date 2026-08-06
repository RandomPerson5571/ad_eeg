"""QC metric computations — read logs/checkpoints only (no reprocessing)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mne
import numpy as np

from eeg.io import list_subjects, load_checkpoint, read_json
from eeg.paths import checkpoint_path, qc_report_dir, subject_log_path

BAND_RANGES = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}


def _mean_band_power(psd_data: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return 0.0
    return float(psd_data[:, mask].mean())


def compute_band_power_from_raw(raw: mne.io.BaseRaw, fmin: float = 1, fmax: float = 40) -> dict[str, float]:
    """Mean band power (µV²/Hz) averaged across EEG channels."""
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        return {band: 0.0 for band in BAND_RANGES}
    psd = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax, picks=picks, verbose=False)
    data = psd.get_data().mean(axis=0)
    freqs = psd.freqs
    return {
        band: _mean_band_power(data[np.newaxis, :], freqs, lo, hi)
        for band, (lo, hi) in BAND_RANGES.items()
    }


def compute_band_power_table(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
) -> dict[str, Any]:
    """Band power before/after and deltas (gamma capped at 40 Hz filter)."""
    before = compute_band_power_from_raw(raw_before)
    after = compute_band_power_from_raw(raw_after)
    flat: dict[str, Any] = {"band_power": {"before": before, "after": after, "delta": {}}}
    for band in BAND_RANGES:
        b, a = before[band], after[band]
        flat[f"{band}_before"] = round(b, 8)
        flat[f"{band}_after"] = round(a, 8)
        flat[f"{band}_delta"] = round(a - b, 8)
        flat["band_power"]["delta"][band] = round(a - b, 8)
    return flat


def compute_psd_delta(raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw, fmin=1, fmax=40) -> dict:
    psd_before = raw_before.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose=False)
    psd_after = raw_after.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose=False)
    mean_before = float(np.mean(psd_before.get_data()))
    mean_after = float(np.mean(psd_after.get_data()))
    return {
        "psd_mean_before": mean_before,
        "psd_mean_after": mean_after,
        "psd_ratio_after_before": mean_after / mean_before if mean_before else None,
    }


def compute_snr(epochs: mne.Epochs) -> float:
    data = epochs.get_data()
    signal_power = np.mean(data**2)
    noise_floor = np.percentile(np.abs(data), 10) ** 2
    return float(10 * np.log10(signal_power / noise_floor)) if noise_floor > 0 else 0.0


def load_subject_log(dataset_name: str, experiment: str, participant_id: str) -> dict[str, Any] | None:
    path = subject_log_path(dataset_name, experiment, participant_id)
    if not path.exists():
        return None
    return read_json(path)


def log_to_summary_row(log: dict[str, Any]) -> dict[str, Any]:
    """Flatten a subject preprocessing log into a CSV-friendly row."""
    stages = log.get("stages", {})
    raw_stage = stages.get("raw", {})
    ica_stage = stages.get("ica", {})
    clean_stage = stages.get("clean", {})
    epoch_stage = stages.get("epochs", {})

    n_before = log.get("n_epochs_before_ar", epoch_stage.get("n_epochs_before_ar"))
    n_rejected = log.get("n_epochs_rejected", epoch_stage.get("n_epochs_rejected", 0))
    pct_rej = log.get("pct_epochs_rejected")
    if pct_rej is None and n_before:
        pct_rej = round(100 * (n_rejected or 0) / n_before, 2)

    row: dict[str, Any] = {
        "participant_id": log.get("participant_id"),
        "dataset_name": log.get("dataset_name"),
        "experiment": log.get("experiment"),
        "status": log.get("status"),
        "duration_s": log.get("duration_s", raw_stage.get("duration_s")),
        "sfreq": log.get("sfreq", raw_stage.get("sfreq")),
        "n_channels": log.get("n_channels", raw_stage.get("n_channels")),
        "n_eeg_channels": log.get("n_eeg_channels", raw_stage.get("n_eeg_channels")),
        "n_bad_channels": len(log.get("bad_channels", clean_stage.get("bad_channels", []))),
        "bad_channels": ";".join(log.get("bad_channels", clean_stage.get("bad_channels", []))),
        "ica_n_components_fitted": log.get(
            "ica_n_components_fitted", ica_stage.get("ica_n_components_fitted", log.get("ica_n_components"))
        ),
        "ica_n_removed": log.get("ica_n_removed", ica_stage.get("ica_n_removed", 0)),
        "ica_eeg_rank": log.get("ica_eeg_rank", ica_stage.get("ica_eeg_rank")),
        "n_epochs_before_ar": n_before,
        "n_epochs_after_ar": log.get("n_epochs_after_ar", epoch_stage.get("n_epochs_after_ar")),
        "n_epochs_rejected": n_rejected,
        "pct_epochs_rejected": pct_rej,
        "snr_db": log.get("snr_db", epoch_stage.get("snr_db")),
        "runtime_seconds": log.get("runtime_seconds"),
        "psd_mean_before": log.get("psd_mean_before"),
        "psd_mean_after": log.get("psd_mean_after"),
        "asr_enabled": clean_stage.get("asr", {}).get("asr_enabled"),
    }
    for band in BAND_RANGES:
        row[f"{band}_before"] = log.get(f"{band}_before", clean_stage.get(f"{band}_before"))
        row[f"{band}_after"] = log.get(f"{band}_after", clean_stage.get(f"{band}_after"))
        row[f"{band}_delta"] = log.get(f"{band}_delta", clean_stage.get(f"{band}_delta"))

    if log.get("status") == "error":
        for stage_name, stage_data in stages.items():
            if stage_data.get("status") == "error":
                row["error"] = stage_data.get("error", "unknown")
                row["error_stage"] = stage_name
                break
    return row


def metrics_from_log(log: dict[str, Any], dataset_spec=None, subject_num: int | None = None) -> dict[str, Any]:
    """Build QC metrics dict from an existing subject log (no reprocessing)."""
    row = log_to_summary_row(log)
    if dataset_spec is not None and subject_num is not None:
        participants = list_subjects(dataset_spec)
        prow = participants.iloc[subject_num - 1]
        row["label"] = prow.get("Group", "")
        row["dataset_id"] = dataset_spec.id
        row["subject_num"] = subject_num
    return row


def preprocessing_metrics(
    dataset_spec,
    subject_num: int,
    experiment: str = "baseline",
) -> dict[str, Any]:
    """Return QC metrics from preprocessed subject log (requires prior preprocessing run)."""
    participants = list_subjects(dataset_spec)
    participant_id = participants.iloc[subject_num - 1]["participant_id"]
    log = load_subject_log(dataset_spec.name, experiment, participant_id)
    if log is None:
        raise FileNotFoundError(
            f"No preprocessing log for {participant_id}. "
            f"Run preprocess_dataset.py first."
        )
    return metrics_from_log(log, dataset_spec, subject_num)


def load_checkpoints_for_qc(
    dataset_name: str,
    experiment: str,
    participant_id: str,
) -> dict[str, mne.io.BaseRaw | mne.Epochs]:
    """Load available checkpoints for visualization (no stage re-execution)."""
    loaded: dict[str, mne.io.BaseRaw | mne.Epochs] = {}
    for stage in ("raw", "filtered", "clean", "epochs"):
        cp = checkpoint_path(dataset_name, experiment, participant_id, stage)
        if cp.exists():
            loaded[stage] = load_checkpoint(cp, stage)
    return loaded
