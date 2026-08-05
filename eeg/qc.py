"""QC metric computations (no plotting)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np

from eeg.config import load_base_configs, load_experiment
from eeg.io import list_subjects, read_eeg_data
from eeg.paths import raw_eeg_path
from eeg.preprocessing import convert_to_epochs, preprocess_EEG, stage_filtered


def _subject_row(dataset_spec, subject_num: int):
    participants = list_subjects(dataset_spec)
    row = participants.iloc[subject_num - 1]
    return row["participant_id"], row["Group"]


def compute_psd_delta(raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw, fmin=1, fmax=40) -> dict:
    psd_before = raw_before.compute_psd(method="welch", fmin=fmin, fmax=fmax)
    psd_after = raw_after.compute_psd(method="welch", fmin=fmin, fmax=fmax)
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


def preprocessing_metrics(
    dataset_spec,
    subject_num: int,
    experiment: str = "baseline",
) -> dict[str, Any]:
    """Run preprocessing and return QC metrics without plotting."""
    base = load_base_configs()
    sfreq = base["features"]["sampling_rate"]
    config = load_experiment(experiment)
    participant_id, label = _subject_row(dataset_spec, subject_num)
    path = raw_eeg_path(dataset_spec, subject_num)
    raw = read_eeg_data(path, sfreq=sfreq)

    raw_copy = raw.copy()
    if raw_copy.info["sfreq"] != sfreq:
        raw_copy.resample(sfreq)
    n_epochs_before = len(convert_to_epochs(raw_copy))

    prep = config["experiment"]["preprocessing"]
    filtered, _ = stage_filtered(raw, config)
    clean_eeg, epochs, reject_log, meta = preprocess_EEG(
        raw,
        freq_filter=prep.get("freq_filter", True),
        notch_filter=prep.get("notch_filter", False),
        bad_channels=prep.get("bad_channels", False),
        referencing=prep.get("referencing", False),
        asr=prep.get("asr", False),
        asr_cutoff=prep.get("asr_cutoff", 17),
        run_ica=prep.get("run_ica", False),
        AR=prep.get("AR", True),
        fle=True,
        return_reject_log=True,
    )

    n_rejected = int(reject_log.bad_epochs.sum()) if reject_log is not None else 0
    psd_delta = compute_psd_delta(raw, filtered)

    return {
        "dataset_name": dataset_spec.name,
        "dataset_id": dataset_spec.id,
        "subject_num": subject_num,
        "participant_id": participant_id,
        "label": label,
        "raw_file": str(path),
        "duration_seconds": round(raw.n_times / raw.info["sfreq"], 2),
        "n_channels": len(raw.ch_names),
        "sfreq_hz": raw.info["sfreq"],
        "bad_channels_detected": meta.get("bad_channels", []),
        "n_bad_channels": len(meta.get("bad_channels", [])),
        "n_epochs_before_ar": n_epochs_before,
        "n_epochs_after_ar": len(epochs),
        "n_epochs_rejected": n_rejected,
        "pct_epochs_rejected": round(100 * n_rejected / max(n_epochs_before, 1), 2),
        "snr_db": compute_snr(epochs),
        "epoch_shape": list(epochs.get_data().shape[1:]),
        **psd_delta,
    }
