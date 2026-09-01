"""QC metric computations — read logs/checkpoints only (no reprocessing)."""

from __future__ import annotations

from typing import Any

import mne
import numpy as np

from eeg.io import list_subjects, load_checkpoint, read_json, write_json
from eeg.paths import resolve_checkpoint_path, subject_log_path

BAND_RANGES = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}

V2_TO_UV2 = 1e12  # V²/Hz → µV²/Hz
LOG_EPS_V2 = 1e-30  # floor for log10 in V²/Hz space

_LEGACY_BAND_SUFFIXES = ("_before", "_after", "_delta")
_LEGACY_PSD_KEYS = ("psd_mean_before", "psd_mean_after")


def _mean_band_power(psd_data: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return 0.0
    return float(psd_data[:, mask].mean())


def compute_band_power_from_raw(raw: mne.io.BaseRaw, fmin: float = 1, fmax: float = 40) -> dict[str, float]:
    """Mean band power (V²/Hz, MNE native) averaged across EEG channels."""
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
    """Band power before/after in µV²/Hz and log10(V²/Hz); gamma capped at 40 Hz."""
    before = compute_band_power_from_raw(raw_before)
    after = compute_band_power_from_raw(raw_after)

    before_uv2: dict[str, float] = {}
    after_uv2: dict[str, float] = {}
    delta_uv2: dict[str, float] = {}
    before_log10: dict[str, float] = {}
    after_log10: dict[str, float] = {}

    flat: dict[str, Any] = {
        "band_power": {
            "before_uv2": before_uv2,
            "after_uv2": after_uv2,
            "delta_uv2": delta_uv2,
            "before_log10": before_log10,
            "after_log10": after_log10,
        }
    }

    for band in BAND_RANGES:
        b, a = before[band], after[band]
        b_uv2 = b * V2_TO_UV2
        a_uv2 = a * V2_TO_UV2
        d_uv2 = a_uv2 - b_uv2
        b_log = float(np.log10(b + LOG_EPS_V2))
        a_log = float(np.log10(a + LOG_EPS_V2))

        before_uv2[band] = round(b_uv2, 4)
        after_uv2[band] = round(a_uv2, 4)
        delta_uv2[band] = round(d_uv2, 4)
        before_log10[band] = round(b_log, 4)
        after_log10[band] = round(a_log, 4)

        flat[f"{band}_before_uv2"] = before_uv2[band]
        flat[f"{band}_after_uv2"] = after_uv2[band]
        flat[f"{band}_delta_uv2"] = delta_uv2[band]
        flat[f"{band}_before_log10"] = before_log10[band]
        flat[f"{band}_after_log10"] = after_log10[band]

    return flat


def compute_psd_delta(raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw, fmin=1, fmax=40) -> dict:
    psd_before = raw_before.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose=False)
    psd_after = raw_after.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose=False)
    mean_before = float(np.mean(psd_before.get_data()))
    mean_after = float(np.mean(psd_after.get_data()))
    return {
        "psd_mean_before_uv2": round(mean_before * V2_TO_UV2, 4),
        "psd_mean_after_uv2": round(mean_after * V2_TO_UV2, 4),
        "psd_ratio_after_before": mean_after / mean_before if mean_before else None,
    }


def _strip_legacy_spectral_keys(clean_stage: dict[str, Any]) -> None:
    for band in BAND_RANGES:
        for suffix in _LEGACY_BAND_SUFFIXES:
            clean_stage.pop(f"{band}{suffix}", None)
    for key in _LEGACY_PSD_KEYS:
        clean_stage.pop(key, None)
    bp = clean_stage.get("band_power")
    if isinstance(bp, dict):
        for legacy in ("before", "after", "delta"):
            bp.pop(legacy, None)


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
        "psd_mean_before_uv2": log.get(
            "psd_mean_before_uv2", clean_stage.get("psd_mean_before_uv2", clean_stage.get("psd_mean_before"))
        ),
        "psd_mean_after_uv2": log.get(
            "psd_mean_after_uv2", clean_stage.get("psd_mean_after_uv2", clean_stage.get("psd_mean_after"))
        ),
        "psd_ratio_after_before": log.get(
            "psd_ratio_after_before", clean_stage.get("psd_ratio_after_before")
        ),
        "asr_enabled": clean_stage.get("asr", {}).get("asr_enabled"),
    }
    for band in BAND_RANGES:
        for suffix in ("before_uv2", "after_uv2", "delta_uv2", "before_log10", "after_log10"):
            key = f"{band}_{suffix}"
            val = log.get(key)
            if val is None:
                val = clean_stage.get(key)
            if val is None and suffix.endswith("_uv2"):
                legacy = f"{band}_{suffix.replace('_uv2', '')}"
                legacy_val = clean_stage.get(legacy)
                # ponytail: skip legacy V²/Hz keys — round(1e-11, 8) == 0.0
                if legacy_val not in (None, 0, 0.0):
                    val = legacy_val
            row[key] = val

    if log.get("status") == "error":
        for stage_name, stage_data in stages.items():
            if stage_data.get("status") == "error":
                row["error"] = stage_data.get("error", "unknown")
                row["error_stage"] = stage_name
                break
    return row


def compute_snr(
    epochs: mne.Epochs,
    signal_band: tuple[float, float] = (1.0, 30.0),
    noise_band: tuple[float, float] = (30.0, 40.0),
) -> float:
    """Estimate spectral SNR as EEG-band power / high-frequency noise power.

    This is a QC proxy for resting EEG, not reconstruction SNR: no clean
    reference signal exists at this stage. Both numerator and denominator are
    integrated Welch PSD powers, so the returned dB value has the standard
    ``10 * log10(P_signal / P_noise)`` definition.
    """
    fmin = min(signal_band[0], noise_band[0])
    fmax = max(signal_band[1], noise_band[1])
    spectrum = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose=False)
    psd, freqs = spectrum.get_data(return_freqs=True)

    def _integrated_power(band: tuple[float, float]) -> float:
        mask = (freqs >= band[0]) & (freqs < band[1])
        if mask.sum() < 2:
            return 0.0
        return float(np.trapezoid(psd[..., mask], freqs[mask], axis=-1).mean())

    signal_power = _integrated_power(signal_band)
    noise_power = _integrated_power(noise_band)
    if signal_power <= 0 or noise_power <= 0:
        return 0.0
    return float(10.0 * np.log10(signal_power / noise_power))


def load_subject_log(dataset_name: str, experiment: str, participant_id: str) -> dict[str, Any] | None:
    path = subject_log_path(dataset_name, experiment, participant_id)
    if not path.exists():
        return None
    return read_json(path)


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
        cp = resolve_checkpoint_path(dataset_name, experiment, participant_id, stage)
        if cp.exists():
            loaded[stage] = load_checkpoint(cp, stage)
    return loaded


def backfill_spectral_qc(
    dataset_name: str,
    experiment: str,
    limit: int | None = None,
) -> list[str]:
    """Recompute clean-stage spectral metrics from raw+clean checkpoints; patch logs."""
    from eeg.preprocessing import _flatten_subject_log

    logs_dir = subject_log_path(dataset_name, experiment, "sub-000").parent
    if not logs_dir.exists():
        return []

    patched: list[str] = []
    paths = sorted(logs_dir.glob("*.json"))
    if limit is not None:
        paths = paths[:limit]

    for path in paths:
        log = read_json(path)
        if log.get("status") not in ("ok", "skipped"):
            continue

        participant_id = log.get("participant_id", path.stem)
        checkpoints = load_checkpoints_for_qc(dataset_name, experiment, participant_id)
        raw = checkpoints.get("raw")
        clean = checkpoints.get("clean")
        if raw is None or clean is None:
            continue

        band = compute_band_power_table(raw, clean)
        psd = compute_psd_delta(raw, clean)

        clean_stage = log.setdefault("stages", {}).setdefault("clean", {})
        _strip_legacy_spectral_keys(clean_stage)
        clean_stage.update(band)
        clean_stage.update(psd)

        _flatten_subject_log(log, dataset_name, experiment)
        write_json(path, log)
        patched.append(participant_id)

    return patched
