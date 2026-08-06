"""Staged EEG preprocessing with checkpoint support."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np

from asrpy import ASR
from mne.preprocessing import ICA

import autoreject

from eeg.config import load_base_configs
from eeg.io import (
    load_checkpoint,
    read_eeg_data,
    save_checkpoint,
    sha256_file,
    try_load_checkpoint,
    write_json,
)
from eeg.paths import checkpoint_path, resolve_checkpoint_path, subject_log_path
from eeg.qc import compute_band_power_table, compute_psd_delta, compute_snr


@dataclass
class PreprocessResult:
    participant_id: str
    status: str
    log: dict[str, Any] = field(default_factory=dict)
    epochs: mne.Epochs | None = None


def _cfg_value(config: dict, *keys, default=None):
    cur = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


_LEGACY_1020_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def _standardize_channel_names(raw: mne.io.BaseRaw) -> None:
    """Map legacy 10-20 aliases (T3/T4/T5/T6) to standard_1020 names."""
    mapping = {ch: _LEGACY_1020_ALIASES[ch] for ch in raw.ch_names if ch in _LEGACY_1020_ALIASES}
    if mapping:
        raw.rename_channels(mapping)


def apply_standard_montage(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, dict]:
    """Standardize channel names and apply 10-20 montage."""
    _standardize_channel_names(raw)
    raw.set_montage("standard_1020", on_missing="warn")
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    meta: dict[str, Any] = {"montage_set": True, "n_eeg_channels": len(picks)}
    if len(picks) != 19:
        meta["warnings"] = [f"Expected 19 EEG channels, found {len(picks)}."]
    return raw, meta


def detect_bad_channels(raw, flat_std=1e-15, noisy_z=5.0):
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        return []
    data = raw.get_data(picks=picks)
    ch_names = [raw.ch_names[i] for i in picks]
    stds = np.std(data, axis=1)
    med_std = np.median(stds) or 1e-15
    bads = []
    for ch, std in zip(ch_names, stds):
        if std < flat_std or std > med_std * noisy_z:
            bads.append(ch)
    return bads


def convert_to_epochs(clean_eeg, epoch_length=4.0, epoch_overlap=2.0):
    return mne.make_fixed_length_epochs(
        clean_eeg, duration=epoch_length, overlap=epoch_overlap, preload=True
    )


def stage_raw(raw_path: Path, sfreq: int) -> tuple[mne.io.BaseRaw, dict]:
    raw = read_eeg_data(raw_path, sfreq=sfreq)
    if raw.info["sfreq"] != sfreq:
        raw.resample(sfreq)
    _, montage_meta = apply_standard_montage(raw)
    meta = {
        "duration_s": round(raw.n_times / raw.info["sfreq"], 2),
        "sfreq": float(raw.info["sfreq"]),
        "n_channels": len(raw.ch_names),
        **montage_meta,
    }
    return raw, meta


def stage_filtered(raw: mne.io.BaseRaw, config: dict) -> tuple[mne.io.BaseRaw, dict]:
    prep = _cfg_value(config, "experiment", "preprocessing", default={})
    filt = _cfg_value(config, "experiment", "filtering", default=_cfg_value(config, "filtering", default={}))
    bc = _cfg_value(config, "experiment", "bad_channels", default=_cfg_value(config, "bad_channels", default={}))
    out = raw.copy()
    meta: dict[str, Any] = {
        "freq_filter": bool(prep.get("freq_filter", True)),
        "notch_filter": bool(prep.get("notch_filter")),
        "l_freq": filt.get("l_freq", 0.5),
        "h_freq": filt.get("h_freq", 40),
        "notch_freq": filt.get("notch_freq"),
        "bad_channels_detected": [],
    }
    if prep.get("bad_channels", False):
        bads = detect_bad_channels(
            out,
            flat_std=bc.get("flat_std", 1e-15),
            noisy_z=bc.get("noisy_z", 5.0),
        )
        out.info["bads"] = list(set(out.info.get("bads", []) + bads))
        meta["bad_channels_detected"] = bads

    good_picks = mne.pick_types(out.info, eeg=True, exclude="bads")
    if prep.get("freq_filter", True):
        out.filter(
            l_freq=filt.get("l_freq", 0.5),
            h_freq=filt.get("h_freq", 40),
            fir_design="firwin",
            filter_length="auto",
            picks=good_picks,
        )
    if prep.get("notch_filter") and filt.get("notch_freq"):
        out.notch_filter(freqs=filt["notch_freq"], picks=good_picks)
    return out, meta


def _resolve_ica_n_components(raw: mne.io.BaseRaw, ica_n: Any) -> tuple[int | float, Any]:
    """Resolve ICA n_components; 'auto' or None → n_EEG - 1."""
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    n_eeg = len(eeg_picks)
    if ica_n is None or ica_n == "auto":
        return max(1, n_eeg - 1), "auto"
    if isinstance(ica_n, float):
        return ica_n, ica_n
    return int(ica_n), ica_n


def stage_ica(raw: mne.io.BaseRaw, config: dict) -> tuple[mne.io.BaseRaw, dict]:
    prep = _cfg_value(config, "experiment", "preprocessing", default={})
    meta: dict[str, Any] = {
        "ica_converged": None,
        "ica_n_components": None,
        "ica_n_components_requested": None,
        "ica_n_components_fitted": None,
        "ica_n_removed": 0,
        "ica_excluded_indices": [],
        "ica_eog_indices": [],
        "ica_ecg_indices": [],
        "ica_n_iter": None,
        "ica_pca_var_ratio_top5": [],
        "ica_eeg_rank": None,
    }
    if not prep.get("run_ica", False):
        return raw.copy(), meta

    out = raw.copy()
    sfreq = out.info["sfreq"]
    seconds = out.n_times / sfreq
    if seconds < 30:
        meta["warnings"] = ["Recording too short for ICA, skipping."]
        return out, meta

    bads_excluded = list(out.info.get("bads", []))
    meta["ica_bad_channels_excluded"] = bads_excluded

    ica_n_cfg = prep.get("ica_n_components", "auto")
    ica_n, ica_n_requested = _resolve_ica_n_components(out, ica_n_cfg)
    ica_seed = prep.get("ica_random_state", 97)
    ica_picks = mne.pick_types(out.info, eeg=True, exclude="bads")
    raw_ica = out.copy().filter(l_freq=1.0, h_freq=None, picks=ica_picks, verbose=False)
    ica = ICA(n_components=ica_n, method="infomax", random_state=ica_seed)
    ica.fit(raw_ica)

    n_fitted = int(ica.n_components_)
    eeg_picks = mne.pick_types(out.info, eeg=True, exclude=[])
    n_eeg = len(eeg_picks)
    if n_fitted < n_eeg * 0.5:
        meta.setdefault("warnings", []).append(
            f"ICA fitted only {n_fitted} components (< 50% of {n_eeg} EEG channels)."
        )

    pca_var = getattr(ica, "pca_explained_variance_", None)
    if pca_var is not None and len(pca_var) > 0:
        cumvar = np.cumsum(pca_var) / np.sum(pca_var)
        meta["ica_pca_var_ratio_top5"] = [round(float(v), 4) for v in cumvar[:5]]

    try:
        rank = mne.compute_rank(raw_ica, meg=False, eeg=True)
        meta["ica_eeg_rank"] = int(rank.get("eeg", rank) if isinstance(rank, dict) else rank)
    except Exception:
        meta["ica_eeg_rank"] = None

    meta.update(
        {
            "ica_n_components_requested": ica_n_requested,
            "ica_n_components_fitted": n_fitted,
            "ica_n_components": n_fitted,
            "ica_converged": True,
            "ica_n_iter": int(getattr(ica, "n_iter_", 0) or 0),
        }
    )

    eog_inds, _ = ica.find_bads_eog(out, ch_name=["Fp1", "Fp2"], threshold=3.0)
    meta["ica_eog_indices"] = list(eog_inds)
    ica.exclude = list(eog_inds)
    if "ECG" in out.ch_names:
        ecg_inds, _ = ica.find_bads_ecg(out, threshold=3.0, filterlength="auto")
        meta["ica_ecg_indices"] = list(ecg_inds)
        ica.exclude = list(set(ica.exclude + list(ecg_inds)))
    meta["ica_excluded_indices"] = list(ica.exclude)
    meta["ica_n_removed"] = len(ica.exclude)
    ica.apply(out)
    return out, meta


def _asr_window_size_s(asr_model: ASR) -> float | None:
    for attr in ("windowlen", "window_len", "win_len"):
        val = getattr(asr_model, attr, None)
        if val is not None:
            return float(val)
    return None


def stage_clean(
    raw: mne.io.BaseRaw,
    config: dict,
    raw_before: mne.io.BaseRaw | None = None,
) -> tuple[mne.io.BaseRaw, dict]:
    prep = _cfg_value(config, "experiment", "preprocessing", default={})
    out = raw.copy()
    meta: dict[str, Any] = {}

    bads = list(out.info.get("bads", []))
    meta["bad_channels"] = bads
    if prep.get("bad_channels", False) and bads:
        out.interpolate_bads(reset_bads=True)
        meta["bad_channels_interpolated"] = bads

    asr_enabled = bool(prep.get("asr", False))
    referencing = bool(prep.get("referencing", False))
    asr_meta: dict[str, Any] = {"asr_enabled": asr_enabled}
    if asr_enabled:
        sfreq = int(_cfg_value(config, "features", "sampling_rate", default=500))
        cutoff = prep.get("asr_cutoff", 17)
        calibration_duration_s = round(out.n_times / out.info["sfreq"], 2)
        asr_model = ASR(sfreq=sfreq, cutoff=cutoff)
        asr_model.fit(out)
        out = asr_model.transform(out)
        asr_meta.update(
            {
                "asr_cutoff": cutoff,
                "asr_sfreq": sfreq,
                "asr_calibration_duration_s": calibration_duration_s,
                "asr_window_size_s": _asr_window_size_s(asr_model),
                "asr_win_len": getattr(asr_model, "win_len", None),
                "asr_win_overlap": getattr(asr_model, "win_overlap", None),
            }
        )
        corrected = getattr(asr_model, "n_corrected_samples", None)
        if corrected is not None:
            asr_meta["asr_corrected_samples"] = int(corrected)
    meta["asr"] = asr_meta
    meta["asr_before_car"] = asr_enabled and referencing

    if referencing:
        out.set_eeg_reference("average")

    if raw_before is not None:
        band = compute_band_power_table(raw_before, out)
        psd = compute_psd_delta(raw_before, out)
        meta.update(band)
        meta.update(psd)

    return out, meta


def stage_epochs(raw: mne.io.BaseRaw, config: dict) -> tuple[mne.Epochs, dict]:
    prep = _cfg_value(config, "experiment", "preprocessing", default={})
    epoch_cfg = _cfg_value(config, "experiment", "epoching", default=_cfg_value(config, "epoching", default={}))
    meta: dict[str, Any] = {}

    if prep.get("erp", False):
        raise NotImplementedError("ERP epoching not supported in staged pipeline.")

    epochs = convert_to_epochs(
        raw,
        epoch_length=epoch_cfg.get("length", 4.0),
        epoch_overlap=epoch_cfg.get("overlap", 2.0),
    )
    n_before = len(epochs)
    meta["n_epochs_before_ar"] = n_before
    meta["reject_log"] = None

    if prep.get("AR", True):
        ar = autoreject.AutoReject(
            n_interpolate=[1, 2, 3, 4], random_state=11, n_jobs=1, verbose=False
        )
        ar.fit(epochs[: min(20, len(epochs))])
        epochs, reject_log = ar.transform(epochs, return_log=True)
        n_rejected = int(reject_log.bad_epochs.sum())
        meta["n_epochs_rejected"] = n_rejected
        meta["n_epochs_after_ar"] = len(epochs)
        meta["reject_log"] = reject_log
    else:
        meta["n_epochs_after_ar"] = n_before
        meta["n_epochs_rejected"] = 0

    meta["pct_epochs_rejected"] = round(
        100 * meta["n_epochs_rejected"] / max(n_before, 1), 2
    )
    meta["snr_db"] = round(compute_snr(epochs), 2)

    return epochs, meta


# Legacy single-call API (used by util/qc and tests)
def preprocess_EEG(
    eeg_raw,
    freq_filter=True,
    notch_filter=False,
    bad_channels=False,
    referencing=False,
    asr=False,
    asr_cutoff=None,
    run_ica=False,
    AR=True,
    erp=False,
    fle=True,
    verbose_plots=False,
    return_reject_log=False,
):
    base = load_base_configs()
    sfreq = base["features"]["sampling_rate"]
    config = {
        "experiment": {
            "preprocessing": {
                "freq_filter": freq_filter,
                "notch_filter": notch_filter,
                "bad_channels": bad_channels,
                "referencing": referencing,
                "asr": asr,
                "asr_cutoff": asr_cutoff or 17,
                "run_ica": run_ica,
                "AR": AR,
                "erp": erp,
                "fle": fle,
            },
            "filtering": {"l_freq": 0.5, "h_freq": 40, "notch_freq": [50]},
            "epoching": {"length": 4.0, "overlap": 2.0},
        },
        "features": {"sampling_rate": sfreq},
    }

    out, _ = apply_standard_montage(eeg_raw.copy())
    filtered, _ = stage_filtered(out, config)
    ica_out, _ = stage_ica(filtered, config) if run_ica else (filtered, {})
    clean, clean_meta = stage_clean(ica_out, config)
    epochs, epoch_meta = stage_epochs(clean, config)

    meta = {**clean_meta, **epoch_meta}
    reject_log = epoch_meta.get("reject_log")
    if return_reject_log:
        return clean, epochs, reject_log, meta
    return clean, epochs


STAGE_FUNCS = {
    "raw": None,
    "filtered": stage_filtered,
    "ica": stage_ica,
    "clean": stage_clean,
    "epochs": stage_epochs,
}


def _flatten_subject_log(log: dict[str, Any], dataset_name: str, experiment: str) -> None:
    """Promote key stage metrics to top-level log for CSV export."""
    log["dataset_name"] = dataset_name
    log["experiment"] = experiment
    stages = log.get("stages", {})
    raw_s = stages.get("raw", {})
    filtered_s = stages.get("filtered", {})
    ica_s = stages.get("ica", {})
    clean_s = stages.get("clean", {})
    epoch_s = stages.get("epochs", {})

    for key in ("duration_s", "sfreq", "n_channels", "n_eeg_channels"):
        if key in raw_s:
            log[key] = raw_s[key]

    for key in (
        "ica_n_components_fitted",
        "ica_n_components",
        "ica_n_removed",
        "ica_eeg_rank",
        "ica_n_iter",
        "ica_excluded_indices",
    ):
        if key in ica_s:
            log[key] = ica_s[key]

    if "bad_channels" in clean_s:
        log["bad_channels"] = clean_s["bad_channels"]
    elif "bad_channels_detected" in filtered_s:
        log["bad_channels"] = filtered_s["bad_channels_detected"]

    for key in (
        "n_epochs_before_ar",
        "n_epochs_after_ar",
        "n_epochs_rejected",
        "pct_epochs_rejected",
        "snr_db",
    ):
        if key in epoch_s:
            log[key] = epoch_s[key]

    for band in ("delta", "theta", "alpha", "beta", "gamma"):
        for suffix in ("_before_uv2", "_after_uv2", "_delta_uv2", "_before_log10", "_after_log10"):
            k = f"{band}{suffix}"
            if k in clean_s:
                log[k] = clean_s[k]
    for key in ("psd_mean_before_uv2", "psd_mean_after_uv2", "psd_ratio_after_before"):
        if key in clean_s:
            log[key] = clean_s[key]


def _log_is_valid(log: dict, raw_sha256: str, fingerprint: str) -> bool:
    return (
        log.get("raw_sha256") == raw_sha256
        and log.get("config_fingerprint") == fingerprint
        and log.get("status") == "ok"
    )


def _furthest_valid_stage(
    dataset: str,
    experiment: str,
    participant_id: str,
    raw_sha256: str,
    fingerprint: str,
) -> str | None:
    from eeg.io import read_json

    log_path = subject_log_path(dataset, experiment, participant_id)
    if not log_path.exists():
        return None
    log = read_json(log_path)
    if not _log_is_valid(log, raw_sha256, fingerprint):
        return None

    completed = log.get("stages_completed", [])
    order = ["raw", "filtered", "ica", "clean", "epochs"]

    # Fast path: final artifact valid → fully done
    if "epochs" in completed:
        cp = checkpoint_path(dataset, experiment, participant_id, "epochs")
        if try_load_checkpoint(cp, "epochs") is not None:
            return "epochs"

    valid = None
    for stage in order:
        if stage not in completed:
            break
        cp = checkpoint_path(dataset, experiment, participant_id, stage)
        if try_load_checkpoint(cp, stage) is not None:
            valid = stage
        else:
            break
    return valid


def preprocess_subject(
    raw_path: Path,
    participant_id: str,
    dataset_name: str,
    experiment: str,
    config: dict,
    config_fp: str,
    force: bool = False,
) -> PreprocessResult:
    """Run or resume full checkpoint chain for one subject."""
    t0 = time.perf_counter()
    base = load_base_configs()
    sfreq = base["features"]["sampling_rate"]
    raw_sha256 = sha256_file(raw_path)

    log: dict[str, Any] = {
        "participant_id": participant_id,
        "raw_sha256": raw_sha256,
        "config_fingerprint": config_fp,
        "stages": {},
        "warnings": [],
        "stages_completed": [],
    }

    if not force:
        furthest = _furthest_valid_stage(
            dataset_name, experiment, participant_id, raw_sha256, config_fp
        )
        if furthest == "epochs":
            epochs_path = resolve_checkpoint_path(
                dataset_name, experiment, participant_id, "epochs"
            )
            epochs = load_checkpoint(epochs_path, "epochs")
            from eeg.io import read_json

            log_path = subject_log_path(dataset_name, experiment, participant_id)
            if log_path.exists():
                log = read_json(log_path)
            log["status"] = "skipped"
            log["runtime_seconds"] = round(time.perf_counter() - t0, 2)
            _flatten_subject_log(log, dataset_name, experiment)
            write_json(log_path, log)
            return PreprocessResult(participant_id, "skipped", log, epochs)

    stage_order = ["raw", "filtered", "ica", "clean", "epochs"]
    start_idx = 0
    if not force:
        furthest = _furthest_valid_stage(
            dataset_name, experiment, participant_id, raw_sha256, config_fp
        )
        if furthest:
            start_idx = stage_order.index(furthest)

    current_raw: mne.io.BaseRaw | None = None
    current_epochs: mne.Epochs | None = None

    for i, stage in enumerate(stage_order):
        if i < start_idx:
            continue

        cp = checkpoint_path(dataset_name, experiment, participant_id, stage)
        stage_t0 = time.perf_counter()

        try:
            if stage == "raw":
                current_raw, meta = stage_raw(raw_path, sfreq)
                save_checkpoint(current_raw, cp)
            elif stage == "filtered":
                assert current_raw is not None or start_idx > 0
                if current_raw is None:
                    prev = resolve_checkpoint_path(
                        dataset_name, experiment, participant_id, "raw"
                    )
                    current_raw = load_checkpoint(prev, "raw")
                current_raw, meta = stage_filtered(current_raw, config)
                save_checkpoint(current_raw, cp)
            elif stage == "ica":
                if current_raw is None:
                    prev = resolve_checkpoint_path(
                        dataset_name, experiment, participant_id, "filtered"
                    )
                    current_raw = load_checkpoint(prev, "filtered")
                current_raw, meta = stage_ica(current_raw, config)
                save_checkpoint(current_raw, cp)
            elif stage == "clean":
                if current_raw is None:
                    prev = resolve_checkpoint_path(
                        dataset_name, experiment, participant_id, "ica"
                    )
                    current_raw = load_checkpoint(prev, "ica")
                raw_before = None
                raw_cp = resolve_checkpoint_path(
                    dataset_name, experiment, participant_id, "raw"
                )
                if raw_cp.exists():
                    raw_before = load_checkpoint(raw_cp, "raw")
                current_raw, meta = stage_clean(current_raw, config, raw_before=raw_before)
                save_checkpoint(current_raw, cp)
            elif stage == "epochs":
                if current_raw is None:
                    prev = resolve_checkpoint_path(
                        dataset_name, experiment, participant_id, "clean"
                    )
                    current_raw = load_checkpoint(prev, "clean")
                current_epochs, meta = stage_epochs(current_raw, config)
                save_checkpoint(current_epochs, cp)

            elapsed = round(time.perf_counter() - stage_t0, 2)
            stage_meta = {k: v for k, v in meta.items() if k != "reject_log"}
            log["stages"][stage] = {"runtime_s": elapsed, "status": "ok", **stage_meta}
            log["stages_completed"].append(stage)
            if meta.get("warnings"):
                log["warnings"].extend(meta["warnings"])

        except Exception as exc:
            log["stages"][stage] = {
                "runtime_s": round(time.perf_counter() - stage_t0, 2),
                "status": "error",
                "error": str(exc),
            }
            log["status"] = "error"
            log["runtime_seconds"] = round(time.perf_counter() - t0, 2)
            write_json(subject_log_path(dataset_name, experiment, participant_id), log)
            return PreprocessResult(participant_id, "error", log)

    log["status"] = "ok"
    log["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    _flatten_subject_log(log, dataset_name, experiment)
    write_json(subject_log_path(dataset_name, experiment, participant_id), log)
    return PreprocessResult(participant_id, "ok", log, current_epochs)
