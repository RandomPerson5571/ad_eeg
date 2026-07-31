"""Quality-control helpers for manual preprocessing and feature inspection."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from config import (
    EPOCH_LENGTH,
    EPOCH_OVERLAP,
    FEATURE_COLUMNS,
    PARQUET_COMBINED_FILE,
    PREPROCESS_DEFAULTS,
    RESULTS_DIR,
    SAMPLING_RATE,
    TARGET_CHANNELS,
)
from util.extract_features import connectivity_for_epochs, extract_eeg_features
from util.io import list_subjects, load_features_df, raw_eeg_path, read_eeg_data
from util.preprocessing import convert_to_epochs, preprocess_EEG

QC_DIR = Path(RESULTS_DIR) / "qc"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _subject_label(dataset_id: int, subject_num: int) -> str:
    participants = list_subjects(dataset_id)
    row = participants.iloc[subject_num - 1]
    return row["participant_id"], row["Group"]


def preprocessing_summary(dataset_id: int, subject_num: int) -> dict:
    """Run preprocessing and return a JSON-serializable summary."""
    participant_id, label = _subject_label(dataset_id, subject_num)
    path = raw_eeg_path(dataset_id, subject_num)
    raw = read_eeg_data(path, sfreq=SAMPLING_RATE)

    raw_copy = raw.copy()
    if raw_copy.info["sfreq"] != SAMPLING_RATE:
        raw_copy.resample(SAMPLING_RATE)

    n_epochs_before = len(convert_to_epochs(raw_copy))
    clean_eeg, epochs, reject_log, meta = preprocess_EEG(
        raw, return_reject_log=True, **PREPROCESS_DEFAULTS
    )

    n_rejected = int(reject_log.bad_epochs.sum()) if reject_log is not None else 0
    return {
        "dataset_id": dataset_id,
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
        "epoch_shape": list(epochs.get_data().shape[1:]),
        "epoch_length_s": EPOCH_LENGTH,
        "epoch_overlap_s": EPOCH_OVERLAP,
    }


def plot_preprocessing_report(
    dataset_id: int,
    subject_num: int,
    output_dir: Path | None = None,
    show: bool = False,
) -> Path:
    """Save a multi-panel preprocessing QC figure and summary JSON."""
    out_dir = _ensure_dir(output_dir or QC_DIR / "preprocessing")
    participant_id, label = _subject_label(dataset_id, subject_num)
    stem = f"dataset{dataset_id}_sub{subject_num:03d}"

    path = raw_eeg_path(dataset_id, subject_num)
    raw = read_eeg_data(path, sfreq=SAMPLING_RATE)
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=0.5, h_freq=40, fir_design="firwin", filter_length="auto")

    clean_eeg, epochs, reject_log, meta = preprocess_EEG(
        raw, return_reject_log=True, **PREPROCESS_DEFAULTS
    )

    raw_copy = raw.copy()
    if raw_copy.info["sfreq"] != SAMPLING_RATE:
        raw_copy.resample(SAMPLING_RATE)
    n_epochs_before = len(convert_to_epochs(raw_copy))
    n_rejected = int(reject_log.bad_epochs.sum()) if reject_log is not None else 0
    summary = {
        "dataset_id": dataset_id,
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
        "epoch_shape": list(epochs.get_data().shape[1:]),
        "epoch_length_s": EPOCH_LENGTH,
        "epoch_overlap_s": EPOCH_OVERLAP,
    }

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Preprocessing QC — {participant_id} ({label}), dataset {dataset_id}",
        fontsize=13,
        y=0.98,
    )
    grid = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # Raw vs filtered PSD
    ax_psd = fig.add_subplot(grid[0, 0])
    picks = [ch for ch in TARGET_CHANNELS if ch in raw.ch_names]
    for name, data, style in [
        ("raw", raw, "-"),
        ("filtered", raw_filtered, "--"),
    ]:
        psd = data.compute_psd(method="welch", fmin=1, fmax=40, picks=picks)
        freqs = psd.freqs
        for i, ch in enumerate(picks):
            ax_psd.semilogy(freqs, psd.get_data()[i], style, alpha=0.8, label=f"{ch} ({name})")
    ax_psd.set_xlim(1, 40)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD")
    ax_psd.set_title("Raw vs band-pass (0.5–40 Hz)")
    ax_psd.legend(fontsize=7, ncol=2)
    ax_psd.grid(True, alpha=0.3)

    # AutoReject reject log
    ax_ar = fig.add_subplot(grid[0, 1])
    if reject_log is not None:
        reject_log.plot("horizontal", ax=ax_ar, show=False)
        ax_ar.set_title(
            f"AutoReject — {summary['n_epochs_rejected']}/{summary['n_epochs_before_ar']} rejected"
        )
    else:
        ax_ar.text(0.5, 0.5, "AutoReject disabled", ha="center", va="center")
        ax_ar.set_axis_off()

    # Sample epoch waveforms
    ax_epoch = fig.add_subplot(grid[1, 0])
    epoch_data = epochs.get_data()[0]
    times = np.arange(epoch_data.shape[1]) / SAMPLING_RATE
    for i, ch in enumerate(picks):
        ch_idx = epochs.ch_names.index(ch)
        offset = i * 50e-6
        ax_epoch.plot(times, epoch_data[ch_idx] + offset, label=ch)
    ax_epoch.set_xlabel("Time (s)")
    ax_epoch.set_ylabel("Amplitude (+ offset)")
    ax_epoch.set_title("First retained epoch")
    ax_epoch.legend(fontsize=8)
    ax_epoch.grid(True, alpha=0.3)

    # Summary text
    ax_info = fig.add_subplot(grid[1, 1])
    ax_info.axis("off")
    lines = [
        f"Participant: {participant_id}  |  Label: {label}",
        f"Duration: {summary['duration_seconds']} s  |  Channels: {summary['n_channels']}",
        f"Epochs before AR: {summary['n_epochs_before_ar']}",
        f"Epochs after AR: {summary['n_epochs_after_ar']}",
        f"Rejected: {summary['n_epochs_rejected']} ({summary['pct_epochs_rejected']}%)",
        f"Epoch shape (ch × samples): {summary['epoch_shape']}",
    ]
    ax_info.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=10)

    png_path = out_dir / f"{stem}_preprocessing.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    json_path = out_dir / f"{stem}_preprocessing.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return png_path


def _reextract_subject_features(dataset_id: int, subject_num: int) -> pd.DataFrame:
    """Re-run feature extraction for one subject (spot-check against parquet)."""
    path = raw_eeg_path(dataset_id, subject_num)
    raw = read_eeg_data(path, sfreq=SAMPLING_RATE)
    _, epochs = preprocess_EEG(raw, **PREPROCESS_DEFAULTS)
    ch_names = raw.ch_names
    data = epochs.get_data()
    connectivity = connectivity_for_epochs(epochs, ch_names)
    rows = extract_eeg_features(data, ch_names=ch_names, subject_connectivity=connectivity)
    return pd.DataFrame(rows)


def feature_summary(df: pd.DataFrame | None = None) -> dict:
    """Aggregate stats for the feature parquet store."""
    if df is None:
        if not Path(PARQUET_COMBINED_FILE).exists():
            raise FileNotFoundError(
                f"{PARQUET_COMBINED_FILE} not found. Run ingest_features.py first."
            )
        df = load_features_df()

    missing = df[FEATURE_COLUMNS].isna().sum().to_dict()
    by_label = (
        df.groupby("label")[FEATURE_COLUMNS]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .to_dict()
    )
    epochs_per_subject = df.groupby(["participant_id", "label"]).size()

    return {
        "n_rows": len(df),
        "n_subjects": int(df["participant_id"].nunique()),
        "n_datasets": int(df["dataset_id"].nunique()),
        "epochs_per_subject": {
            f"{pid}|{label}": int(n)
            for (pid, label), n in epochs_per_subject.items()
        },
        "label_counts": df["label"].value_counts().to_dict(),
        "missing_values": missing,
        "features_by_label": by_label,
    }


def plot_feature_report(
    df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    show: bool = False,
) -> Path:
    """Save feature distribution, per-subject, and correlation QC figures."""
    if df is None:
        if not Path(PARQUET_COMBINED_FILE).exists():
            raise FileNotFoundError(
                f"{PARQUET_COMBINED_FILE} not found. Run ingest_features.py first."
            )
        df = load_features_df()

    out_dir = _ensure_dir(output_dir or QC_DIR / "features")
    summary = feature_summary(df)

    # Per-feature histograms by diagnostic group
    n_feats = len(FEATURE_COLUMNS)
    n_cols = 4
    n_rows = int(np.ceil(n_feats / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    labels = sorted(df["label"].dropna().unique())
    colors = {"A": "#d62728", "F": "#ff7f0e", "C": "#2ca02c"}

    for ax, col in zip(axes, FEATURE_COLUMNS):
        for lab in labels:
            subset = df.loc[df["label"] == lab, col].dropna()
            ax.hist(subset, bins=20, alpha=0.5, label=lab, color=colors.get(lab, None))
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)

    for ax in axes[n_feats:]:
        ax.set_axis_off()

    fig.suptitle("Feature distributions by diagnostic group", y=1.01)
    fig.tight_layout()
    dist_path = out_dir / "feature_distributions.png"
    fig.savefig(dist_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # Epochs per subject
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    counts = df.groupby(["participant_id", "label"]).size().reset_index(name="n_epochs")
    counts = counts.sort_values("n_epochs")
    bar_colors = [colors.get(l, "gray") for l in counts["label"]]
    ax2.barh(counts["participant_id"], counts["n_epochs"], color=bar_colors)
    ax2.set_xlabel("Epochs per subject")
    ax2.set_title("Epoch counts after ingest")
    fig2.tight_layout()
    epochs_path = out_dir / "epochs_per_subject.png"
    fig2.savefig(epochs_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig2)

    # Correlation heatmap
    corr = df[FEATURE_COLUMNS].corr()
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    im = ax3.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax3.set_xticks(range(len(FEATURE_COLUMNS)))
    ax3.set_yticks(range(len(FEATURE_COLUMNS)))
    ax3.set_xticklabels(FEATURE_COLUMNS, rotation=90, fontsize=8)
    ax3.set_yticklabels(FEATURE_COLUMNS, fontsize=8)
    fig3.colorbar(im, ax=ax3, label="r")
    ax3.set_title("Feature correlation matrix")
    fig3.tight_layout()
    corr_path = out_dir / "feature_correlation.png"
    fig3.savefig(corr_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig3)

    json_path = out_dir / "feature_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return dist_path


def spot_check_features(
    dataset_id: int,
    subject_num: int,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare parquet features for one subject against a fresh re-extraction."""
    participant_id, _ = _subject_label(dataset_id, subject_num)
    if df is None:
        df = load_features_df()

    stored = df[(df["dataset_id"] == dataset_id) & (df["participant_id"] == participant_id)]
    if stored.empty:
        raise ValueError(f"No parquet rows for {participant_id} in dataset {dataset_id}")

    fresh = _reextract_subject_features(dataset_id, subject_num)
    fresh["source"] = "reextracted"
    stored = stored[FEATURE_COLUMNS + ["epoch_id"]].copy()
    stored["source"] = "parquet"

    comparison = pd.concat([stored, fresh[FEATURE_COLUMNS + ["epoch_id", "source"]]], ignore_index=True)
    return comparison
