"""Plotting helpers for QC and training results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from eeg.io import write_json
from eeg.paths import qc_report_dir
from eeg.qc import load_checkpoints_for_qc, preprocessing_metrics


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_preprocessing_panels_from_checkpoints(
    dataset_spec,
    subject_num: int,
    experiment: str = "baseline",
    output_dir: Path | None = None,
) -> Path:
    """Save multi-panel QC figure from existing checkpoints (no reprocessing)."""
    base_sfreq = dataset_spec  # placeholder for type clarity
    del base_sfreq

    participant_id = f"sub-{subject_num:03d}"
    out_dir = _ensure_dir(
        output_dir or qc_report_dir(dataset_spec.name, experiment)
    )
    metrics = preprocessing_metrics(dataset_spec, subject_num, experiment)
    checkpoints = load_checkpoints_for_qc(dataset_spec.name, experiment, participant_id)

    if "raw" not in checkpoints:
        raise FileNotFoundError(
            f"Missing raw checkpoint for {participant_id}. Run preprocessing first."
        )

    raw = checkpoints["raw"]
    filtered = checkpoints.get("filtered", raw)
    clean = checkpoints.get("clean", filtered)
    epochs = checkpoints.get("epochs")
    sfreq = raw.info["sfreq"]
    target_channels = ["Fp1", "Fp2", "F7", "Cz"]

    fig = plt.figure(figsize=(14, 10))
    label = metrics.get("label", "")
    fig.suptitle(
        f"Preprocessing QC — {participant_id} ({label}), {dataset_spec.name}",
        fontsize=13,
        y=0.98,
    )
    grid = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    ax_psd = fig.add_subplot(grid[0, 0])
    picks = [ch for ch in target_channels if ch in raw.ch_names]
    for name, data, style in [
        ("raw", raw, "-"),
        ("clean", clean, "--"),
    ]:
        psd = data.compute_psd(method="welch", fmin=1, fmax=40, picks=picks, verbose=False)
        freqs = psd.freqs
        for i, ch in enumerate(picks):
            ax_psd.semilogy(freqs, psd.get_data()[i], style, alpha=0.8, label=f"{ch} ({name})")
    ax_psd.set_xlim(1, 40)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD")
    ax_psd.set_title("Raw vs clean")
    ax_psd.legend(fontsize=7, ncol=2)
    ax_psd.grid(True, alpha=0.3)

    ax_ar = fig.add_subplot(grid[0, 1])
    n_rej = metrics.get("n_epochs_rejected", 0)
    n_before = metrics.get("n_epochs_before_ar", 0)
    ax_ar.text(
        0.5,
        0.5,
        f"AutoReject\n{n_rej}/{n_before} rejected\n({metrics.get('pct_epochs_rejected', 0)}%)",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax_ar.set_axis_off()
    ax_ar.set_title("Epoch rejection")

    ax_epoch = fig.add_subplot(grid[1, 0])
    if epochs is not None and len(epochs) > 0:
        epoch_data = epochs.get_data()[0]
        times = np.arange(epoch_data.shape[1]) / sfreq
        ep_picks = [ch for ch in picks if ch in epochs.ch_names]
        for i, ch in enumerate(ep_picks):
            ch_idx = epochs.ch_names.index(ch)
            offset = i * 50e-6
            ax_epoch.plot(times, epoch_data[ch_idx] + offset, label=ch)
        ax_epoch.set_xlabel("Time (s)")
        ax_epoch.set_ylabel("Amplitude (+ offset)")
        ax_epoch.set_title("First retained epoch")
        ax_epoch.legend(fontsize=8)
        ax_epoch.grid(True, alpha=0.3)
    else:
        ax_epoch.text(0.5, 0.5, "No epochs checkpoint", ha="center", va="center")
        ax_epoch.set_axis_off()

    ax_info = fig.add_subplot(grid[1, 1])
    ax_info.axis("off")
    lines = [
        f"Bad channels: {metrics.get('bad_channels', '')}",
        f"SNR: {metrics.get('snr_db', 'n/a')} dB",
        f"ICA fitted: {metrics.get('ica_n_components_fitted', 'n/a')}",
        f"ICA removed: {metrics.get('ica_n_removed', 'n/a')}",
        f"Rank: {metrics.get('ica_eeg_rank', 'n/a')}",
        f"Epochs: {metrics.get('n_epochs_after_ar', 'n/a')} / {metrics.get('n_epochs_before_ar', 'n/a')}",
        f"Rejected: {metrics.get('pct_epochs_rejected', 'n/a')}%",
        f"Runtime: {metrics.get('runtime_seconds', 'n/a')} s",
    ]
    band_line = " | ".join(
        f"{b[0].upper()}: {metrics.get(f'{b}_delta', 'n/a')}"
        for b in ("delta", "theta", "alpha", "beta", "gamma")
        if metrics.get(f"{b}_delta") is not None
    )
    if band_line:
        lines.append(f"Band Δ: {band_line}")
    ax_info.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=10, family="monospace")

    out_path = out_dir / f"{participant_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    write_json(out_path.with_suffix(".json"), metrics)
    return out_path


def plot_preprocessing_panels(
    dataset_spec,
    subject_num: int,
    experiment: str = "baseline",
    output_dir: Path | None = None,
) -> Path:
    """Save QC figure from checkpoints (alias for checkpoint-based plotting)."""
    return plot_preprocessing_panels_from_checkpoints(
        dataset_spec, subject_num, experiment, output_dir
    )
