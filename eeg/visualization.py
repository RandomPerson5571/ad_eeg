"""Plotting helpers for QC and training results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from eeg.config import load_base_configs, load_experiment
from eeg.io import read_eeg_data
from eeg.paths import raw_eeg_path, results_dir
from eeg.preprocessing import preprocess_EEG, stage_filtered
from eeg.qc import preprocessing_metrics


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_preprocessing_panels(
    dataset_spec,
    subject_num: int,
    experiment: str = "baseline",
    output_dir: Path | None = None,
) -> Path:
    """Save multi-panel preprocessing QC figure."""
    base = load_base_configs()
    sfreq = base["features"]["sampling_rate"]
    target_channels = ["Fp1", "Fp2", "F7", "Cz"]
    config = load_experiment(experiment)

    out_dir = _ensure_dir(
        output_dir or results_dir(dataset_spec.name, experiment) / "qc"
    )
    participant_id = f"sub-{subject_num:03d}"
    stem = participant_id

    path = raw_eeg_path(dataset_spec, subject_num)
    raw = read_eeg_data(path, sfreq=sfreq)
    filtered, _ = stage_filtered(raw, config)

    prep = config["experiment"]["preprocessing"]
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

    metrics = preprocessing_metrics(dataset_spec, subject_num, experiment)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Preprocessing QC — {participant_id} ({metrics['label']}), {dataset_spec.name}",
        fontsize=13,
        y=0.98,
    )
    grid = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    ax_psd = fig.add_subplot(grid[0, 0])
    picks = [ch for ch in target_channels if ch in raw.ch_names]
    for name, data, style in [("raw", raw, "-"), ("filtered", filtered, "--")]:
        psd = data.compute_psd(method="welch", fmin=1, fmax=40, picks=picks)
        freqs = psd.freqs
        for i, ch in enumerate(picks):
            ax_psd.semilogy(freqs, psd.get_data()[i], style, alpha=0.8, label=f"{ch} ({name})")
    ax_psd.set_xlim(1, 40)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD")
    ax_psd.set_title("Raw vs band-pass")
    ax_psd.legend(fontsize=7, ncol=2)
    ax_psd.grid(True, alpha=0.3)

    ax_ar = fig.add_subplot(grid[0, 1])
    if reject_log is not None:
        reject_log.plot("horizontal", ax=ax_ar, show=False)
        ax_ar.set_title(
            f"AutoReject — {metrics['n_epochs_rejected']}/{metrics['n_epochs_before_ar']} rejected"
        )
    else:
        ax_ar.text(0.5, 0.5, "AutoReject disabled", ha="center", va="center")
        ax_ar.set_axis_off()

    ax_epoch = fig.add_subplot(grid[1, 0])
    epoch_data = epochs.get_data()[0]
    times = np.arange(epoch_data.shape[1]) / sfreq
    for i, ch in enumerate(picks):
        ch_idx = epochs.ch_names.index(ch)
        offset = i * 50e-6
        ax_epoch.plot(times, epoch_data[ch_idx] + offset, label=ch)
    ax_epoch.set_xlabel("Time (s)")
    ax_epoch.set_ylabel("Amplitude (+ offset)")
    ax_epoch.set_title("First retained epoch")
    ax_epoch.legend(fontsize=8)
    ax_epoch.grid(True, alpha=0.3)

    ax_info = fig.add_subplot(grid[1, 1])
    ax_info.axis("off")
    lines = [
        f"Bad channels: {metrics['bad_channels_detected']}",
        f"SNR: {metrics['snr_db']:.2f} dB",
        f"Epochs: {metrics['n_epochs_after_ar']} / {metrics['n_epochs_before_ar']}",
        f"Rejected: {metrics['pct_epochs_rejected']}%",
        f"PSD ratio (after/before): {metrics.get('psd_ratio_after_before', 'n/a')}",
    ]
    ax_info.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=10, family="monospace")

    out_path = out_dir / f"{stem}_preprocessing_qc.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    from eeg.io import write_json

    write_json(out_path.with_suffix(".json"), metrics)
    return out_path
