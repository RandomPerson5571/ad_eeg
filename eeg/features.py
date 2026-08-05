"""Feature extraction from preprocessed epochs."""

from __future__ import annotations

import mne
import numpy as np

from biomarkers import compute_band_power, compute_connectivity, compute_regional_complexity
from eeg.config import load_base_configs


def connectivity_for_epochs(epochs, ch_names):
    base = load_base_configs()
    sfreq = base["features"]["sampling_rate"]
    data = epochs.get_data()
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    epochs_mne = mne.EpochsArray(data, info=info)
    return compute_connectivity(epochs_mne)


def extract_eeg_features(data, ch_names=None, verbose=False, subject_connectivity=None):
    base = load_base_configs()
    sfreq = base["features"]["sampling_rate"]
    regional = base["features"].get("regional_channels", {"posterior": ["P3", "Pz", "P4", "O1", "O2"]})

    merged_features = []
    for epoch_id, epoch in enumerate(data):
        if verbose:
            print(f"Epoch: {epoch_id}")

        if ch_names is None:
            ch_names = [f"EEG{i}" for i in range(epoch.shape[0])]
        ch_types = ["eeg"] * len(ch_names)
        epochs_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        epoch_mne = mne.EpochsArray(epoch[np.newaxis, :, :], info=epochs_info)

        complexity_features = {}
        for region_name, region_channels in regional.items():
            complexity_features.update(
                compute_regional_complexity(epoch, ch_names, region_channels, region_name=region_name)
            )

        band_power_features = compute_band_power(epoch_mne, target_channels=None)
        connectivity_features = (
            subject_connectivity
            if subject_connectivity is not None
            else compute_connectivity(epoch_mne)
        )

        epoch_features = {
            "epoch_id": epoch_id,
            **band_power_features,
            **connectivity_features,
            **complexity_features,
        }
        merged_features.append(epoch_features)

    return merged_features


def extract_from_epochs(epochs: mne.Epochs) -> list[dict]:
    data = epochs.get_data()
    ch_names = epochs.ch_names
    subject_connectivity = connectivity_for_epochs(epochs, ch_names)
    return extract_eeg_features(data, ch_names=ch_names, subject_connectivity=subject_connectivity)
