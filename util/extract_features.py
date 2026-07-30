import mne

import numpy as np
from biomarkers import lempel_ziv_complexity, multiscale_entropy, compute_band_power, compute_connectivity
from config import SAMPLING_RATE


def connectivity_for_epochs(epochs, ch_names):
    """Compute subject-level connectivity once and reuse for all epochs."""
    data = epochs.get_data()
    info = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_RATE, ch_types=["eeg"] * len(ch_names))
    epochs_mne = mne.EpochsArray(data, info=info)
    return compute_connectivity(epochs_mne)

def extract_eeg_features(data, ch_names=None, verbose=False, subject_connectivity=None):

    merged_features = []

    for epoch_id, epoch in enumerate(data):
        if verbose:
            print(f"Epoch: {epoch_id}")

        if ch_names is None:
            ch_names = [f"EEG{i}" for i in range(epoch.shape[0])]
        ch_types = ["eeg"] * len(ch_names)

        epochs_info = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_RATE, ch_types=ch_types)

        epoch_mne = mne.EpochsArray(epoch[np.newaxis, :, :], info=epochs_info)

        epoch_mean_ts = epoch.mean(axis=0)

        lzc = lempel_ziv_complexity(epoch_mean_ts)

        mse_vector = multiscale_entropy(epoch_mean_ts)
        mse_mean = np.nanmean(mse_vector)
            
        band_power_features = compute_band_power(epoch_mne, target_channels=None)

        connectivity_features = subject_connectivity if subject_connectivity is not None else compute_connectivity(epoch_mne)
        
        complexity_features = {
            "lzc": lzc,
            "mse_mean": mse_mean
        }
            
        epoch_features = {
            "epoch_id" : epoch_id,
            **band_power_features,
            **connectivity_features,
            **complexity_features
        }

        merged_features.append(epoch_features)

    return merged_features