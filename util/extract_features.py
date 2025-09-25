import mne

import numpy as np
from biomarkers import lempel_ziv_complexity, multiscale_entropy, compute_band_power

def extract_eeg_features(data):
    for epoch_id, epoch in enumerate(data):

        epoch_mne = mne.EpochsArray(epoch[np.newaxis, :, :], info=epochs_info)

        epoch_mean_ts = epoch.mean(axis=0)

        binary_seq = (epoch_mean_ts > np.median(epoch_mean_ts)).astype(int).tolist()
        lzc = lempel_ziv_complexity(binary_seq)

        mse_vector = multiscale_entropy(epoch_mean_ts)
        mse_mean = np.nanmean(mse_vector)
            
        band_power_features = compute_band_power(epoch_mne, target_channels=None)
        
        connectivity_features = {
            "lzc": lzc,
            "mse_mean": mse_mean
        }
            
        merged_features = {**band_power_features, **connectivity_features}
        
    return merged_features