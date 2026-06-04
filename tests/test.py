from util import get_participant_data, read_eeg_data, extract_eeg_features, preprocess_EEG, plot_eeg, plot_power_spectrum
from config import SAMPLING_RATE, TARGET_CHANNELS, ALL_CHANNELS
import mne

import numpy as np
import math

from asrpy import ASR
from mne.preprocessing import ICA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import pandas as pd

# df = pd.read_parquet("parquet_files/all_features.parquet")

def band_SNR(clean_epochs, picks=None, fmin=1, fmax=40):
    from mne.time_frequency import psd_array_welch
    bands = {'delta': (1,4), 'theta': (4,8), 'alpha': (8,13), 'beta': (13,30)}
    eps = 1e-12

    if hasattr(clean_epochs, 'get_data'):
        clean_data = clean_epochs.get_data(picks=picks)
    else:
        clean_data = clean_epochs

    n_epochs, n_channels, n_times = clean_data.shape

    band_snr = {b: np.zeros((n_epochs, n_channels)) for b in bands}

    for e in range(n_epochs):
        for ch in range(n_channels):
            psd, freqs = psd_array_welch(clean_data[e,ch,:], sfreq=SAMPLING_RATE, n_fft=256, fmin=fmin, fmax=fmax)
            total_power = np.sum(psd)
            for b, (fmin, fmax) in bands.items():
                band_power = np.sum(psd[(freqs >= fmin) & (freqs < fmax)])
                band_snr[b][e, ch] = 10 * np.log10(band_power / (total_power - band_power + eps))

    return band_snr

def reconstruction_SNR(clean_epochs, raw_epochs, target_channels=None):
    eps = 1e-12

    if hasattr(clean_epochs, 'get_data'):
        clean_data = clean_epochs.get_data(picks=target_channels)
    else:
        clean_data = clean_epochs

    if hasattr(raw_epochs, 'get_data'):
        raw_data = raw_epochs.get_data(picks=target_channels)
    else:
        raw_data = raw_epochs

    n_epochs, n_channels, n_times = clean_data.shape

    reconstruction_SNR = np.zeros((n_epochs, n_channels))

    clean_data_demean = clean_data - clean_data.mean(axis=2, keepdims=True)
    raw_data_demean = raw_data - raw_data.mean(axis=2, keepdims=True)

    for e in range(n_epochs):
        for ch in range(n_channels):
            reconstruction_SNR[e, ch] = 10*math.log10(
                np.sum(clean_data_demean[e, ch,:] ** 2) /
                np.sum((raw_data_demean[e,ch,:] - clean_data_demean[e,ch,:]) ** 2 + eps)
            )

    return reconstruction_SNR

def balanced_accuracy(confusion_matrix, y_test, y_pred, C):
    cm = confusion_matrix(y_test, y_pred)
    sum = 0
    for i in range(C):
        TP = cm[i][i]
        FN = np.sum(cm[i]) - TP
        sum += TP/(TP+FN)

    sum /= C

    return sum

def plot_snr_comparison(snr_results_list, pipelines, band='alpha'):
    """
    Plots reconstruction and band SNR comparison for multiple pipelines.

    Parameters
    ----------
    snr_results_list : list of dict
        Output from compute_epoch_snr for each pipeline.
    pipelines : list of str
        Names of pipelines for labels.
    band : str
        Which band to plot SNR for.
    """
    n_pipelines = len(snr_results_list)

    plt.figure(figsize=(12,5))

    # Reconstruction SNR
    plt.subplot(1,2,1)
    data = [res['reconstruction_snr'].flatten() for res in snr_results_list]
    plt.violinplot(data)
    plt.xticks(np.arange(1,n_pipelines+1), pipelines)
    plt.title('Reconstruction SNR (dB)')
    plt.ylabel('SNR per channel & epoch')

    # Band-based SNR
    plt.subplot(1,2,2)
    data = [res['band_snr'][band].flatten() for res in snr_results_list]
    plt.violinplot(data)
    plt.xticks(np.arange(1,n_pipelines+1), pipelines)
    plt.title(f'{band.capitalize()} Band SNR (dB)')
    plt.ylabel('SNR per channel & epoch')

    plt.tight_layout()
    plt.show()

dataset = 2
i = 0

participant_data = get_participant_data()

participant_path = "EEG_data\\dataset{0}\\sub-{1:03d}\\eeg\\sub-{1:03d}_task-eyesclosed_eeg.set".format(dataset, i+1)

print(f"Processing participant {i+1}/88 (dataset {dataset})")

eeg_raw = read_eeg_data(participant_path, sfreq=SAMPLING_RATE)

clean_eeg_model1, epochs_model1 = preprocess_EEG(eeg_raw, freq_filter=True, notch_filter=False, asr=False, referencing=False, AR=True, fle=True)
# clean_eeg_model2, epochs_model2 = preprocess_EEG(eeg_raw, freq_filter=True, notch_filter=True, asr=True, asr_cutoff=10, referencing=True, AR=False, fle=True)
# clean_eeg_model3, epochs_model3 = preprocess_EEG(eeg_raw, freq_filter=True, notch_filter=True, run_ica=True, asr=True, asr_cutoff=10, referencing=True, AR=False, fle=True)

band_SNR_model1 = band_SNR(epochs_model1)
# band_SNR_model2 = band_SNR(epochs_model2)
# band_SNR_model3 = band_SNR(epochs_model3)

all_snr_results = band_SNR_model1
for band in band_SNR_model1.keys():
    band_SNR_model1[band][~np.isfinite(band_SNR_model1[band])] = np.nan

band_means = {band: np.nanmean(snr_array) for band, snr_array in band_SNR_model1}

for band, mean_snr in band_means.items():
    print(f"{band}: {mean_snr:.2f} dB")
# for snr_results in all_snr_results:
#     for band in snr_results.keys():
#         snr_results[band][~np.isfinite(snr_results[band])] = np.nan

#     band_means = {band: np.nanmean(snr_array) for band, snr_array in snr_results['band_snr'].items()}

#     print("Mean band SNR:")
#     for band, mean_snr in band_means.items():
#         print(f"{band}: {mean_snr:.2f} dB")

# derivatives_path = "EEG_data\\dataset{0}\\derivatives\\sub-{1:03d}\\eeg\\sub-{1:03d}_task-eyesclosed_eeg.set".format(dataset, i+1)
# derivative = read_eeg_data(derivatives_path, sfreq=SAMPLING_RATE)

# plot_eeg(derivative, target_channels=TARGET_CHANNELS)
# plot_power_spectrum(derivative, target_channels=TARGET_CHANNELS)

# path = "EEG_data\dataset2\sub-001\eeg\sub-001_task-eyesclosed_eeg.set"

# eeg_raw = read_eeg_data(path, SAMPLING_RATE)

# plot_eeg(eeg_raw, target_channels=TARGET_CHANNELS)

# derivative_path = "EEG_data\dataset2\derivatives\sub-001\eeg\sub-001_task-eyesclosed_eeg.set"
# eeg_derivative = read_eeg_data(derivative_path, SAMPLING_RATE)
# plot_power_spectrum(eeg_derivative, target_channels=TARGET_CHANNELS)
# plot_power_spectrum(eeg_derivative, target_channels=ALL_CHANNELS)

# plot_eeg(eeg_derivative, target_channels=TARGET_CHANNELS)

# clean_eeg = preprocess_EEG(eeg_raw)

# plot_eeg(clean_eeg, target_channels=TARGET_CHANNELS)
# plot_eeg(clean_eeg, target_channels=ALL_CHANNELS)
# plot_power_spectrum(clean_eeg, target_channels=TARGET_CHANNELS)
# plot_power_spectrum(clean_eeg, target_channels=ALL_CHANNELS)

# participant_data = get_participant_data()


# participant_id = participant_data[1].iloc[0, 0]

# print(participant_id)

# print(participant_data[0])

# group = participant_data[0].loc[participant_data[0]['participant_id'] == 'sub-001', 'Group'].values[0]
# print(group)

# print(participant_data[0])

# value = participant_data.loc[participant_data["participant_id"] == "sub-005", "Age"].values[0]
# print(value)



# i = 0

# participant_path = "EEG_data\\dataset2\\sub-{0:03d}\\eeg\\sub-{0:03d}_task-eyesclosed_eeg.set".format(i+1)

# eeg_raw = read_eeg_data(participant_path, sfreq=SAMPLING_RATE)

# epochs = prepr

# df = pd.DataFrame(epoch_features)

# print(df.head())