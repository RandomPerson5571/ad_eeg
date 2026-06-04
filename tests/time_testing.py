import mne
import numpy as np
import pandas as pd

import time
import matplotlib.pyplot as plt

from biomarkers import lempel_ziv_complexity, multiscale_entropy, compute_band_power, compute_connectivity, fast_lempel_ziv_complexity, LZ76

from util import read_eeg_data, preprocess_EEG, plot_power_spectrum, plot_eeg, extract_eeg_features, get_participant_data, save_as_parquet, convert_to_epochs
from config import SAMPLING_RATE, TARGET_CHANNELS, DATASETS

participant_data = get_participant_data()

for dataset in DATASETS:
    for i in range(1):

        ind = dataset-1

        participant_id = participant_data[ind].iloc[0, i]
        dataset_id = dataset
        label = participant_data[ind].loc[participant_data[ind]['participant_id'] == participant_id, 'Group'].values[0]

        participant_path = "EEG_data\\dataset{0}\\sub-{1:03d}\\eeg\\sub-{1:03d}_task-eyesclosed_eeg.set".format(dataset, i+1)

        print(f"Processing participant {i+1}/88 (dataset {dataset})")
        print(f"Reading data for participant {participant_id} from {participant_path}")

        eeg_raw = read_eeg_data(participant_path, sfreq=SAMPLING_RATE)

        preprocessing_time_start = time.perf_counter()

        clean_eeg, epochs = preprocess_EEG(eeg_raw, freq_filter=True, notch_filter=False, asr=False, referencing=False, AR=True, fle=True)

        preprocessing_time_end = time.perf_counter()

        print(f"Preprocessing took {preprocessing_time_end - preprocessing_time_start:.2f} seconds")
        data = epochs.get_data()
            
        print(f"Data shape (n_epochs, n_channels, n_times): {data.shape}")

        features = []

        derivatives_path = "EEG_data\\dataset{0}\\derivatives\\sub-{1:03d}\\eeg\\sub-{1:03d}_task-eyesclosed_eeg.set".format(dataset, i+1)
        derivative = read_eeg_data(derivatives_path, sfreq=SAMPLING_RATE)

        derivative_psd = derivative.compute_psd(
            method="welch",
            fmin=1,
            fmax=250,
            n_fft=2048,
            n_overlap=1024,
            picks=TARGET_CHANNELS
        )
        derivative_psd.plot()

        psd = clean_eeg.compute_psd(
            method="welch",
            fmin=1,
            fmax=250,
            n_fft=2048,
            n_overlap=1024,
            picks=TARGET_CHANNELS
        )
        psd.plot()

        timeout = input("Press enter to continue: ")

        for epoch_id, epoch in enumerate(data):
            processing_epoch_time_start = time.perf_counter()
            print(f"Epoch: {epoch_id}")

            ch_names = [f"EEG{i}" for i in range(epoch.shape[0])]
            ch_types = ["eeg"] * epoch.shape[0]

            epochs_info = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_RATE, ch_types=ch_types)

            epoch_mne = mne.EpochsArray(epoch[np.newaxis, :, :], info=epochs_info)

            epoch_mean_ts = epoch.mean(axis=0)

            # lzc_time_start = time.perf_counter()
            # lzc = lempel_ziv_complexity(epoch_mean_ts)
            # lzc_time_end = time.perf_counter()

            # fast_lzc_time_start = time.perf_counter()
            # lzc = fast_lempel_ziv_complexity(epoch_mean_ts)
            # fast_lzc_time_end = time.perf_counter()

            # lz76_time_start = time.perf_counter()
            # lzc = LZ76(epoch_mean_ts)
            # lz76_time_end = time.perf_counter()

            mse_time_start = time.perf_counter()
            mse_vector = multiscale_entropy(epoch_mean_ts)
            mse_mean = np.nanmean(mse_vector)
            mse_time_end = time.perf_counter()
                
            band_power_time_start = time.perf_counter()
            band_power_features = compute_band_power(epoch_mne, target_channels=None)
            band_power_time_end = time.perf_counter()

            conn_time_start = time.perf_counter()
            connectivity_features = compute_connectivity(epoch_mne)
            conn_time_end = time.perf_counter()

            processing_epoch_time_end = time.perf_counter()

            # print(f"LZC computation time: {lzc_time_end - lzc_time_start:.2f} seconds")
            # print(f"fast LZC computation time: {fast_lzc_time_end - fast_lzc_time_start:.2f} seconds")
            # print(f"LZ76 computation time: {lz76_time_end - lz76_time_start:.2f} seconds")
            print(f"MSE computation time: {mse_time_end - mse_time_start:.2f} seconds")
            print(f"Band power computation time: {band_power_time_end - band_power_time_start:.2f} seconds")
            print(f"Connectivity computation time: {conn_time_end - conn_time_start:.2f} seconds")
            print(f"Total epoch processing time: {processing_epoch_time_end - processing_epoch_time_start:.2f} seconds")