import mne
import numpy as np
import pandas as pd

import time

import matplotlib.pyplot as plt

from util import read_eeg_data, preprocess_EEG, plot_power_spectrum, plot_eeg, extract_eeg_features, get_participant_data, save_as_parquet, convert_to_epochs
from config import SAMPLING_RATE, TARGET_CHANNELS, DATASETS

numParticipants = 1 # change to 88 for all participants
i = 0

participant_data = get_participant_data()

"""
"participant_id" : participant_id
"Gender" : gender
"Age" : age
"Group" : group
"MMSE" : mmse
"""

for dataset in DATASETS:

    # for i in range(len(participant_data[0])):
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

        derivative.plot(show_scrollbars=False, block=True)
        plt.show()

        derivative_psd = derivative.compute_psd(
            method="welch",
            fmin=1,
            fmax=250,
            n_fft=2048,
            n_overlap=1024,
            picks=TARGET_CHANNELS
        )
        derivative_psd.plot()

        clean_eeg.plot(show_scrollbars=False, block=True)

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


        feature_extraction_time_start = time.perf_counter()
        all_epoch_features = extract_eeg_features(data)
        
        feature_extraction_time_end = time.perf_counter()

        print(f"Feature extraction took {feature_extraction_time_end - feature_extraction_time_start:.2f} seconds")

        df = pd.DataFrame(all_epoch_features)

        save_as_parquet(df, participant_id, dataset_id, label)
            
        print(df.head())