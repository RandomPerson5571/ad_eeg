import mne
import numpy as np
import pandas as pd

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

    for i in range(numParticipants):

        ind = dataset-1

        participant_id = participant_data[ind].iloc[0, i]
        dataset_id = dataset
        label = participant_data[ind].loc[participant_data[ind]['participant_id'] == participant_id, 'Group'].values[0]

        participant_path = "EEG_data\\dataset2\\sub-{0:03d}\\eeg\\sub-{0:03d}_task-eyesclosed_eeg.set".format(i+1)

        print(f"Processing participant {i+1}/88 (dataset {dataset})")
        print(f"Reading data for participant {participant_id} from {participant_path}")

        eeg_raw = read_eeg_data(participant_path, sfreq=SAMPLING_RATE)

        clean_eeg = preprocess_EEG(eeg_raw)

        epochs = convert_to_epochs(clean_eeg)

        data = epochs.get_data()
            
        print(f"Data shape (n_epochs, n_channels, n_times): {data.shape}")

        features = []

        derivatives_path = "EEG_data\\dataset{0}\\derivatives\\sub-{1:03d}\\eeg\\sub-{1:03d}_task-eyesclosed_eeg.set".format(dataset, i+1)
        derivative = read_eeg_data(derivatives_path, sfreq=SAMPLING_RATE)

        plot_eeg(derivative, target_channels=TARGET_CHANNELS)
        plot_power_spectrum(derivative, target_channels=TARGET_CHANNELS)
        plot_eeg(clean_eeg, target_channels=TARGET_CHANNELS)
        plot_power_spectrum(clean_eeg, target_channels=TARGET_CHANNELS)

        all_epoch_features = extract_eeg_features(data)

        df = pd.DataFrame(all_epoch_features)

        save_as_parquet(df, participant_id, dataset_id, label)
            
        print(df.head())