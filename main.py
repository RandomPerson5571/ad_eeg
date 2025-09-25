import mne
import numpy as np
import pandas as pd

from util import read_eeg_data, preprocess_EEG, plot_power_spectrum, plot_eeg, extract_eeg_features, get_participant_data, save_as_parquet
from config import SAMPLING_RATE, TARGET_CHANNELS

numParticipants = 1
i = 0 # change to 88 for all participants

participant_data = get_participant_data()

for i in range(numParticipants):

    participant_id = participant_data[i]["participant_id"]
    participant_path = "EEG_data\\dataset2\\sub-{0:03d}\\eeg\\sub-{0:03d}_task-eyesclosed_eeg.set".format(i+1)

    print(f"Processing participant {i+1}/88")
    print(f"Reading data for participant {participant_id} from {participant_path}")

    eeg_raw = read_eeg_data(participant_path, sfreq=SAMPLING_RATE)

    epochs = preprocess_EEG(eeg_raw)

    data = epochs.get_data()
    
    print(f"Data shape (n_epochs, n_channels, n_times): {data.shape}")

    features = []

    # derivatives_path = "EEG_data\\dataset2\\derivatives\\sub-{0:03d}\\eeg\\sub-{0:03d}_task-eyesclosed_eeg.set".format(i+1)
    derivatives_path = "EEG_data\dataset2\derivatives\sub-001\eeg\sub-001_task-eyesclosed_eeg.set"
    derivative = read_eeg_data(derivatives_path, sfreq=SAMPLING_RATE)

    plot_eeg(derivative, target_channels=TARGET_CHANNELS)
    plot_power_spectrum(derivative, target_channels=TARGET_CHANNELS)
    plot_eeg(epochs, target_channels=TARGET_CHANNELS)
    plot_power_spectrum(epochs, target_channels=TARGET_CHANNELS)

    # all_epoch_features = extract_eeg_features(data)

    # features.append(all_epoch_features)

    df = pd.DataFrame(features)
    print(df.head())