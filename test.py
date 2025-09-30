from util import get_participant_data, read_eeg_data, extract_eeg_features, preprocess_EEG, plot_eeg, plot_power_spectrum
from config import SAMPLING_RATE, TARGET_CHANNELS, ALL_CHANNELS

import pandas as pd

path = "EEG_data\dataset2\sub-001\eeg\sub-001_task-eyesclosed_eeg.set"

eeg_raw = read_eeg_data(path, SAMPLING_RATE)

plot_eeg(eeg_raw, target_channels=TARGET_CHANNELS)

derivative_path = "EEG_data\dataset2\derivatives\sub-001\eeg\sub-001_task-eyesclosed_eeg.set"
eeg_derivative = read_eeg_data(derivative_path, SAMPLING_RATE)
plot_power_spectrum(eeg_derivative, target_channels=TARGET_CHANNELS)
plot_power_spectrum(eeg_derivative, target_channels=ALL_CHANNELS)

plot_eeg(eeg_derivative, target_channels=TARGET_CHANNELS)

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