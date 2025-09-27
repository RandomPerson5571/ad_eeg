import os
import mne
import numpy as np

from config import BASE_DIR, PARQUET_COMBINED_FILE, DATASETS, CLEAN_DATA_DIR
import pandas as pd

def read_eeg_data(file_path, sfreq):
    if os.path.splitext(file_path)[1].lower() == ".set":
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
    else: # reads the folder with EEG data in .txt format
        files = [f for f in os.listdir(file_path) if f.endswith('.txt')]
        
        all_data = []
        ch_names = []

        for f in files:
            path = os.path.join(file_path, f)
            signal = np.loadtxt(path)
            all_data.append(signal)
            ch_names.append(f.split('.')[0])

        data = np.vstack(all_data)

        ch_types = ["eeg"] * len(ch_names)

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
    return raw

def save_as_parquet(df, participant_id, dataset_id, label):
    combined_parquet_path = PARQUET_COMBINED_FILE
    df["participant_id"] = participant_id
    df["dataset_id"] = dataset_id
    df["label"] = label
    # Group A - Alzheimer's
    # Group C - Control (healthy)
    # Group F - Frontaltemporal Dementia

    if os.path.exists(combined_parquet_path):
        existing = pd.read_parquet(combined_parquet_path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_parquet(combined_parquet_path, engine="pyarrow")
    else:
        df.to_parquet(combined_parquet_path, engine="pyarrow")

    print(f"Combined dataset saved to {combined_parquet_path}")

def load_features():
    combined_parquet_path = PARQUET_COMBINED_FILE
    df = pd.read_parquet(combined_parquet_path)

    features = df[[
    'lzc', 'mse_mean', 'rel_alpha', 'rel_beta',
    'rel_theta', 'rel_delta', 'alpha_peak_freq',
    'theta_alpha_ratio', 'theta_beta_ratio', 'slow_fast_ratio'
    ]]
    labels = df["label"]

    return features, labels

def save_clean_eeg(clean_eeg, dataset, participant):
    dataset_id = "dataset" + str(dataset)
    participant_id = "sub-" + str(participant)
    clean_eeg_dir = os.path.join(CLEAN_DATA_DIR, dataset_id, participant_id)

    clean_eeg.save(clean_eeg_dir, overwrite=True)


# dataset1_AD_eyes_closed = os.path.join(BASE_DIR, "EEG_data", "Dataset1", "AD", "Eyes_closed")
# dataset1_AD_eyes_open = os.path.join(BASE_DIR, "EEG_data", "Dataset1", "AD", "Eyes_open")
# dataset1_healthy_eyes_closed = os.path.join(BASE_DIR, "EEG_data", "Dataset1", "Healthy", "Eyes_closed")
# dataset1_healthy_eyes_open = os.path.join(BASE_DIR, "EEG_data", "Dataset1", "Healthy", "Eyes_open")
# backup data

def get_participant_data():
    participants = []

    for dataset in DATASETS:
        dataset_id = "dataset" + str(dataset)
        participant_data = os.path.join(BASE_DIR, "EEG_data", dataset_id, "participants.tsv")
        participants.append(parse_data_file(participant_data, dataset))

    return participants

def parse_data_file(participant_data, dataset):
    df = pd.read_csv(participant_data, sep="\t")
    
    df["Dataset"] = dataset

    return df

# def parse_data_file(participant_data, dataset):
#     participants = []

#     with open(participant_data, 'r') as participant_file:
#         header = participant_file.readline().strip().split('\t')
#         for line in participant_file:
#             fields = line.strip().split('\t')
#             if len(fields) == len(header):
#                 participant = dict(zip(header, fields))
#                 participant["dataset"] = dataset
#                 participants.append(participant)
#     return participants