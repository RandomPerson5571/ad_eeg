import json
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from config import (
    BASE_DIR,
    CLEAN_DATA_DIR,
    DATASETS,
    DATASET_TASKS,
    FEATURE_COLUMNS,
    PARQUET_COMBINED_FILE,
    RAW_DATA_DIR,
    RESULTS_DIR,
)


def read_eeg_data(file_path, sfreq):
    if os.path.splitext(file_path)[1].lower() == ".set":
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
    else:
        files = [f for f in os.listdir(file_path) if f.endswith(".txt")]

        all_data = []
        ch_names = []

        for f in files:
            path = os.path.join(file_path, f)
            signal = np.loadtxt(path)
            all_data.append(signal)
            ch_names.append(f.split(".")[0])

        data = np.vstack(all_data)
        ch_types = ["eeg"] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
    return raw


def raw_eeg_path(dataset_id, subject_num, task=None):
    task = task or DATASET_TASKS.get(dataset_id, "eyesclosed")
    return os.path.join(
        RAW_DATA_DIR,
        f"dataset{dataset_id}",
        f"sub-{subject_num:03d}",
        "eeg",
        f"sub-{subject_num:03d}_task-{task}_eeg.set",
    )


def derivatives_eeg_path(dataset_id, subject_num, task=None):
    task = task or DATASET_TASKS.get(dataset_id, "eyesclosed")
    return os.path.join(
        RAW_DATA_DIR,
        f"dataset{dataset_id}",
        "derivatives",
        f"sub-{subject_num:03d}",
        "eeg",
        f"sub-{subject_num:03d}_task-{task}_eeg.set",
    )


def save_as_parquet(df, participant_id, dataset_id, label):
    combined_parquet_path = PARQUET_COMBINED_FILE
    df = df.copy()
    df["participant_id"] = participant_id
    df["dataset_id"] = dataset_id
    df["label"] = label

    os.makedirs(os.path.dirname(combined_parquet_path), exist_ok=True)

    if os.path.exists(combined_parquet_path):
        existing = pd.read_parquet(combined_parquet_path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_parquet(combined_parquet_path, engine="pyarrow")
        print(f"Combined dataset saved to {combined_parquet_path}")
    else:
        df.to_parquet(combined_parquet_path, engine="pyarrow")
        print(f"Initialized new dataset and saved to {combined_parquet_path}")


def load_features_df():
    return pd.read_parquet(PARQUET_COMBINED_FILE)


def load_features(include_metadata=False):
    df = load_features_df()
    features = df[FEATURE_COLUMNS]
    labels = df["label"]

    if include_metadata:
        metadata = df[["participant_id", "dataset_id", "epoch_id"]]
        return features, labels, metadata
    return features, labels


def save_clean_eeg(clean_eeg, dataset, participant):
    dataset_id = "dataset" + str(dataset)
    participant_id = "sub-" + str(participant)
    clean_eeg_dir = os.path.join(CLEAN_DATA_DIR, dataset_id, participant_id)
    os.makedirs(os.path.dirname(clean_eeg_dir), exist_ok=True)
    clean_eeg.save(clean_eeg_dir, overwrite=True)


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


def list_subjects(dataset_id):
    participants = parse_data_file(
        os.path.join(RAW_DATA_DIR, f"dataset{dataset_id}", "participants.tsv"),
        dataset_id,
    )
    return participants


def write_ingest_log(entries, path=None):
    path = path or os.path.join(RESULTS_DIR, "ingest_log.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"Ingest log saved to {path}")
