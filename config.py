import os

# ----------------------
# DATA PATHS
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "EEG_data")
CLEAN_DATA_DIR = os.path.join(BASE_DIR, "EEG_data", "EEG_clean_data")
PARQUET_DIR = os.path.join(BASE_DIR, "parquet_files")


# Datasets
DATASETS = [2, 3]      # Active datasets

# ----------------------
# EEG PARAMETERS
# ----------------------
SAMPLING_RATE = 500   # Hz

# Filtering
NOTCH_FREQ = [50, 100, 150]     # Set to None if not needed (e.g., 50.0 for EU)

# Epoching
EPOCH_LENGTH = 4.0    # seconds
EPOCH_OVERLAP = 2.0   # seconds

# ----------------------
# CHANNELS
# ----------------------
ALL_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
MASTOID_CHANNELS = ["A1", "A2"]
TARGET_CHANNELS = ["Fp1", "Fp2", "F7", "Cz"]

PARQUET_COMBINED_FILE = os.path.join(PARQUET_DIR, "all_features.parquet")

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42
