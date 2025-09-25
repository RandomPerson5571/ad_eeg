import os

# ----------------------
# DATA PATHS
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "EEG_data")
PARQUET_DIR = os.path.join(BASE_DIR, "parquet_files")

# Datasets
DATASETS = [2, 3]      # Active datasets

# ----------------------
# EEG PARAMETERS
# ----------------------
SAMPLING_RATE = 500   # Hz

# Filtering
HP_FILTER = 1.0       # High-pass cutoff (Hz)
LP_FILTER = 40.0      # Low-pass cutoff (Hz)
NOTCH_FREQ = 60.0     # Set to None if not needed (e.g., 50.0 for EU)

# Epoching
EPOCH_LENGTH = 4.0    # seconds
EPOCH_OVERLAP = 2.0   # seconds

# ----------------------
# CHANNELS
# ----------------------
ALL_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
MASTOID_CHANNELS = ["A1", "A2"]
TARGET_CHANNELS = ["Fp1", "Fp2", "F7", "Cz"]

# ----------------------
# FREQUENCY BANDS
# ----------------------
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (12, 30),
    "gamma": (30, 40),
}

# ----------------------
# BIOMARKERS TO COMPUTE
# ----------------------
BIOMARKERS = {
    "band_power": True,
    "band_ratios": True,
    "complexity": True,       # Lempel-Ziv, etc.
    "connectivity": True,     # wPLI, coherence, etc.
    "entropy": True,          # Sample entropy, etc.
}

# ----------------------
# ML SETTINGS
# ----------------------
PARQUET_COMBINED_FILE = os.path.join(PARQUET_DIR, "all_features.parquet")

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42
