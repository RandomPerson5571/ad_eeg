import os

# ----------------------
# DATA PATHS
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "EEG_data")
CLEAN_DATA_DIR = os.path.join(BASE_DIR, "EEG_data", "EEG_clean_data")
PARQUET_DIR = os.path.join(BASE_DIR, "parquet_files")


# Datasets and their BIDS task names
DATASETS = [2, 3]
DATASET_TASKS = {
    2: "eyesclosed",
    3: "photomark",
}

# ----------------------
# EEG PARAMETERS
# ----------------------
SAMPLING_RATE = 500   # Hz

# Filtering
NOTCH_FREQ = [50, 100, 150]     # Set to None if not needed (e.g., 50.0 for EU)

# Preprocessing (production defaults — aligned with Miltiadous et al. ds004504)
ASR_CUTOFF = 17
ICA_RANDOM_STATE = 97
ICA_N_COMPONENTS = 0.95
BAD_CHANNEL_FLAT_STD = 1e-15
BAD_CHANNEL_NOISY_Z = 5.0

PREPROCESS_DEFAULTS = {
    "freq_filter": True,
    "notch_filter": True,
    "bad_channels": True,
    "referencing": True,
    "asr": True,
    "asr_cutoff": ASR_CUTOFF,
    "run_ica": True,
    "AR": True,
    "fle": True,
}

FAST_PREPROCESS_DEFAULTS = {
    "freq_filter": True,
    "notch_filter": False,
    "bad_channels": False,
    "referencing": False,
    "asr": False,
    "run_ica": False,
    "AR": True,
    "fle": True,
}

# Epoching
EPOCH_LENGTH = 4.0    # seconds
EPOCH_OVERLAP = 2.0   # seconds

# ----------------------
# CHANNELS
# ----------------------
ALL_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
MASTOID_CHANNELS = ["A1", "A2"]
TARGET_CHANNELS = ["Fp1", "Fp2", "F7", "Cz"]

# Welch PSD window (frequency resolution ≈ SAMPLING_RATE / N_FFT Hz)
N_FFT = 512

# Regional channel groups for complexity metrics
REGIONAL_CHANNELS = {
    "posterior": ["P3", "Pz", "P4", "O1", "O2"],
}

def parquet_path(dataset_id):
    return os.path.join(PARQUET_DIR, f"features_dataset{dataset_id}.parquet")


PARQUET_COMBINED_FILE = os.path.join(PARQUET_DIR, "all_features.parquet")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Feature columns used for classifier training
FEATURE_COLUMNS = [
    "lzc_posterior",
    "mse_posterior",
    "rel_alpha",
    "rel_beta",
    "rel_theta",
    "rel_delta",
    "alpha_peak_freq",
    "theta_alpha_ratio",
    "theta_beta_ratio",
    "slow_fast_ratio",
    "theta_wpli",
    "alpha_wpli",
]

# Zenodo record for derived artifacts (update after upload)
ZENODO_RECORD_ID = None

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42
