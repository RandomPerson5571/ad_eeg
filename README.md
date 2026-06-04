# EEG Project

## Overview

This repository supports EEG data ingestion, preprocessing, feature extraction, and classifier training for dementia-related EEG analysis.

The current production-ready pipelines include:

- `main.py` for reading raw EEG, cleaning, epoching, and saving features to `parquet_files/all_features.parquet`
- `classifier_models/train_XGBoost.py` for training a tuned XGBoost classifier
- `classifier_models/cnn.py` for training a shallow neural network classifier on extracted features

## Requirements

Install the Python dependencies listed in `requirements.txt` and activate the project virtual environment before running any scripts.

Recommended installation:

```powershell
cd "C:\Users\griff\Downloads\EEG Project Folder"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data preparation

1. Place the EEG data under `EEG_data/` in the expected BIDS-like structure.
2. Confirm dataset IDs in `config.py` via `DATASETS`.
3. Run `main.py` to preprocess EEG and save extracted features into the parquet store.

> Note: `main.py` currently processes a single participant by default. Extend the loop in `main.py` or adapt it to your dataset if you need multi-subject feature extraction.

## Feature extraction

The feature extraction pipeline lives in `util/extract_features.py` and uses:

- Lempel–Ziv complexity
- multiscale entropy
- relative band power
- spectral ratios
- connectivity features

Extracted features are saved through `util/io.py` into `parquet_files/all_features.parquet`.

## Training models

### Train the XGBoost production classifier

```powershell
python classifier_models\train_XGBoost.py
```

This script:

- loads features from `parquet_files/all_features.parquet`
- splits data into train/test sets
- performs Bayesian hyperparameter search over the XGBoost classifier
- saves the tuned model to `classifier_models/saved_models/xgboost_eeg_classifier.joblib`

### Train the EEG MLP classifier

```powershell
python classifier_models\cnn.py
```

This script:

- loads the same feature store
- trains a feature-based neural network classifier
- saves the model to `classifier_models/saved_models/eeg_mlp_classifier.joblib`

## Production notes

- Ensure `parquet_files/all_features.parquet` exists before training.
- Use `joblib.load` to restore trained models for inference.
- Store the trained model artifacts in `classifier_models/saved_models`.
- Use the saved model path as a source of truth for production inference.

### Example inference snippet

```python
import joblib
import pandas as pd

artifact = joblib.load("classifier_models/saved_models/xgboost_eeg_classifier.joblib")
pipeline = artifact["pipeline"] if isinstance(artifact, dict) else artifact

example_features = pd.DataFrame([
    {
        "lzc": 0.52,
        "mse_mean": 0.95,
        "rel_alpha": 0.12,
        "rel_beta": 0.15,
        "rel_theta": 0.22,
        "rel_delta": 0.10,
        "alpha_peak_freq": 10.2,
        "theta_alpha_ratio": 1.8,
        "theta_beta_ratio": 1.4,
        "slow_fast_ratio": 0.8,
    }
])

prediction = pipeline.predict(example_features)
print("Predicted label:", prediction)
```

## Recommended production workflow

1. Run raw EEG ingestion and preprocessing.
2. Extract and save epoch-level features.
3. Train a model on the full feature store.
4. Save the final model artifact and use it for inference in a separate deployment script.
