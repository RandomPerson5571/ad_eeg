# Methodology

## Overview

This pipeline ingests pre-recorded EEG (no live acquisition), preprocesses signals, extracts epoch-level biomarkers, and trains classifiers for dementia group discrimination.

## Label encoding

| Code | Group |
|------|-------|
| A | Alzheimer's disease |
| F | Frontotemporal dementia |
| C | Healthy control |

## Preprocessing protocol

Applied to raw `sub-*/eeg/*_task-eyesclosed_eeg.set` files (not publisher derivatives):

1. Band-pass filter: 0.5–40 Hz (FIR, zero-phase)
2. Fixed-length epoching: 4 s windows, 2 s overlap
3. AutoReject: interpolate bad epochs (`n_interpolate=[1,2,3,4]`, `random_state=11`)

Connectivity (wPLI) is computed once per subject across all epochs and attached to each epoch row.

Optional steps (disabled by default): notch filter, ASR, ICA, average referencing.

Parameters are defined in `config.py` and snapshotted to `results/preprocessing_config.json` after each pipeline run.

## Feature extraction

Per epoch, the following features are computed:

| Feature | Description |
|---------|-------------|
| lzc | Lempel–Ziv complexity (mean across channels) |
| mse_mean | Mean multiscale entropy |
| rel_alpha, rel_beta, rel_theta, rel_delta | Relative band power |
| alpha_peak_freq | Peak frequency in alpha band |
| theta_alpha_ratio, theta_beta_ratio, slow_fast_ratio | Spectral ratios |
| theta_wpli, alpha_wpli | Weighted phase lag index connectivity |

Implementation: `util/extract_features.py`, `biomarkers/`.

## Train/test split

**Subject-level split** via `GroupShuffleSplit` in `classifier_models/train_utils.py`:

- All epochs from one participant stay in the same fold
- Default test fraction: 20% (`TEST_SIZE=0.2`, `RANDOM_STATE=42`)
- Split subject IDs saved in model artifacts and `results/subject_splits.json`

This avoids epoch-level leakage where multiple epochs from the same recording appear in both train and test.

## Models

| Script | Model | Notes |
|--------|-------|-------|
| `classifier_models/train_XGBoost.py` | XGBoost + TargetEncoder | Bayesian hyperparameter search, balanced accuracy scoring |
| `classifier_models/train_mlp.py` | MLP (128, 64) + StandardScaler | Early stopping, feature-based (not raw-waveform CNN) |

## Limitations

- Class imbalance across A / F / C groups
- No external validation on independent cohorts
- Connectivity features add compute cost per epoch
- `classifier_models/cnn.py` is a backward-compatible alias for the MLP trainer

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact commands and artifact versions.
