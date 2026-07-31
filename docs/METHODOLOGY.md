# Methodology

## Overview

This pipeline ingests pre-recorded EEG (no live acquisition), preprocesses signals, extracts epoch-level biomarkers, and trains classifiers for dementia group discrimination.

## Datasets

| Dataset | Task | Paradigm |
|---------|------|----------|
| 2 | `eyesclosed` | Eyes-closed resting state |
| 3 | `photomark` | Eyes-open photo stimulation |

Each dataset is ingested and trained **separately** to avoid mixing paradigms. Feature files are stored per dataset (`parquet_files/features_dataset{N}.parquet`).

## Label encoding

| Code | Group |
|------|-------|
| A | Alzheimer's disease |
| F | Frontotemporal dementia |
| C | Healthy control |

## Preprocessing protocol

Applied to raw `sub-*/eeg/*_task-{task}_eeg.set` files (not publisher derivatives). Production defaults are in `config.PREPROCESS_DEFAULTS`:

1. Band-pass filter: 0.5–40 Hz (FIR, zero-phase)
2. Notch filter: 50, 100, 150 Hz (line noise)
3. Bad-channel detection: flat/noisy channels interpolated
4. Average reference
5. ASR artifact removal (`ASR_CUTOFF=17`, aligned with Miltiadous et al.)
6. ICA: fit on 1 Hz high-pass copy; EOG components removed via Fp1/Fp2
7. Fixed-length epoching: 4 s windows, 2 s overlap
8. AutoReject: interpolate bad epochs (`n_interpolate=[1,2,3,4]`, `random_state=11`)

For quick dev runs, use `python scripts/ingest_features.py --fast` to skip steps 2–6 (bandpass + epoch + AR only).

Connectivity (wPLI) is computed once per subject across all epochs and attached to each epoch row.

Parameters are defined in `config.py` and snapshotted to `results/preprocessing_config.json` after each pipeline run.

## Feature extraction

Per epoch, the following features are computed:

| Feature | Description |
|---------|-------------|
| lzc_posterior | Lempel–Ziv complexity on posterior regional average (P3, Pz, P4, O1, O2), z-score normalized |
| mse_posterior | Multiscale entropy on posterior regional average, z-score normalized |
| rel_alpha, rel_beta, rel_theta, rel_delta | Relative band power (Welch PSD, `N_FFT=512`, resolution ≈ 1 Hz) |
| alpha_peak_freq | Peak frequency in alpha band |
| theta_alpha_ratio, theta_beta_ratio, slow_fast_ratio | Spectral ratios |
| theta_wpli, alpha_wpli | Weighted phase lag index connectivity (theta 4–8 Hz, alpha 8–13 Hz) |

Implementation: `util/extract_features.py`, `biomarkers/`.

## Train/test split

**Subject-level split** via `GroupShuffleSplit` in `classifier_models/train_utils.py`:

- All epochs from one participant stay in the same fold
- Default test fraction: 20% (`TEST_SIZE=0.2`, `RANDOM_STATE=42`)
- Split subject IDs saved per dataset in `results/subject_splits_dataset{N}.json`

This avoids epoch-level leakage where multiple epochs from the same recording appear in both train and test.

Training requires an explicit dataset: `python scripts/run_pipeline.py --train xgboost,mlp --dataset 2`

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
- Combining dataset 2 and dataset 3 features manually may confound resting-state and task-driven biomarkers

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact commands and artifact versions.

For a literature-based review of pipeline correctness and known issues, see [internal/PIPELINE_AUDIT.md](internal/PIPELINE_AUDIT.md).
