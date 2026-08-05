# Methodology

## Overview

This pipeline ingests pre-recorded EEG (no live acquisition), preprocesses signals, extracts epoch-level biomarkers, and trains classifiers for dementia group discrimination.

## Datasets

| Dataset | Alias | Task | Paradigm |
|---------|-------|------|----------|
| eyesclosed | `2` | eyesclosed | Eyes-closed resting state |
| photomark | `3` | photomark | Eyes-open photo stimulation |

Each dataset is processed and trained **separately** via `--dataset eyesclosed` or `--dataset photomark`.

## Label encoding

| Code | Group |
|------|-------|
| A | Alzheimer's disease |
| F | Frontotemporal dementia |
| C | Healthy control |

## Preprocessing protocol (staged checkpoints)

Experiment configs in `experiments/` define preprocessing. Default `baseline`:

1. Load raw → `sub-NNN_raw.fif`
2. Band-pass 0.5–40 Hz + notch → `sub-NNN_filtered.fif`
3. ICA (if enabled) → `sub-NNN_ica.fif`
4. Bad channels + average ref + ASR → `sub-NNN_clean.fif`
5. Epoching (4 s, 2 s overlap) + AutoReject → `sub-NNN_epo.fif`

Stages are resumable: each subject log stores `raw_sha256` and `config_fingerprint`.

For quick dev: `--experiment fast` (bandpass + epoch + AR only).

```bash
python scripts/preprocess_dataset.py --dataset eyesclosed --experiment baseline
python scripts/extract_features.py --dataset eyesclosed --experiment baseline
```

Parameters are snapshotted in `data/preprocessed/{dataset}/{experiment}/metadata.json`.

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

Implementation: `eeg/features.py`, `biomarkers/`.

## Train/test split

**Subject-level split** via `GroupShuffleSplit` in `eeg/training.py`:

- All epochs from one participant stay in the same fold
- Default: `configs/training.yaml` (`test_size: 0.2`, `random_state: 42`)
- Split IDs saved in `data/results/{dataset}/{experiment}/subject_splits.json`

```bash
python scripts/train_model.py --dataset eyesclosed --experiment baseline --model xgboost,mlp
```

Training produces ROC curves and confusion matrices inline (no separate evaluate stage).

## Models

| Script | Model | Notes |
|--------|-------|-------|
| `scripts/train_model.py` | XGBoost + TargetEncoder | Bayesian hyperparameter search |
| `scripts/train_model.py` | MLP (128, 64) + StandardScaler | Early stopping |

Legacy trainers in `classifier_models/` remain as thin wrappers.

## Limitations

- Class imbalance across A / F / C groups
- No external validation on independent cohorts
- Connectivity features add compute cost per epoch
- `classifier_models/cnn.py` is a backward-compatible alias for the MLP trainer
- Combining dataset 2 and dataset 3 features manually may confound resting-state and task-driven biomarkers

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact commands and artifact versions.

For a literature-based review of pipeline correctness and known issues, see [internal/PIPELINE_AUDIT.md](internal/PIPELINE_AUDIT.md).
