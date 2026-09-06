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

See **[PREPROCESSING.md](PREPROCESSING.md)** for the full pipeline reference: stage-by-stage signal processing, experiment configs, checkpoint layout, QC, and resume behavior.

Summary (default `baseline` experiment):

1. Load raw → `sub-NNN_raw.fif`
2. Band-pass 0.5–40 Hz + notch → `sub-NNN_filtered_raw.fif`
3. ICA (if enabled) → `sub-NNN_ica_raw.fif`
4. Bad channels + average ref + ASR → `sub-NNN_clean_raw.fif`
5. Epoching (4 s, 2 s overlap) + AutoReject → `sub-NNN_epo.fif`

```bash
python scripts/preprocess_dataset.py --dataset eyesclosed --experiment baseline
python scripts/extract_features.py --dataset eyesclosed --experiment baseline
```

## Feature extraction

Per epoch, the following features are computed:

| Feature | Description |
|---------|-------------|
| lzc_posterior | Lempel–Ziv complexity on posterior regional average (P3, Pz, P4, O1, O2), z-score normalized |
| mse_posterior | Multiscale entropy on posterior regional average, z-score normalized |
| rel_alpha, rel_beta, rel_theta, rel_delta | Relative band power (Welch PSD, `N_FFT=512`, resolution ≈ 1 Hz) |
| alpha_peak_freq | Peak frequency in alpha band |
| theta_alpha_ratio, theta_beta_ratio, slow_fast_ratio | Spectral ratios |
| theta_wpli, alpha_wpli | Mean weighted phase lag index over unique off-diagonal channel pairs (theta 4–8 Hz, alpha 8–13 Hz) |

Implementation: `eeg/features.py`, `biomarkers/`.

## Evaluation

Evaluation uses nested subject-stratified grouped cross-validation: subjects are
stratified first, then each split is expanded to all of that subject's epochs.

- All epochs from one participant stay together in every inner and outer split.
- Variance, correlation, and mutual-information feature selection are fitted inside each fold.
- Inner folds select hyperparameters; outer folds produce unbiased out-of-fold (OOF) predictions.
- Epoch probabilities are averaged per participant before classification and scoring.
- Reported metrics, plots, and bootstrap confidence intervals use all outer-fold subject OOF predictions.
- PR AUC is macro one-vs-rest average precision (binary runs use positive-class average precision).
- `predictions.csv` contains one row per subject; `epoch_predictions.csv` retains diagnostic epoch outputs.

Fold assignments, selected features, and inner-CV parameters are saved in
`benchmark_detail.json`. Defaults are defined in `configs/training.yaml`.

```bash
python scripts/train_model.py --dataset eyesclosed --experiment baseline --model xgboost,mlp
```

Training produces OOF ROC, precision-recall, calibration, and confusion-matrix figures.

## Follow-up analysis suite

After the primary benchmark, run the complete follow-up analysis with:

```bash
python scripts/run_extended_analysis.py \
  --dataset eyesclosed --experiment baseline --permutations 1000
```

The suite keeps the same outer subject folds and performs all model selection on
inner grouped folds. It writes `analysis_summary.csv`, a human-readable
`extended_analysis.md` report, per-run `fold_metrics.csv` files, and
`permutation_scores.csv` under
`data/results/{dataset}/{experiment}/analysis/`. It includes:

- sigmoid probability calibration and binary/class-specific threshold tuning;
- L1, L2, and elastic-net logistic regression regularization sweeps;
- individual band-power, spectral, connectivity, complexity, posterior-region,
  and global-feature ablations;
- inner-fold univariate top-k, L1-based, and PCA selection comparisons; and
- a subject-label permutation null distribution with a corrected empirical
  p-value.

The current 12-column feature table does not contain channel-specific features,
so channel-level ablations are reported as unavailable rather than being
silently approximated. To enable them, add channel-resolved columns and list
their groups under `training.analysis.channel_groups` in the training config.

The ordinary `scripts/benchmark.py` command retains the original argmax policy.
Threshold tuning is valid only when enabled because its operating point must be
selected inside the inner CV, never from outer test subjects.

## Models

| Script | Model | Notes |
|--------|-------|-------|
| `scripts/train_model.py` | XGBoost | Nested grouped grid search |
| `scripts/train_model.py` | MLP (128, 64) + StandardScaler | Nested grouped grid search |
| `notebooks/kaggle/06_deep_learning.ipynb` | EEGNet-style CNN | Raw epoch arrays, subject-level CV and early stopping |

## Deep-learning evaluation

Notebook 06 consumes the float32 epoch arrays exported by notebook 02. Its compact
EEGNet-style convolutional network is evaluated with stratified subject-level outer
folds. A separate subject-level subset of each training fold controls early stopping;
outer test subjects are never used for normalization, sampling, or model selection.
Per-channel normalization statistics are learned from training subjects only.

Training samples are weighted so each diagnostic class has equal total mass and each
subject within a class contributes equal mass regardless of epoch count. Epoch
probabilities are averaged per subject before calculating balanced accuracy, macro F1,
MCC, ROC AUC, PR AUC, confusion matrices, and bootstrap confidence intervals.

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
