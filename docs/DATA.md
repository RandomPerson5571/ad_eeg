# Data Guide

## Source dataset

This project analyzes pre-recorded clinical EEG from:

**Miltiadous et al. (2023)** — *A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG*

- DOI: [10.3390/data8060095](https://doi.org/10.3390/data8060095)
- 88 subjects: 36 AD, 23 FTD, 29 healthy controls
- Eyes-closed resting state, 500 Hz, 19-channel 10–20 montage

## Obtaining raw data

1. Download the dataset from the publisher via the DOI above.
2. Extract into `EEG_data/` at the project root.
3. Validate layout:

```bash
python scripts/download_data.py
```

### Expected directory layout

```
EEG_data/
  dataset2/
    participants.tsv
    sub-001/eeg/sub-001_task-eyesclosed_eeg.set
  dataset3/
    participants.tsv
    sub-001/eeg/sub-001_task-photomark_eeg.set
```

Dataset aliases (see `configs/dataset.yaml`):

| Alias | Canonical name | Task |
|-------|----------------|------|
| `2`, `eyesclosed` | eyesclosed | eyesclosed |
| `3`, `photomark` | photomark | photomark |
| `all` | both datasets | — |

## Derived data layout

Raw EEG stays in `EEG_data/`. All derived artifacts are under `data/`:

```
data/
  audit/{dataset}/
    metadata.csv
    patient_summary.csv
    dataset_summary.json
    environment.json
  preprocessed/{dataset}/{experiment}/
    sub-001_raw.fif
    sub-001_filtered_raw.fif
    sub-001_ica_raw.fif
    sub-001_clean_raw.fif
    sub-001_epo.fif
    epochs/sub-001.npy
    metadata.json
    logs/sub-001.json
  features/{dataset}/{experiment}/
    subject_features.parquet
    selected_features.parquet
    feature_importance.csv
    logs/sub-001.json
  models/{dataset}/{experiment}/
    xgboost.joblib
    mlp.joblib
  results/{dataset}/{experiment}/
    benchmark.csv
    benchmark_detail.json
    benchmark_metadata.json
    predictions.csv
    figures/
    metrics.json
    confusion_matrix_xgboost.png
    roc_xgboost.png
```

### Experiment configs

Preprocessing parameters live in `experiments/` (not CLI flags). See [PREPROCESSING.md](PREPROCESSING.md) for stage-by-stage details and all experiment presets.

Each run writes `metadata.json` with MNE/Python versions, git commit, and config fingerprint.

### Legacy paths (compat)

| Legacy | New equivalent |
|--------|----------------|
| `parquet_files/features_dataset{N}.parquet` | `data/features/{name}/baseline/subject_features.parquet` |
| `classifier_models/saved_models/` | `data/models/{name}/{experiment}/` |
| `results/` | `data/results/{name}/{experiment}/` |

## Derived artifacts (Zenodo)

Large derived files are hosted separately on Zenodo (not in git). After uploading, update `data/manifest.json` with the record ID, then:

```bash
python scripts/fetch_artifacts.py --record-id YOUR_RECORD_ID
```

## Ethics

No new human subjects were recruited in this repository. All recordings come from the published Miltiadous dataset.
