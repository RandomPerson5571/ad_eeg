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
    sub-002/eeg/...
    derivatives/sub-001/eeg/...   # optional reference preprocessing
  dataset3/
    participants.tsv
    sub-001/eeg/sub-001_task-photomark_eeg.set   # eyes-open photo stimulation
    ...

Task names per dataset are set in `config.py` (`DATASET_TASKS`).

### participants.tsv schema

| Column | Description |
|--------|-------------|
| participant_id | e.g. `sub-001` |
| Gender | M / F |
| Age | years |
| Group | `A` = Alzheimer's, `F` = FTD, `C` = Control |
| MMSE | Mini-Mental State Examination score |

## Derived artifacts (Zenodo)

Large derived files are hosted separately on Zenodo (not in git):

| File | Description |
|------|-------------|
| `parquet_files/all_features.parquet` | Epoch-level biomarker features |
| `classifier_models/saved_models/*.joblib` | Trained classifiers |
| `results/metrics.json` | Evaluation metrics |
| `results/preprocessing_config.json` | Parameter snapshot |

After uploading to Zenodo, update `data/manifest.json` with the record ID, then:

```bash
python scripts/fetch_artifacts.py --record-id YOUR_RECORD_ID
```

## Ethics

No new human subjects were recruited in this repository. All recordings come from the published Miltiadous dataset. Refer to the original paper for ethics approval and data-use terms.

## License

- **Code:** MIT (see `LICENSE`)
- **Raw EEG data:** governed by the source dataset license from the publisher
