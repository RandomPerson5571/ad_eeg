# Reproducibility Guide

## Environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Python 3.10+ recommended.

## Full reproduction from raw data

```bash
# 1. Validate data layout
python scripts/download_data.py

# 2. Extract features (per dataset)
python scripts/ingest_features.py --dataset 2 --all
python scripts/ingest_features.py --dataset 3 --all

# Quick smoke test (minimal preprocessing — bandpass + epoch + AR only):
python scripts/ingest_features.py --dataset 2 --subject 1 --fast

# 3. Train models with subject-level splits (one dataset at a time)
python scripts/run_pipeline.py --train xgboost,mlp --dataset 2

# Or ingest and train for one dataset:
python scripts/ingest_features.py --dataset 2 --all
python scripts/run_pipeline.py --train xgboost,mlp --dataset 2
```

## Reproduction from Zenodo artifacts

If you only need to retrain or evaluate without re-processing raw EEG:

```bash
python scripts/fetch_artifacts.py --record-id YOUR_ZENODO_RECORD_ID
python classifier_models/train_XGBoost.py --dataset 2
python classifier_models/train_mlp.py --dataset 2
```

## Fixed random seeds

| Component | Seed |
|-----------|------|
| Train/test split | `RANDOM_STATE=42` |
| AutoReject | `11` |
| ICA (if enabled) | `97` |
| MLP / XGBoost | `RANDOM_STATE=42` |

## Output artifacts

| Path | Contents |
|------|----------|
| `parquet_files/features_dataset{N}.parquet` | Per-dataset feature store |
| `parquet_files/all_features.parquet` | Legacy combined feature store |
| `results/ingest_log.json` | Per-subject ingest status |
| `results/preprocessing_config.json` | Config snapshot |
| `results/metrics_dataset{N}.json` | Combined model metrics per dataset |
| `results/metrics_xgboost_dataset{N}.json` | XGBoost-only metrics |
| `results/metrics_mlp_dataset{N}.json` | MLP-only metrics |
| `results/subject_splits_dataset{N}.json` | Train/test subject IDs per dataset |
| `classifier_models/saved_models/*_dataset{N}.joblib` | Model bundles per dataset |

## Zenodo upload checklist

1. Run full pipeline locally
2. Upload: parquet, models, metrics JSON files
3. Mint DOI on Zenodo
4. Update `data/manifest.json` with `zenodo_record_id` and `doi`
5. Commit manifest update to git
