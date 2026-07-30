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

# 2. Extract features (all subjects, both datasets)
python scripts/ingest_features.py --all-datasets --all

# 3. Train models with subject-level splits
python scripts/run_pipeline.py --train xgboost,mlp

# Or run everything:
python scripts/run_pipeline.py --ingest --all-datasets --train xgboost,mlp
```

## Reproduction from Zenodo artifacts

If you only need to retrain or evaluate without re-processing raw EEG:

```bash
python scripts/fetch_artifacts.py --record-id YOUR_ZENODO_RECORD_ID
python classifier_models/train_XGBoost.py
python classifier_models/train_mlp.py
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
| `parquet_files/all_features.parquet` | Feature store |
| `results/ingest_log.json` | Per-subject ingest status |
| `results/preprocessing_config.json` | Config snapshot |
| `results/metrics.json` | Combined model metrics |
| `results/metrics_xgboost.json` | XGBoost-only metrics |
| `results/metrics_mlp.json` | MLP-only metrics |
| `classifier_models/saved_models/*.joblib` | Model bundles |

## Zenodo upload checklist

1. Run full pipeline locally
2. Upload: parquet, models, metrics JSON files
3. Mint DOI on Zenodo
4. Update `data/manifest.json` with `zenodo_record_id` and `doi`
5. Commit manifest update to git
