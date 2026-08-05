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

## Full reproduction from raw data (staged pipeline)

```bash
# 1. Validate data layout
python scripts/download_data.py

# 2. Inspect one subject (local dev)
python scripts/inspect_preprocessing.py --dataset eyesclosed --subject sub-001

# 3. Preprocess with checkpoints (resume automatic)
python scripts/preprocess_dataset.py --dataset eyesclosed --experiment baseline --workers 4

# 4. Extract features from epochs only
python scripts/extract_features.py --dataset eyesclosed --experiment baseline --workers 4

# 5. Train + evaluate (ROC, confusion matrix inline)
python scripts/train_model.py --dataset eyesclosed --experiment baseline --model xgboost,mlp

# Or run all stages:
python pipeline.py --dataset eyesclosed --experiment baseline --stages preprocess,features,train --workers 4
```

### Fast smoke test

```bash
python scripts/preprocess_dataset.py --dataset eyesclosed --experiment fast --limit 2
python scripts/extract_features.py --dataset eyesclosed --experiment fast --limit 2
```

### Overnight (Kaggle)

```bash
python scripts/preprocess_dataset.py --dataset all --experiment baseline --workers 4
python scripts/extract_features.py --dataset all --experiment baseline --workers 8
python scripts/train_model.py --dataset eyesclosed --experiment baseline --model xgboost,mlp
```

## Idempotency

- Each subject log stores `raw_sha256` and `config_fingerprint`.
- Re-running a stage skips subjects whose checkpoints match the current raw file and experiment config.
- Use `--force` to ignore checkpoints and rerun.

## Fixed random seeds

| Component | Seed |
|-----------|------|
| Train/test split | `configs/training.yaml` → `random_state: 42` |
| AutoReject | `11` |
| ICA | `experiments/*.yaml` → `ica_random_state: 97` |

## Legacy commands (deprecated)

```bash
python scripts/ingest_features.py --dataset 2 --all   # → preprocess + extract
python scripts/run_pipeline.py --train xgboost,mlp --dataset 2
```

## Zenodo upload checklist

1. Run full pipeline locally
2. Upload: `data/features/`, `data/models/`, `data/results/`
3. Mint DOI on Zenodo
4. Update `data/manifest.json`
5. Commit manifest update to git
