# Notebooks

Interactive walkthroughs for the EEG dementia classification pipeline.

## Setup

```bash
pip install -r requirements.txt
python -m ipykernel install --user --name=eeg-project
jupyter lab
```

Raw EEG data must be present under `EEG_data/` (see `docs/DATA.md`) or fetch precomputed features from Zenodo via `scripts/fetch_artifacts.py`.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_dataset_overview.ipynb` | Participant demographics and class balance |
| `02_preprocessing_and_qc.ipynb` | Single-subject preprocessing QC (PSD, AutoReject, epochs) |
| `03_feature_extraction.ipynb` | Feature distributions, correlations, spot-check vs parquet |
| `04_model_training.ipynb` | Subject-level split training |
| `05_results_and_interpretation.ipynb` | Metrics, confusion matrices, limitations |

Each notebook calls functions from `util/` and `classifier_models/` — logic is not duplicated in notebook cells.

## Manual QC (CLI)

Without opening a notebook, generate inspection reports:

```bash
# Preprocessing QC for one subject (PNG + JSON in results/qc/preprocessing/)
python scripts/inspect_qc.py preprocess --dataset 2 --subject 1

# Feature store QC (PNGs + JSON in results/qc/features/)
python scripts/inspect_qc.py features

# Re-extract one subject and compare to parquet
python scripts/inspect_qc.py spot-check --dataset 2 --subject 1
```
