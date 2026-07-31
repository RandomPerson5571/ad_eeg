# EEG Dementia Classification Pipeline

[![Dataset DOI](https://img.shields.io/badge/Dataset-10.3390%2Fdata8060095-blue)](https://doi.org/10.3390/data8060095)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Reproducible pipeline for EEG-based classification of Alzheimer's disease, frontotemporal dementia, and healthy controls using hand-crafted biomarkers and classical ML.

> **Note:** This project ingests pre-recorded clinical EEG. There is no live data acquisition hardware integration.

## Quick start

```bash
git clone <your-repo-url>
cd eeg-project

python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 1. Obtain raw EEG data

Download from [Miltiadous et al. (DOI 10.3390/data8060095)](https://doi.org/10.3390/data8060095) and extract into `EEG_data/`.

```bash
python scripts/download_data.py
```

See [docs/DATA.md](docs/DATA.md) for layout and citation details.

### 2. Extract features

```bash
# All subjects, both datasets
python scripts/ingest_features.py --dataset 2 --all
python scripts/ingest_features.py --dataset 3 --all

# Single subject (testing)
python scripts/ingest_features.py --dataset 2 --subject 1
```

### 3. Train models

```bash
python scripts/run_pipeline.py --train xgboost,mlp --dataset 2
```

Or individually:

```bash
python classifier_models/train_XGBoost.py --dataset 2
python classifier_models/train_mlp.py --dataset 2
```

### 4. End-to-end

```bash
python scripts/ingest_features.py --dataset 2 --all
python scripts/run_pipeline.py --train xgboost,mlp --dataset 2
```

## Zenodo artifacts

Derived files (features, models, metrics) can be downloaded from Zenodo after upload:

```bash
python scripts/fetch_artifacts.py --record-id YOUR_RECORD_ID
```

Update `data/manifest.json` with your Zenodo record ID after publishing artifacts.

## Jupyter notebooks

```bash
jupyter lab notebooks/
```

| Notebook | Description |
|----------|-------------|
| [01_dataset_overview](notebooks/01_dataset_overview.ipynb) | Demographics and class balance |
| [02_preprocessing_and_qc](notebooks/02_preprocessing_and_qc.ipynb) | Single-subject preprocessing |
| [03_feature_extraction](notebooks/03_feature_extraction.ipynb) | Feature distributions |
| [04_model_training](notebooks/04_model_training.ipynb) | Subject-level model training |
| [05_results_and_interpretation](notebooks/05_results_and_interpretation.ipynb) | Metrics and limitations |

### Manual QC

Inspect preprocessing and features without notebooks:

```bash
python scripts/inspect_qc.py preprocess --dataset 2 --subject 1
python scripts/inspect_qc.py features
python scripts/inspect_qc.py spot-check --dataset 2 --subject 1
```

Reports are saved under `results/qc/`.

## Project structure

```
├── biomarkers/           # Feature computation (band power, complexity, connectivity)
├── classifier_models/    # XGBoost and MLP trainers
├── docs/                 # Data, methodology, reproducibility guides
├── notebooks/            # Interactive walkthroughs
├── scripts/              # CLI entry points (ingest, train, download, fetch, inspect_qc)
├── util/                 # I/O, preprocessing, feature extraction, QC
├── config.py             # Paths and parameters
└── EEG_data/             # Raw EEG (gitignored, download separately)
```

## Documentation

- [Data guide](docs/DATA.md) — source dataset, layout, Zenodo artifacts
- [Methodology](docs/METHODOLOGY.md) — preprocessing, features, evaluation
- [Pipeline audit](docs/PIPELINE_AUDIT.md) — literature review of preprocessing and feature extraction
- [Reproducibility](docs/REPRODUCIBILITY.md) — exact commands and seeds

## Inference example

```python
import joblib
import pandas as pd

artifact = joblib.load("classifier_models/saved_models/xgboost_eeg_classifier_dataset2.joblib")
pipeline = artifact["pipeline"]
feature_names = artifact["feature_names"]

example = pd.DataFrame([{name: 0.0 for name in feature_names}])
example["rel_alpha"] = 0.12
example["lzc_posterior"] = 0.52

prediction = pipeline.predict(example)
label = artifact["label_encoder"].inverse_transform(prediction)
print("Predicted group:", label)
```

## Citation

If you use this code or the underlying dataset, please cite:

```bibtex
@article{miltiadous2023dataset,
  title={A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG},
  author={Miltiadous, Andreas and others},
  journal={Data},
  year={2023},
  doi={10.3390/data8060095}
}
```

See also [CITATION.cff](CITATION.cff).

## License

Code: MIT — see [LICENSE](LICENSE). Raw EEG data is governed by the source dataset license.
