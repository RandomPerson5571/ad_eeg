# Kaggle Notebook Pipeline

**These notebooks run on [Kaggle](https://www.kaggle.com) cloud CPUs — not on your laptop.**

Upload each notebook to Kaggle, enable **Internet**, attach input datasets, run, then **Save Version → Create Dataset** to chain stages.

| Notebook | Stage | Kaggle inputs |
|----------|-------|---------------|
| `00_dataset_audit` | Raw EEG audit | Raw EEG dataset |
| `01_preprocessing` | Preprocess + epoch | Raw EEG dataset |
| `02_epoching` | Export `.npy` epochs | Pipeline output from 01 |
| `03_feature_extraction` | Biomarker features | Pipeline output from 01/02 |
| `04_feature_selection` | Variance + correlation + MI | Pipeline output from 03 |
| `05_classical_ml` | `run_benchmark()` | Pipeline output from 03/04 |
| `06_deep_learning` | **Stub** — no DL code | Pipeline output from 02 |
| `07_ablation` | Experiment variants | Pipeline output from 03+ |
| `08_final_benchmark` | Aggregate figures | Pipeline output from 05/07 |

## Kaggle setup (every notebook)

1. **Create notebook** on kaggle.com → upload or paste from `notebooks/kaggle/`.
2. **Settings → Internet → On** (clone repo + pip install).
3. **Settings → Accelerator → None** (CPU; GPU not needed for Phase 1).
4. **Add Data** → attach datasets (see config cell in each notebook).
5. Edit the config cell:
   - `RAW_EEG_INPUT` — slug of dataset with `EEG_data/dataset2/` and `dataset3/`
   - `PIPELINE_INPUT` — slug of prior notebook output (contains `data/` folder)
6. Run all cells.
7. **Save Version** with output → **Create Dataset** for the next notebook.

## How code gets onto Kaggle

Each notebook clones the repo into `/kaggle/working/ad_eeg`:

```text
git clone https://github.com/RandomPerson5571/ad_eeg.git /kaggle/working/ad_eeg
pip install -r requirements.txt
```

Raw EEG is symlinked from `/kaggle/input/<RAW_EEG_INPUT>/EEG_data`.  
Prior stages are copied from `/kaggle/input/<PIPELINE_INPUT>/data`.

## Chaining example

```text
00_audit     → Save as "eeg-audit-v1"
01_preprocess (RAW_EEG) → Save as "eeg-preprocessed-baseline-v1"
03_features  (PIPELINE_INPUT=eeg-preprocessed-baseline-v1) → Save as "eeg-features-v1"
05_ml        (PIPELINE_INPUT=eeg-features-v1) → Save as "eeg-benchmark-v1"
```

## Path aliases

| Spec alias | Actual path |
|------------|-------------|
| `artifacts/metadata.csv` | `data/audit/{dataset}/metadata.csv` |
| `cleaned/*.fif` | `data/preprocessed/{dataset}/{experiment}/*_clean.fif` |
| `epochs/*.npy` | `data/preprocessed/{dataset}/{experiment}/epochs/*.npy` |
| `features/features.parquet` | `data/features/{dataset}/{experiment}/subject_features.parquet` |
| `results/benchmark.csv` | `data/results/{dataset}/{experiment}/benchmark.csv` |

## Regenerate notebooks locally

```bash
python scripts/generate_kaggle_notebooks.py
```

This only regenerates `.ipynb` files — execution still happens on Kaggle.

## CLI equivalent (local or Kaggle)

```bash
python pipeline.py --dataset eyesclosed --experiment baseline --stages preprocess,features,train
python scripts/benchmark.py --dataset eyesclosed --experiment baseline
```

Legacy notebooks in `notebooks/colab/` are for Google Colab, not Kaggle.
