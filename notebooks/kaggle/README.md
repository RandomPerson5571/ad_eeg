# Kaggle Notebook Pipeline

**These notebooks run on [Kaggle](https://www.kaggle.com) cloud CPUs — not on your laptop.**

Upload each notebook to Kaggle, enable **Internet**, attach input datasets, run, then **Save Version → Create Dataset** to chain stages. Production artifacts are written directly to `/kaggle/working/pipeline_output/data`; there is no publish-time copy.

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
   - `PIPELINE_INPUT` — slug of prior notebook output (accepts `pipeline_output/data/` or `data/`)
6. Run all cells.
7. **Save Version** with output → **Create Dataset** for the next notebook.

## How code gets onto Kaggle

Each notebook clones the repo into temporary storage at `/kaggle/temp/ad_eeg`:

```text
git clone https://github.com/RandomPerson5571/ad_eeg.git /kaggle/temp/ad_eeg
pip install -r requirements-kaggle.txt
```

Raw EEG is symlinked from `/kaggle/input/<RAW_EEG_INPUT>/EEG_data`.  
The repo's `data/` path is symlinked to `/kaggle/working/pipeline_output/data`, so new files land directly in the saved artifact tree. Prior stages are copied once from the attached read-only dataset into that tree; they are not copied again at the end.

Only files under `/kaggle/working` are included when Kaggle saves notebook output. Keeping the clone under `/kaggle/temp` prevents the GitHub repository (including `.git`) from leaking into the final dataset.

## Preprocessing storage policy

Notebook 01 defaults to `KEEP_INTERMEDIATE_CHECKPOINTS = False`. After a subject has a valid epochs checkpoint and its QC summary has been generated, the large `raw`, `filtered`, `ica`, and `clean` FIF checkpoints are deleted. Epochs, logs, QC reports, and failed-subject resume checkpoints remain available. Set the option to `True` only when you need every resumable checkpoint and have enough Kaggle storage.

In `inspect` and `test` modes, preprocessing checkpoints stay under `/kaggle/temp`; only the smaller summaries and QC plots in `/kaggle/working/test_output` are saved. `full` mode is the only mode that writes production data to `pipeline_output/data`.

Before restoring an upstream dataset, the setup cell checks available space and reserves 512 MiB. Every production stage reports current artifact size and free `/kaggle/working` capacity.

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
