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
   - notebooks 00–01: set `RAW_EEG_INPUT` to the dataset with
     `EEG_data/dataset2/` and/or `dataset3/`
   - notebooks 02–08: set `PIPELINE_INPUT` to the preceding notebook's saved
     output dataset; leave `RAW_EEG_INPUT = None`
   - use the mounted dataset slug (for example `preprocessed-eeg-data`), not a
     Kaggle web path such as `datasets/owner/preprocessed-eeg-data`
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

The restore step accepts both the current `pipeline_output/data/preprocessed/...`
tree and the earlier direct preprocessing tree
`pipeline_output/{dataset}/{experiment}/*_epo.fif`. Legacy trees are mapped into
the downstream notebook's working view without modifying the attached dataset.

Notebook 03 is intentionally resumable and runs at most 24 not-yet-completed
subjects per version by default (about 4–5 hours at the observed Kaggle rate).
Each subject is committed under
`data/features/{dataset}/{experiment}/subject_partitions/` before the next subject
starts. After a `PARTIAL` result, save the output as a Dataset and use that newest
dataset as `PIPELINE_INPUT` for another notebook 03 run. Continue until it prints
`COMPLETE`; notebook 04 rejects a partial subject cohort.

## Chaining example

```text
00_audit       (RAW_EEG_INPUT=raw-eeg) → optional audit dataset
01_preprocess  (RAW_EEG_INPUT=raw-eeg) → eeg-preprocessed-v1
02_epoching    (PIPELINE_INPUT=eeg-preprocessed-v1) → eeg-epochs-v1
03_features    (PIPELINE_INPUT=eeg-epochs-v1) → eeg-features-v1
04_selection   (PIPELINE_INPUT=eeg-features-v1) → eeg-selected-v1
05_ml          (PIPELINE_INPUT=eeg-selected-v1) → eeg-benchmark-v1
08_final       (PIPELINE_INPUT=eeg-benchmark-v1) → final results
```

Notebook 01 stores a compact participant/label index inside its existing
`metadata.json`; it does not add or rearrange preprocessing output paths. This
index supplies labels to notebook 03, so raw EEG is not a hidden downstream
dependency. Notebook 02 enumerates the saved `*_epo.fif` files directly and
exports each one as a float32 array with shape
`(n_epochs, n_channels, n_samples)`.

For preprocessing artifacts created before `participant_index` was embedded in
`metadata.json`, the eyes-closed pipeline uses the authoritative OpenNeuro
ds004504 group ranges bundled in `configs/dataset.yaml`. This compatibility
fallback does not write to or alter the preprocessing artifact.

Every production notebook runs an input contract before its stage and an output
contract afterward. Contracts check required files, identity/label columns, and
array/table shapes so an incompatible artifact fails in the notebook that
produced or first consumes it.

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
