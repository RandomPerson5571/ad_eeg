#!/usr/bin/env python3
"""Generate Kaggle pipeline notebooks (00–08) — Kaggle cloud only."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle"
OUT.mkdir(parents=True, exist_ok=True)

KAGGLE_INTRO = """\
# Kaggle Notebook Pipeline

**Run on [kaggle.com](https://www.kaggle.com) only** — not on your laptop.

1. **Settings → Internet → On** (needed for `git clone` + `pip install`)
2. **Settings → Accelerator → None** (CPU is enough for Phase 1)
3. **Add Data** → attach input dataset(s) listed in the config cell below
4. Run all cells, then **Save Version** → **Save as Dataset** to pass `data/` to the next notebook
"""

KAGGLE_CONFIG = '''\
# --- Kaggle configuration (edit slugs to match your input datasets) ---
REPO_URL = "https://github.com/RandomPerson5571/ad_eeg.git"
REPO_BRANCH = "main"
PROJECT_DIR = "/kaggle/working/ad_eeg"

# Kaggle dataset slug with raw EEG (must contain EEG_data/dataset2/ and dataset3/)
RAW_EEG_INPUT = "REPLACE_WITH_RAW_EEG_DATASET_SLUG"

# Optional: output from a prior pipeline notebook (must contain data/ at root)
PIPELINE_INPUT = None  # e.g. "REPLACE_WITH_PRIOR_PIPELINE_OUTPUT_SLUG"
'''

KAGGLE_SETUP = '''\
import os
import shutil
import subprocess
import sys
from pathlib import Path

IS_KAGGLE = Path("/kaggle/input").exists()
if not IS_KAGGLE:
    raise RuntimeError(
        "This notebook runs on Kaggle only. "
        "Upload to kaggle.com, enable Internet, attach input datasets, then run."
    )

PROJECT_DIR = Path(PROJECT_DIR)


def run(cmd, cwd=None):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


if not PROJECT_DIR.exists():
    run(f"git clone --branch {REPO_BRANCH} --depth 1 {REPO_URL} {PROJECT_DIR}")

os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))
run(f"{sys.executable} -m pip install -q -r requirements.txt", cwd=PROJECT_DIR)
print(f"Project root: {PROJECT_DIR.resolve()}", flush=True)


def _find_eeg_root(slug: str) -> Path | None:
    base = Path("/kaggle/input") / slug
    if not base.exists():
        return None
    if (base / "EEG_data").is_dir():
        return base / "EEG_data"
    if (base / "dataset2").is_dir():
        return base
    for child in base.iterdir():
        if child.is_dir() and (child / "EEG_data").is_dir():
            return child / "EEG_data"
        if child.is_dir() and (child / "dataset2").is_dir():
            return child
    return None


eeg_link = PROJECT_DIR / "EEG_data"
if RAW_EEG_INPUT:
    src = _find_eeg_root(RAW_EEG_INPUT)
    if src is None:
        raise FileNotFoundError(
            f"Raw EEG not found for slug '{RAW_EEG_INPUT}'. "
            "Add Data → your dataset with EEG_data/dataset2/ and dataset3/."
        )
    if eeg_link.is_symlink():
        eeg_link.unlink()
    elif eeg_link.is_dir() and not eeg_link.is_symlink():
        pass
    elif eeg_link.exists():
        eeg_link.unlink()
    if not eeg_link.exists():
        os.symlink(src, eeg_link)
    print(f"EEG_data → {src}", flush=True)

if PIPELINE_INPUT:
    pipeline_src = Path("/kaggle/input") / PIPELINE_INPUT / "data"
    if not pipeline_src.exists():
        pipeline_src = Path("/kaggle/input") / PIPELINE_INPUT
        if not (pipeline_src / "preprocessed").exists() and not (pipeline_src / "audit").exists():
            raise FileNotFoundError(
                f"Pipeline input '{PIPELINE_INPUT}' has no data/ folder. "
                "Save the previous notebook version as a Dataset first."
            )
    dest = PROJECT_DIR / "data"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(pipeline_src, dest, dirs_exist_ok=True)
    print(f"Restored pipeline data from {pipeline_src}", flush=True)
'''

KAGGLE_SAVE = '''\
# Publish artifacts for the next notebook: Save Version → Output → Create Dataset
if Path("/kaggle/input").exists():
    out = Path("/kaggle/working/pipeline_output")
    src = Path("data")
    if src.exists():
        shutil.copytree(src, out / "data", dirs_exist_ok=True)
        print(f"Output ready: {out}")
        print("Save Version → Save output as new Kaggle Dataset, then attach in the next notebook.")
'''

CONFIG_CELL = '''\
from eeg.config import load_experiment, resolve_dataset
from eeg.repro import init_repro, snapshot_environment

EXPERIMENT = "baseline"
dataset_spec = resolve_dataset("eyesclosed")[0]
config = load_experiment(EXPERIMENT)

CONFIG = {
    "dataset": dataset_spec.name,
    "experiment": EXPERIMENT,
    "seed": config.get("training", {}).get("random_state", 42),
    "cv_folds": config.get("training", {}).get("cv_folds", 5),
    "feature_set": "full",
    "normalization": "zscore",
}
repro = init_repro(CONFIG["seed"])
env = snapshot_environment()
print(CONFIG)
'''

# Per-notebook input hints (injected into markdown)
INPUT_HINTS = {
    "00": "**Inputs:** `RAW_EEG_INPUT` only.",
    "01": "**Inputs:** `RAW_EEG_INPUT` only.",
    "02": "**Inputs:** `PIPELINE_INPUT` = output from notebook 01 (preprocessed checkpoints).",
    "03": "**Inputs:** `PIPELINE_INPUT` = output from notebook 01 or 02.",
    "04": "**Inputs:** `PIPELINE_INPUT` = output from notebook 03 (features).",
    "05": "**Inputs:** `PIPELINE_INPUT` = output from notebook 04 (features + selection optional).",
    "06": "**Inputs:** `PIPELINE_INPUT` = output from notebook 02 (epoch `.npy`).",
    "07": "**Inputs:** `PIPELINE_INPUT` = output from notebook 03+ (features); re-run preprocessing per ablation experiment separately.",
    "08": "**Inputs:** `PIPELINE_INPUT` = output from notebook 05/07 (benchmark results).",
}


def _nb_cells(nb_key: str, title_md: str, body_cells: list, save: bool = True):
    cells = [
        ("markdown", KAGGLE_INTRO + "\n" + INPUT_HINTS.get(nb_key, "")),
        ("code", KAGGLE_CONFIG),
        ("code", KAGGLE_SETUP),
    ]
    cells.extend(body_cells)
    if save:
        cells.append(("code", KAGGLE_SAVE))
    return cells


NOTEBOOKS = {
    "00_dataset_audit.ipynb": _nb_cells(
        "00",
        "",
        [
            ("markdown", "# 00 — Dataset Audit\n\nPaper-ready cohort statistics from raw EEG (no preprocessing)."),
            ("code", CONFIG_CELL),
            ("code", '''\
from eeg.audit import audit_dataset, write_audit_artifacts
from eeg.config import resolve_dataset

for spec in resolve_dataset("all"):
    result = audit_dataset(spec)
    paths = write_audit_artifacts(result, spec.name, environment=env)
    print(spec.name, result.dataset_summary)
    display(result.patient_summary.head())
'''),
        ],
    ),
    "01_preprocessing.ipynb": _nb_cells(
        "01",
        "",
        [
            ("markdown", "# 01 — Preprocessing\n\nRuns batch preprocessing on Kaggle CPUs."),
            ("code", CONFIG_CELL),
            ("code", '''\
from scripts.preprocess_dataset import run_preprocess

run_preprocess(CONFIG["dataset"], CONFIG["experiment"], workers=2, limit=None)
# Backlog: raise h_freq to 45 Hz in experiments/baseline.yaml
'''),
        ],
    ),
    "02_epoching.ipynb": _nb_cells(
        "02",
        "",
        [
            ("markdown", "# 02 — Epoching\n\nExport `*_epo.fif` checkpoints to `.npy` for future DL."),
            ("code", CONFIG_CELL),
            ("code", '''\
from eeg.export import export_all_epochs_npy
import numpy as np

paths = export_all_epochs_npy(CONFIG["dataset"], CONFIG["experiment"])
print(f"Exported {len(paths)} subjects")
if paths:
    arr = np.load(paths[0])
    print("shape:", arr.shape, "dtype:", arr.dtype)
'''),
        ],
    ),
    "03_feature_extraction.ipynb": _nb_cells(
        "03",
        "",
        [
            ("markdown", "# 03 — Feature Extraction\n\nOne section per `biomarkers/` module."),
            ("code", CONFIG_CELL),
            ("code", '''\
from scripts.extract_features import run_extract
run_extract(CONFIG["dataset"], CONFIG["experiment"], workers=2)
'''),
            ("code", '''\
from biomarkers import (
    compute_band_power,
    compute_connectivity,
    compute_regional_complexity,
)
print("spectral:", compute_band_power)
print("connectivity:", compute_connectivity)
# TODO: graph, entropy, time_domain — Phase 2
'''),
        ],
    ),
    "04_feature_selection.ipynb": _nb_cells(
        "04",
        "",
        [
            ("markdown", "# 04 — Feature Selection\n\nVariance → correlation → mutual information."),
            ("code", CONFIG_CELL),
            ("code", '''\
from eeg.config import resolve_dataset
from eeg.feature_selection import select_features, save_selection_artifacts
from eeg.io import load_features_df
from eeg.training.datasets import feature_columns

spec = resolve_dataset(CONFIG["dataset"])[0]
df = load_features_df(CONFIG["dataset"], CONFIG["experiment"], spec.id)
cols = feature_columns()
result = select_features(df, cols, config=CONFIG)
paths = save_selection_artifacts(df, result, CONFIG["dataset"], CONFIG["experiment"])
print(f"Selected {result.n_selected}/{result.n_input} features")
display(result.importance.head(20))
'''),
        ],
    ),
    "05_classical_ml.ipynb": _nb_cells(
        "05",
        "",
        [
            ("markdown", "# 05 — Classical ML\n\nAll evaluation via `run_benchmark()` only."),
            ("code", CONFIG_CELL),
            ("code", '''\
from eeg.feature_selection import select_features, save_selection_artifacts
from eeg.io import load_features_df
from eeg.config import resolve_dataset
from eeg.training.datasets import feature_columns
from eeg.training.benchmark import run_benchmark

spec = resolve_dataset(CONFIG["dataset"])[0]
df = load_features_df(CONFIG["dataset"], CONFIG["experiment"], spec.id)
sel = select_features(df, feature_columns(), config=CONFIG)
save_selection_artifacts(df, sel, CONFIG["dataset"], CONFIG["experiment"])

result = run_benchmark(
    dataset=CONFIG["dataset"],
    experiment=CONFIG["experiment"],
    models=["logistic_regression", "random_forest", "xgboost", "mlp"],
    feature_cols=sel.selected_columns,
    config=CONFIG,
)
import pandas as pd
display(pd.read_csv(result["benchmark_csv"]))
'''),
        ],
    ),
    "06_deep_learning.ipynb": _nb_cells(
        "06",
        "",
        [
            ("markdown", """# 06 — Deep Learning (stub)

**Phase 1: no DL training code.**

## Input contract
- `data/preprocessed/{dataset}/{experiment}/epochs/sub-XXX.npy`
- Shape: `(n_epochs, n_channels, n_samples)` float32

## Planned models (backlog)
- EEGNet, ChronoNet, DeepConvNet, ShallowConvNet, Transformer
"""),
            ("code", CONFIG_CELL),
            ("code", '''\
from eeg.paths import epochs_npy_dir
print("Epoch npy dir:", epochs_npy_dir(CONFIG["dataset"], CONFIG["experiment"]))
'''),
        ],
        save=False,
    ),
    "07_ablation.ipynb": _nb_cells(
        "07",
        "",
        [
            ("markdown", "# 07 — Ablation\n\nCompare preprocessing experiment variants via `run_benchmark()`."),
            ("code", CONFIG_CELL),
            ("code", '''\
from eeg.training.benchmark import run_benchmark
import pandas as pd

ablations = ["baseline", "fast", "ica95", "no_asr", "no_ref", "no_overlap"]
frames = []
for exp in ablations:
    CONFIG["experiment"] = exp
    try:
        r = run_benchmark(
            dataset=CONFIG["dataset"],
            experiment=exp,
            models=["random_forest"],
            config=CONFIG,
        )
        df = pd.read_csv(r["benchmark_csv"])
        df["ablation"] = exp
        frames.append(df)
    except Exception as e:
        print(f"Skip {exp}: {e}")

if frames:
    display(pd.concat(frames, ignore_index=True))
'''),
        ],
    ),
    "08_final_benchmark.ipynb": _nb_cells(
        "08",
        "",
        [
            ("markdown", "# 08 — Final Benchmark\n\nAggregate benchmark artifacts and publication figures."),
            ("code", CONFIG_CELL),
            ("code", '''\
from pathlib import Path
import pandas as pd
from eeg.paths import results_dir, figures_dir
from eeg.training.evaluation import plot_runtime_bar

res = results_dir(CONFIG["dataset"], CONFIG["experiment"])
fig = figures_dir(CONFIG["dataset"], CONFIG["experiment"])

csv_files = list(Path("data/results").rglob("benchmark.csv"))
frames = []
for p in csv_files:
    df = pd.read_csv(p)
    df["source"] = str(p.parent)
    frames.append(df)

if frames:
    agg = pd.concat(frames, ignore_index=True)
    out = res / "benchmark_results.csv"
    agg.to_csv(out, index=False)
    display(agg)

    if (res / "benchmark.csv").exists():
        bench = pd.read_csv(res / "benchmark.csv")
        plot_runtime_bar(
            dict(zip(bench["model"], bench["train_time_s"])),
            fig / "runtime_train.png",
            title="Training time",
        )
'''),
        ],
    ),
}


def make_notebook(cells_spec):
    cells = []
    for kind, source in cells_spec:
        cells.append(
            {
                "cell_type": kind,
                "metadata": {},
                "source": [line + "\n" for line in source.splitlines()],
            }
        )
        if kind == "code":
            cells[-1]["outputs"] = []
            cells[-1]["execution_count"] = None
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "kaggle": {
                "accelerator": "none",
                "isInternetRequired": True,
                "isGpuEnabled": False,
            },
        },
        "cells": cells,
    }


for name, spec in NOTEBOOKS.items():
    path = OUT / name
    path.write_text(json.dumps(make_notebook(spec), indent=1), encoding="utf-8")
    print("wrote", path)
