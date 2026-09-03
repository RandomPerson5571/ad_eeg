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
2. **Settings → Accelerator → None**, except notebook 06 where a GPU is recommended
3. **Add Data** → attach input dataset(s) listed in the config cell below
4. Run all cells, then publish the compact `pipeline_output/data/` artifact:
   create the notebook's output Dataset once; on later resumable runs, publish a
   new version of that same Dataset instead of trying to create another Dataset

The repository is cloned into temporary storage and is never copied into the saved
notebook output. Pipeline stages write directly to the final output directory, avoiding
a second full-size copy at publish time.
"""

KAGGLE_CONFIG = '''\
# --- Kaggle configuration (edit slugs to match your input datasets) ---
REPO_URL = "https://github.com/RandomPerson5571/ad_eeg.git"
REPO_BRANCH = "main"
PROJECT_DIR = "/kaggle/temp/ad_eeg"  # temporary; excluded from saved notebook output
OUTPUT_DIR = "/kaggle/working/pipeline_output"  # the only production artifact root

# Raw EEG is used only by notebooks 00 and 01.
RAW_EEG_INPUT = {raw_eeg_input}

# Prior notebook output is used by notebooks 02-08.
PIPELINE_INPUT = {pipeline_input}

# Generated stage contract; do not edit.
NOTEBOOK_STAGE = "{stage}"
REQUIRES_RAW_EEG = {requires_raw}
REQUIRES_PIPELINE_INPUT = {requires_pipeline}
'''

KAGGLE_SETUP = '''\
import os
import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_INPUT_DIR = Path("/kaggle/input")
IS_KAGGLE = KAGGLE_INPUT_DIR.exists()
if not IS_KAGGLE:
    raise RuntimeError(
        "This notebook runs on Kaggle only. "
        "Upload to kaggle.com, enable Internet, attach input datasets, then run."
    )

PROJECT_DIR = Path(PROJECT_DIR)
OUTPUT_DIR = Path(OUTPUT_DIR)
# Notebook 01 defines MODE before setup. Its inspect/test checkpoints are scratch
# data; only their small reports under /kaggle/working/test_output are persisted.
if globals().get("MODE") in {"inspect", "test"}:
    OUTPUT_DIR = Path("/kaggle/temp/pipeline_output")
OUTPUT_DATA_DIR = OUTPUT_DIR / "data"
KAGGLE_WORKING_DIR = Path("/kaggle/working")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


if _is_relative_to(PROJECT_DIR, KAGGLE_WORKING_DIR):
    raise ValueError(
        "PROJECT_DIR must be outside /kaggle/working so the Git clone is not "
        "included in the saved Kaggle output. Use /kaggle/temp/ad_eeg."
    )


def run(cmd, cwd=None):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


if not PROJECT_DIR.exists():
    PROJECT_DIR.parent.mkdir(parents=True, exist_ok=True)
    run(f"git clone --branch {REPO_BRANCH} --depth 1 {REPO_URL} {PROJECT_DIR}")

# Point the code's data/ path at the one-and-only persisted artifact tree.
# Preserve the small tracked seed files (for example data/manifest.json) first.
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
project_data = PROJECT_DIR / "data"
if project_data.is_symlink():
    if project_data.resolve() != OUTPUT_DATA_DIR.resolve():
        project_data.unlink()
elif project_data.exists():
    shutil.copytree(project_data, OUTPUT_DATA_DIR, dirs_exist_ok=True)
    shutil.rmtree(project_data)
if not project_data.exists():
    os.symlink(OUTPUT_DATA_DIR, project_data, target_is_directory=True)

os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))
run(f"{sys.executable} -m pip install -q -r requirements-kaggle.txt", cwd=PROJECT_DIR)
print(f"Project root: {PROJECT_DIR.resolve()}", flush=True)
print(f"Pipeline output: {OUTPUT_DIR.resolve()}", flush=True)


def _tree_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _gib(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def _print_storage(label: str) -> None:
    usage = shutil.disk_usage(KAGGLE_WORKING_DIR)
    output_bytes = _tree_size(OUTPUT_DIR) if OUTPUT_DIR.exists() else 0
    print(
        f"{label}: output={_gib(output_bytes):.2f} GiB, "
        f"working free={_gib(usage.free):.2f} GiB",
        flush=True,
    )


def _is_configured(value) -> bool:
    return bool(value) and not str(value).startswith("REPLACE_WITH_")


def _input_mount_candidates(locator: str) -> list[Path]:
    """Accept a Kaggle slug, mounted path, or copied datasets/<owner>/<slug> path."""
    value = str(locator).strip().rstrip("/")
    supplied = Path(value)
    candidates = [supplied] if supplied.is_absolute() else [KAGGLE_INPUT_DIR / supplied]
    # Kaggle web paths are often copied as datasets/<owner>/<slug>, but the
    # notebook mount uses only /kaggle/input/<slug>.
    slug = supplied.name
    mounted = KAGGLE_INPUT_DIR / slug
    if mounted not in candidates:
        candidates.append(mounted)

    # Also support owner/slug-style and renamed mounts. Kaggle's Input panel is
    # the source of truth, so only add paths that actually exist there.
    if KAGGLE_INPUT_DIR.is_dir():
        for pattern in (f"*/{slug}", f"*/*/{slug}"):
            for match in KAGGLE_INPUT_DIR.glob(pattern):
                if match.is_dir() and match not in candidates:
                    candidates.append(match)

        normalized_slug = slug.lower().replace("_", "-")
        available_mounts = [
            path for path in KAGGLE_INPUT_DIR.iterdir() if path.is_dir()
        ]
        fuzzy_mounts = [
            path
            for path in available_mounts
            if normalized_slug in path.name.lower().replace("_", "-")
            or path.name.lower().replace("_", "-") in normalized_slug
        ]
        if not fuzzy_mounts and len(available_mounts) == 1:
            fuzzy_mounts = available_mounts
        for match in fuzzy_mounts:
            if match not in candidates:
                candidates.append(match)
    return candidates


def _pipeline_data_source(locator: str) -> tuple[Path, str]:
    stage_dirs = {"audit", "preprocessed", "features", "models", "results"}
    dataset_names = {"eyesclosed", "photomark", "dataset2", "dataset3"}
    checked = []
    mounts = _input_mount_candidates(locator)
    for base in mounts:
        for candidate in (
            base / "pipeline_output" / "data",
            base / "data",
            base / "pipeline_output",
            base,
        ):
            checked.append(candidate)
            if candidate.is_dir() and any((candidate / name).exists() for name in stage_dirs):
                return candidate, "data_tree"
            # Older preprocessing outputs used
            # pipeline_output/{dataset}/{experiment}/*_epo.fif directly.
            if candidate.is_dir():
                dataset_dirs = [
                    candidate / name
                    for name in dataset_names
                    if (candidate / name).is_dir()
                ]
                if any(next(ds.rglob("*_epo.fif"), None) for ds in dataset_dirs):
                    return candidate, "preprocessed_tree"

        # Accept an extra wrapper directory introduced while creating a Kaggle
        # Dataset, while still anchoring the layout at a known dataset name.
        if base.is_dir():
            epoch_file = next(base.rglob("*_epo.fif"), None)
            if epoch_file is not None:
                for parent in epoch_file.parents:
                    if parent.name in dataset_names:
                        return parent.parent, "preprocessed_tree"

    mounted_contents = []
    for base in mounts:
        if base.is_dir():
            mounted_contents.extend(
                str(child.relative_to(base))
                for child in sorted(base.iterdir())[:20]
            )
    available_mounts = (
        sorted(path.name for path in KAGGLE_INPUT_DIR.iterdir() if path.is_dir())
        if KAGGLE_INPUT_DIR.is_dir()
        else []
    )
    if not any(path.is_dir() for path in mounts):
        raise FileNotFoundError(
            f"No Kaggle input mount matches PIPELINE_INPUT={locator!r}. "
            f"Available mounts: {available_mounts or ['<none>']}. Open the notebook's "
            "Input pane, choose Add Input, attach the preprocessing dataset, then "
            "restart the session."
        )
    raise FileNotFoundError(
        f"Pipeline input '{locator}' has no recognized preprocessing or pipeline tree. "
        f"Checked: {[str(path) for path in checked]}. Attach the previous notebook's "
        "saved output dataset and use its mounted slug in PIPELINE_INPUT. "
        f"Available mounts: {available_mounts or ['<none>']}. "
        f"Mounted top-level contents: {mounted_contents or ['<empty>']}"
    )


def _restore_pipeline_data(slug: str) -> None:
    pipeline_src, layout = _pipeline_data_source(slug)
    source_bytes = _tree_size(pipeline_src)
    free_bytes = shutil.disk_usage(KAGGLE_WORKING_DIR).free
    reserve_bytes = 512 * 1024 ** 2
    if source_bytes + reserve_bytes > free_bytes:
        raise OSError(
            f"Pipeline input needs about {_gib(source_bytes):.2f} GiB but only "
            f"{_gib(free_bytes):.2f} GiB is free in /kaggle/working. "
            "Use a compact upstream artifact or start a fresh Kaggle session."
        )
    if layout == "data_tree":
        shutil.copytree(pipeline_src, OUTPUT_DATA_DIR, dirs_exist_ok=True)
    else:
        aliases = {
            "eyesclosed": "eyesclosed",
            "dataset2": "eyesclosed",
            "photomark": "photomark",
            "dataset3": "photomark",
        }
        restored = 0
        for source_name, canonical_name in aliases.items():
            source_dataset = pipeline_src / source_name
            if not source_dataset.is_dir():
                continue
            shutil.copytree(
                source_dataset,
                OUTPUT_DATA_DIR / "preprocessed" / canonical_name,
                dirs_exist_ok=True,
            )
            restored += 1
        if not restored:
            raise FileNotFoundError(
                f"Found preprocessing files in {pipeline_src}, but no supported "
                "dataset directory (eyesclosed/photomark/dataset2/dataset3)."
            )
    print(
        f"Restored pipeline data from {pipeline_src} (layout={layout})",
        flush=True,
    )
    _print_storage("After restore")


def summarize_output() -> None:
    if not OUTPUT_DATA_DIR.exists():
        print("No pipeline data was produced.", flush=True)
        return
    leaked_repos = [p for p in OUTPUT_DIR.rglob(".git") if p.is_dir()]
    if leaked_repos:
        raise RuntimeError(f"Refusing to publish a Git repository: {leaked_repos[0]}")
    n_files = sum(1 for p in OUTPUT_DATA_DIR.rglob("*") if p.is_file())
    _print_storage("Final artifact")
    print(f"Output ready: {OUTPUT_DIR} ({n_files} files)", flush=True)
    print(
        "Save Version, then create the output Dataset if this is the first run "
        "or publish a new version of the same Dataset on later runs. Attach the "
        "latest Dataset version before the next run or notebook.",
        flush=True,
    )


def _find_eeg_root(locator: str) -> Path | None:
    for base in _input_mount_candidates(locator):
        if not base.exists():
            continue
        if (base / "EEG_data").is_dir():
            return base / "EEG_data"
        if (base / "dataset2").is_dir() or (base / "dataset3").is_dir():
            return base
        for child in base.iterdir():
            if child.is_dir() and (child / "EEG_data").is_dir():
                return child / "EEG_data"
            if child.is_dir() and (
                (child / "dataset2").is_dir() or (child / "dataset3").is_dir()
            ):
                return child
    return None


eeg_link = PROJECT_DIR / "EEG_data"
if REQUIRES_PIPELINE_INPUT and not _is_configured(PIPELINE_INPUT):
    if _is_configured(RAW_EEG_INPUT):
        raise ValueError(
            f"Notebook {NOTEBOOK_STAGE} consumes a prior pipeline artifact, not raw EEG. "
            "Move the supplied value from RAW_EEG_INPUT to PIPELINE_INPUT."
        )
    raise ValueError(
        f"Notebook {NOTEBOOK_STAGE} requires PIPELINE_INPUT. Attach the preceding "
        "notebook's saved output dataset and enter its Kaggle slug."
    )

if REQUIRES_RAW_EEG and not _is_configured(RAW_EEG_INPUT):
    raise ValueError(
        f"Notebook {NOTEBOOK_STAGE} requires RAW_EEG_INPUT with EEG_data/dataset2/ "
        "and/or dataset3/."
    )

if _is_configured(RAW_EEG_INPUT):
    src = _find_eeg_root(RAW_EEG_INPUT)
    if src is None:
        raise FileNotFoundError(
            f"Raw EEG not found for slug '{RAW_EEG_INPUT}'. "
            "Add Data → your dataset with EEG_data/dataset2/ and/or dataset3/."
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

if _is_configured(PIPELINE_INPUT):
    _restore_pipeline_data(PIPELINE_INPUT)
'''

KAGGLE_SAVE = '''\
# Artifacts already live in their final location; no large publish-time copy is needed.
if Path("/kaggle/input").exists():
    summarize_output()
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

PREPROCESS_PIPELINE_CONFIG = '''\
# --- Pipeline configuration (edit before running) ---
from pathlib import Path

MODE = "test"          # "inspect" | "test" | "full"
DATASET = "dataset2"   # "dataset2" | "dataset3" | "all"
EXPERIMENT = "baseline"
FORCE = False
WORKERS = 2
TEST_SUBJECTS = 5      # used when MODE == "test"
INSPECT_SUBJECT = 1    # subject number when MODE == "inspect"
KEEP_INTERMEDIATE_CHECKPOINTS = False  # False keeps epochs + QC/logs and saves substantial space

VALID_MODES = {"inspect", "test", "full"}
if MODE not in VALID_MODES:
    raise ValueError(f"MODE must be one of {VALID_MODES}, got {MODE!r}")

TEST_OUTPUT = Path("/kaggle/working/test_output")
'''

PREPROCESS_LOAD_CONFIG = '''\
from eeg.config import load_experiment, resolve_dataset
from eeg.repro import init_repro, snapshot_environment

dataset_specs = resolve_dataset(DATASET)
config = load_experiment(EXPERIMENT)

CONFIG = {
    "mode": MODE,
    "dataset": DATASET,
    "datasets": [ds.name for ds in dataset_specs],
    "experiment": EXPERIMENT,
    "force": FORCE,
    "workers": WORKERS,
    "test_subjects": TEST_SUBJECTS,
    "inspect_subject": INSPECT_SUBJECT,
    "seed": config.get("training", {}).get("random_state", 42),
}
repro = init_repro(CONFIG["seed"])
env = snapshot_environment()
print(CONFIG)
'''

PREPROCESS_RUN_CELL = '''\
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from eeg.cli import subject_num_from_id
from eeg.config import experiment_metadata
from eeg.io import write_json
from eeg.paths import STAGE_SUFFIX, preprocessed_dir, qc_report_dir
from eeg.preprocess_report import write_preprocess_report
from eeg.qc import BAND_RANGES, backfill_spectral_qc, preprocessing_metrics
from eeg.runner import summarize_batch
from eeg.visualization import plot_preprocessing_panels_from_checkpoints
from scripts.preprocess_dataset import run_preprocess

QC_PLOTS = False  # set True for per-subject PNGs in full/test mode


def _subject_nums(ds):
    if MODE == "inspect":
        return [INSPECT_SUBJECT]
    return list(range(1, TEST_SUBJECTS + 1))


def _spectral_summary(metrics: dict) -> dict:
    """Extract µV²/Hz band-power fields for notebook summaries."""
    out = {}
    for band in BAND_RANGES:
        if metrics.get(f"{band}_before_uv2") is not None:
            out[f"{band}_before_uv2"] = metrics[f"{band}_before_uv2"]
        if metrics.get(f"{band}_delta_uv2") is not None:
            out[f"{band}_delta_uv2"] = metrics[f"{band}_delta_uv2"]
    return out


def _refresh_dataset_qc(ds, limit=None):
    """Backfill spectral QC from checkpoints and regenerate summary.csv."""
    patched = backfill_spectral_qc(ds.name, EXPERIMENT, limit=limit)
    report_paths = write_preprocess_report(
        ds.name, EXPERIMENT, config=config, qc_plots=QC_PLOTS, dataset_spec=ds
    )
    if patched:
        print(f"  [{ds.name}] backfilled spectral QC for {len(patched)} subject(s)")
    print(f"  [{ds.name}] QC report → {report_paths['summary_csv']}")
    return report_paths


def _compact_dataset_checkpoints(ds):
    """Remove resumable intermediates only after a valid epochs file exists."""
    root = preprocessed_dir(ds.name, EXPERIMENT)
    removed_files = 0
    removed_bytes = 0
    epoch_suffix = STAGE_SUFFIX["epochs"]
    for epoch_path in root.glob(f"*{epoch_suffix}"):
        participant_id = epoch_path.name[: -len(epoch_suffix)]
        for stage in ("raw", "filtered", "ica", "clean"):
            checkpoint = root / f"{participant_id}{STAGE_SUFFIX[stage]}"
            if checkpoint.is_file():
                removed_bytes += checkpoint.stat().st_size
                checkpoint.unlink()
                removed_files += 1
    print(
        f"  [{ds.name}] compacted {removed_files} intermediate checkpoints "
        f"({_gib(removed_bytes):.2f} GiB); epochs, QC, and logs retained"
    )


def _qc_subject(ds, subject_num, out_dir):
    metrics = preprocessing_metrics(ds, subject_num, EXPERIMENT)
    plot_path = plot_preprocessing_panels_from_checkpoints(
        ds, subject_num, EXPERIMENT, out_dir
    )
    alpha = metrics.get("alpha_before_uv2")
    alpha_d = metrics.get("alpha_delta_uv2")
    spectral = (
        f" alpha={alpha:.2f}µV² Δ={alpha_d:+.2f}"
        if alpha is not None and alpha_d is not None
        else ""
    )
    print(
        f"  {metrics['participant_id']}: bad_ch={metrics['n_bad_channels']} "
        f"rejected={metrics['n_epochs_rejected']}/{metrics['n_epochs_before_ar']}"
        f"{spectral} → {plot_path.name}"
    )
    return metrics, plot_path


summary = {
    "mode": MODE,
    "dataset": DATASET,
    "experiment": EXPERIMENT,
    "force": FORCE,
    "keep_intermediate_checkpoints": KEEP_INTERMEDIATE_CHECKPOINTS,
    "started_at": datetime.now(timezone.utc).isoformat(),
}
t0 = time.perf_counter()

if MODE == "inspect":
    out_root = TEST_OUTPUT / "inspect"
    out_root.mkdir(parents=True, exist_ok=True)
    summary["subjects"] = []

    for ds in dataset_specs:
        ds_out = out_root / ds.name
        ds_out.mkdir(parents=True, exist_ok=True)
        print(f"\\n[{ds.name}] preprocess + inspect subject {INSPECT_SUBJECT}")
        run_preprocess(
            ds.name,
            EXPERIMENT,
            workers=1,
            force=FORCE,
            limit=None,
            subject=f"sub-{INSPECT_SUBJECT:03d}",
            qc_plots=True,
        )
        _refresh_dataset_qc(ds, limit=INSPECT_SUBJECT)
        metrics, _ = _qc_subject(ds, INSPECT_SUBJECT, ds_out)
        summary["subjects"].append({**metrics, **_spectral_summary(metrics)})
        summary["qc_report"] = str(qc_report_dir(ds.name, EXPERIMENT) / "summary.csv")

elif MODE == "test":
    out_root = TEST_OUTPUT / "test"
    out_root.mkdir(parents=True, exist_ok=True)
    summary["datasets"] = {}

    for ds in dataset_specs:
        ds_out = out_root / ds.name
        ds_out.mkdir(parents=True, exist_ok=True)
        print(f"\\n[{ds.name}] preprocessing first {TEST_SUBJECTS} subjects...")
        results = run_preprocess(
            ds.name, EXPERIMENT, workers=WORKERS, force=FORCE, limit=TEST_SUBJECTS, qc_plots=QC_PLOTS
        )
        report_paths = _refresh_dataset_qc(ds, limit=TEST_SUBJECTS)
        batch = summarize_batch(results)
        summary["datasets"][ds.name] = {
            "completed": batch.completed,
            "skipped": batch.skipped,
            "failed": batch.failed,
            "qc_report": str(report_paths["summary_csv"]),
            "subjects": [
                {
                    "participant_id": r.log.get("participant_id"),
                    "status": r.status,
                    "runtime_seconds": r.log.get("runtime_seconds"),
                    "n_bad_channels": len(r.log.get("bad_channels", [])),
                    "n_epochs_rejected": r.log.get("n_epochs_rejected"),
                    **_spectral_summary(
                        preprocessing_metrics(
                            ds,
                            subject_num_from_id(r.log.get("participant_id")),
                            EXPERIMENT,
                        )
                    ),
                }
                for r in results
            ],
        }
        print(
            f"[{ds.name}] completed={batch.completed} "
            f"skipped={batch.skipped} failed={batch.failed}"
        )

        print(f"[{ds.name}] QC plots...")
        for sn in _subject_nums(ds):
            _qc_subject(ds, sn, ds_out)

elif MODE == "full":
    summary["datasets"] = {}
    all_results = []

    for ds in dataset_specs:
        print(f"\\n[{ds.name}] full preprocessing run...")
        results = run_preprocess(
            ds.name, EXPERIMENT, workers=WORKERS, force=FORCE, limit=None, qc_plots=QC_PLOTS
        )
        report_paths = _refresh_dataset_qc(ds)
        batch = summarize_batch(results)
        all_results.extend(results)
        summary["datasets"][ds.name] = {
            "completed": batch.completed,
            "skipped": batch.skipped,
            "failed": batch.failed,
            "n_subjects": len(results),
            "qc_report": str(report_paths["summary_csv"]),
        }
        print(
            f"[{ds.name}] completed={batch.completed} "
            f"skipped={batch.skipped} failed={batch.failed}"
        )
        if not KEEP_INTERMEDIATE_CHECKPOINTS:
            _compact_dataset_checkpoints(ds)
        _print_storage(f"After {ds.name}")

    runtimes = [r.log.get("runtime_seconds", 0) for r in all_results if r.log.get("runtime_seconds")]
    summary["mean_runtime_seconds"] = round(sum(runtimes) / len(runtimes), 2) if runtimes else None
    summary["config"] = experiment_metadata(
        DATASET, EXPERIMENT, config, n_processed=len(all_results)
    )

else:
    raise ValueError(f"Unknown MODE: {MODE}")

summary["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
summary["finished_at"] = datetime.now(timezone.utc).isoformat()

if MODE == "full":
    meta_path = Path("data") / "preprocess_full_summary.json"
else:
    meta_path = TEST_OUTPUT / f"preprocess_{MODE}_summary.json"
meta_path.parent.mkdir(parents=True, exist_ok=True)
write_json(meta_path, summary)
print(f"\\nSummary → {meta_path}")
print(json.dumps(summary, indent=2, default=str))
'''

PREPROCESS_SAVE = '''\
# Full-mode artifacts were written directly to the final output directory.
if MODE == "full" and Path("/kaggle/input").exists():
    from eeg.contracts import validate_preprocessed_artifacts

    for dataset_spec in dataset_specs:
        contract = validate_preprocessed_artifacts(dataset_spec.name, EXPERIMENT)
        print(f"Output contract ({dataset_spec.name}):", contract)
    summarize_output()
elif MODE != "full":
    print(f"MODE={MODE!r}: skipping Kaggle dataset publish (use MODE='full' for production output).")
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


RAW_INPUT_NOTEBOOKS = {"00", "01"}


def _kaggle_config(nb_key: str) -> str:
    requires_raw = nb_key in RAW_INPUT_NOTEBOOKS
    requires_pipeline = not requires_raw
    raw_value = '"REPLACE_WITH_RAW_EEG_DATASET_SLUG"' if requires_raw else "None"
    pipeline_value = (
        "None" if requires_raw else '"REPLACE_WITH_PRIOR_PIPELINE_OUTPUT_SLUG"'
    )
    return KAGGLE_CONFIG.format(
        raw_eeg_input=raw_value,
        pipeline_input=pipeline_value,
        stage=nb_key,
        requires_raw=requires_raw,
        requires_pipeline=requires_pipeline,
    )


def _nb_cells(
    nb_key: str,
    title_md: str,
    body_cells: list,
    save: bool = True,
    pre_setup_cells=None,
):
    cells = [
        ("markdown", KAGGLE_INTRO + "\n" + INPUT_HINTS.get(nb_key, "")),
        ("code", _kaggle_config(nb_key)),
    ]
    cells.extend(pre_setup_cells or [])
    cells.append(("code", KAGGLE_SETUP))
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
            (
                "markdown",
                "# 01 — Preprocessing\n\n"
                "Mode-based preprocessing pipeline for fast iteration and production runs.\n\n"
                "| Mode | Purpose | Output |\n"
                "|------|---------|--------|\n"
                "| **inspect** | One subject, interactive QC plots, debugging | "
                "`/kaggle/working/test_output/inspect/` |\n"
                "| **test** | First N subjects, validation metrics, regression check | "
                "`/kaggle/working/test_output/test/` |\n"
                "| **full** | Entire dataset, compact persisted artifacts | "
                "`pipeline_output/data/` → publish as Kaggle Dataset |\n\n"
                "**Flow:** Configuration → Environment setup → Load config → Branch on `MODE` → "
                "(full only) validate the already-persisted artifacts. By default, completed subjects "
                "retain epochs, QC, and logs while large resumable intermediate FIF files are removed.",
            ),
            ("code", PREPROCESS_LOAD_CONFIG),
            ("code", PREPROCESS_RUN_CELL),
            ("code", PREPROCESS_SAVE),
        ],
        save=False,
        pre_setup_cells=[("code", PREPROCESS_PIPELINE_CONFIG)],
    ),
    "02_epoching.ipynb": _nb_cells(
        "02",
        "",
        [
            ("markdown", "# 02 — Epoching\n\nExport `*_epo.fif` checkpoints to `.npy` for future DL."),
            ("code", CONFIG_CELL),
            ("code", '''\
from eeg.contracts import validate_epoch_exports, validate_preprocessed_artifacts
from eeg.export import export_all_epochs_npy

upstream_contract = validate_preprocessed_artifacts(
    CONFIG["dataset"], CONFIG["experiment"], require_participants=False
)
print("Input contract:", upstream_contract)
paths = export_all_epochs_npy(CONFIG["dataset"], CONFIG["experiment"])
output_contract = validate_epoch_exports(paths)
print("Output contract:", output_contract)
'''),
        ],
    ),
    "03_feature_extraction.ipynb": _nb_cells(
        "03",
        "",
        [
            (
                "markdown",
                "# 03 — Feature Extraction\n\n"
                "Feature extraction is checkpointed once per subject. The default "
                "bounded batch stays comfortably below Kaggle's 12-hour limit. "
                "Create one output Dataset for this notebook. After each partial run, "
                "publish a new version of that same Dataset, attach its latest version "
                "on the next run, and keep `PIPELINE_INPUT` set to its slug. Notebook "
                "04 must only be started after this notebook reports `COMPLETE`."
            ),
            ("code", CONFIG_CELL),
            ("code", '''\
from eeg.contracts import validate_preprocessed_artifacts
input_contract = validate_preprocessed_artifacts(
    CONFIG["dataset"], CONFIG["experiment"]
)
print("Input contract:", input_contract)

# At the observed throughput of ~10.7 wall-clock minutes per completed subject
# with two workers, 24 subjects take roughly 4.3 hours. Set to None only outside
# a runtime-limited environment.
FEATURE_SUBJECT_LIMIT = 24

from scripts.extract_features import run_extract
results = run_extract(
    CONFIG["dataset"],
    CONFIG["experiment"],
    workers=2,
    limit=FEATURE_SUBJECT_LIMIT,
)
checkpointed = sum(
    result.get("status") in {"ok", "skipped"} for result in results
)
expected = input_contract["epoch_checkpoints"]

from eeg.contracts import validate_feature_artifact
from eeg.training.datasets import feature_columns
if checkpointed < expected:
    print(
        f"PARTIAL: {checkpointed}/{expected} subjects checkpointed. "
        "Publish this output as a new version of this notebook's existing Kaggle "
        "Dataset (create it only if this is the first batch), attach the latest "
        "version, keep PIPELINE_INPUT set to its slug, and run notebook 03 again."
    )
else:
    print(
        "COMPLETE — output contract:",
        validate_feature_artifact(
            CONFIG["dataset"], CONFIG["experiment"], feature_columns()
        ),
    )
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
from eeg.contracts import validate_feature_artifact, validate_selection_artifacts

spec = resolve_dataset(CONFIG["dataset"])[0]
print(
    "Input contract:",
    validate_feature_artifact(
        CONFIG["dataset"], CONFIG["experiment"], feature_columns()
    ),
)
df = load_features_df(CONFIG["dataset"], CONFIG["experiment"], spec.id)
cols = feature_columns()
result = select_features(df, cols, config=CONFIG)
paths = save_selection_artifacts(df, result, CONFIG["dataset"], CONFIG["experiment"])
print(f"Selected {result.n_selected}/{result.n_input} features")
print("Output contract:", validate_selection_artifacts(CONFIG["dataset"], CONFIG["experiment"]))
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
from eeg.contracts import (
    validate_benchmark_artifacts,
    validate_feature_artifact,
    validate_selection_artifacts,
)

spec = resolve_dataset(CONFIG["dataset"])[0]
print(
    "Input contract:",
    validate_feature_artifact(
        CONFIG["dataset"], CONFIG["experiment"], feature_columns()
    ),
)
df = load_features_df(CONFIG["dataset"], CONFIG["experiment"], spec.id)
sel = select_features(df, feature_columns(), config=CONFIG)
save_selection_artifacts(df, sel, CONFIG["dataset"], CONFIG["experiment"])
print("Selection contract:", validate_selection_artifacts(CONFIG["dataset"], CONFIG["experiment"]))

result = run_benchmark(
    dataset=CONFIG["dataset"],
    experiment=CONFIG["experiment"],
    models=["logistic_regression", "random_forest", "xgboost", "mlp"],
    feature_cols=sel.selected_columns,
    config=CONFIG,
)
import pandas as pd
print("Output contract:", validate_benchmark_artifacts(CONFIG["dataset"], CONFIG["experiment"]))
display(pd.read_csv(result["benchmark_csv"]))
'''),
        ],
    ),
    "06_deep_learning.ipynb": _nb_cells(
        "06",
        "",
        [
            ("markdown", """# 06 — Deep Learning

Train a compact EEGNet-style classifier with leakage-safe subject-level evaluation.

## Input contract
- `data/preprocessed/{dataset}/{experiment}/epochs/sub-XXX.npy`
- Shape: `(n_epochs, n_channels, n_samples)` float32

Use the output Dataset from notebook 02 and enable a Kaggle GPU accelerator. All
epochs belonging to a participant remain together in the outer train/test folds and
the inner early-stopping split. Metrics are calculated after averaging epoch
probabilities for each participant.
"""),
            ("code", CONFIG_CELL),
            ("code", '''\
from eeg.contracts import validate_epoch_exports
from eeg.paths import epochs_npy_dir
epoch_dir = epochs_npy_dir(CONFIG["dataset"], CONFIG["experiment"])
print("Epoch npy dir:", epoch_dir)
print("Input contract:", validate_epoch_exports(sorted(epoch_dir.glob("sub-*.npy"))))
'''),
            ("code", '''\
from eeg.training.deep_learning import run_deep_learning_benchmark
import pandas as pd

DL_CONFIG = {
    "cv_folds": CONFIG["cv_folds"],
    "validation_size": 0.2,
    "max_epochs": 30,
    "patience": 6,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.5,
    "bootstrap_iterations": 1000,
    "seed": CONFIG["seed"],
}

result = run_deep_learning_benchmark(
    CONFIG["dataset"],
    CONFIG["experiment"],
    **DL_CONFIG,
)
display(pd.read_csv(result["benchmark_csv"]))
print("Model:", result["model_path"])
print("Predictions:", result["predictions_csv"])
'''),
        ],
    ),
    "07_ablation.ipynb": _nb_cells(
        "07",
        "",
        [
            ("markdown", "# 07 — Ablation\n\nCompare preprocessing experiment variants via `run_benchmark()`."),
            ("code", CONFIG_CELL),
            ("code", '''\
from eeg.training.benchmark import run_benchmark
from eeg.contracts import validate_benchmark_artifacts, validate_feature_artifact
from eeg.training.datasets import feature_columns
import pandas as pd

print(
    "Input contract:",
    validate_feature_artifact(
        CONFIG["dataset"], CONFIG["experiment"], feature_columns()
    ),
)
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
        print(f"Output contract ({exp}):", validate_benchmark_artifacts(CONFIG["dataset"], exp))
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
if not csv_files:
    raise FileNotFoundError(
        "No benchmark.csv artifacts found. Attach notebook 05/07 output via PIPELINE_INPUT."
    )
frames = []
for p in csv_files:
    df = pd.read_csv(p)
    required = {"model", "dataset", "experiment", "balanced_accuracy"}
    missing = sorted(required - set(df.columns))
    if df.empty or missing:
        raise ValueError(f"Invalid benchmark artifact {p}; missing={missing}")
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


def validate_notebook(name: str, notebook: dict) -> None:
    """Keep storage and packaging regressions out of generated notebooks."""
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    forbidden = {
        "/kaggle/working/ad_eeg": "repository clone would leak into saved output",
        'shutil.copytree(src, out / "data"': "publish step would duplicate the artifact tree",
    }
    for snippet, reason in forbidden.items():
        if snippet in code:
            raise ValueError(f"{name}: {reason}")
    if 'PROJECT_DIR = "/kaggle/temp/ad_eeg"' not in code:
        raise ValueError(f"{name}: temporary repository path is missing")
    if 'OUTPUT_DIR = "/kaggle/working/pipeline_output"' not in code:
        raise ValueError(f"{name}: dedicated output root is missing")
    if name.startswith(("00_", "01_")):
        if 'RAW_EEG_INPUT = "REPLACE_WITH_RAW_EEG_DATASET_SLUG"' not in code:
            raise ValueError(f"{name}: raw EEG input contract is missing")
        if "PIPELINE_INPUT = None" not in code:
            raise ValueError(f"{name}: must not require prior pipeline output")
    else:
        if "RAW_EEG_INPUT = None" not in code:
            raise ValueError(f"{name}: downstream stage must not require raw EEG")
        if 'PIPELINE_INPUT = "REPLACE_WITH_PRIOR_PIPELINE_OUTPUT_SLUG"' not in code:
            raise ValueError(f"{name}: prior-stage pipeline input contract is missing")


for name, spec in NOTEBOOKS.items():
    path = OUT / name
    notebook = make_notebook(spec)
    validate_notebook(name, notebook)
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print("wrote", path)
