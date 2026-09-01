"""I/O helpers: raw EEG loading, checkpoint save/load, hashing, parquet."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from eeg.config import load_base_configs
from eeg.paths import (
    LEGACY_STAGE_SUFFIX,
    STAGE_SUFFIX,
    experiment_metadata_path,
    features_parquet_path,
    legacy_parquet_path,
    preprocessed_dir,
    resolve_features_path,
)

CHECKPOINT_STAGES = ("raw", "filtered", "ica", "clean", "epochs")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def read_eeg_data(file_path: str | Path, sfreq: int | None = None) -> mne.io.BaseRaw:
    cfg = load_base_configs()
    sfreq = sfreq or cfg["features"].get("sampling_rate", 500)
    file_path = str(file_path)
    if os.path.splitext(file_path)[1].lower() == ".set":
        return mne.io.read_raw_eeglab(file_path, preload=True)

    files = [f for f in os.listdir(file_path) if f.endswith(".txt")]
    all_data, ch_names = [], []
    for f in files:
        signal = np.loadtxt(os.path.join(file_path, f))
        all_data.append(signal)
        ch_names.append(f.split(".")[0])
    data = np.vstack(all_data)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    return mne.io.RawArray(data, info)


def save_raw(raw: mne.io.BaseRaw, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(str(path), overwrite=True)


def load_raw(path: Path) -> mne.io.BaseRaw:
    return mne.io.read_raw_fif(str(path), preload=True)


def save_epochs(epochs: mne.Epochs, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(str(path), overwrite=True)


def load_epochs(path: Path) -> mne.Epochs:
    return mne.read_epochs(str(path), preload=True)


def save_checkpoint(obj: mne.io.BaseRaw | mne.Epochs, path: Path) -> None:
    if isinstance(obj, mne.Epochs):
        save_epochs(obj, path)
    else:
        save_raw(obj, path)


def load_checkpoint(path: Path, stage: str) -> mne.io.BaseRaw | mne.Epochs:
    if stage == "epochs":
        return load_epochs(path)
    return load_raw(path)


def _legacy_checkpoint_path(path: Path, stage: str) -> Path | None:
    legacy_suffix = LEGACY_STAGE_SUFFIX.get(stage)
    if not legacy_suffix:
        return None
    suffix = STAGE_SUFFIX[stage]
    name = path.name
    if not name.endswith(suffix):
        return None
    return path.parent / f"{name[: -len(suffix)]}{legacy_suffix}"


def try_load_checkpoint(path: Path, stage: str) -> mne.io.BaseRaw | mne.Epochs | None:
    for candidate in (path, _legacy_checkpoint_path(path, stage)):
        if candidate is None or not candidate.exists():
            continue
        try:
            return load_checkpoint(candidate, stage)
        except Exception:
            continue
    return None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def list_subjects(dataset_spec) -> pd.DataFrame:
    tsv = dataset_spec.raw_dir / "participants.tsv"
    df = pd.read_csv(tsv, sep="\t")
    df["Dataset"] = dataset_spec.id
    return df


def list_preprocessed_subjects(dataset_spec, experiment: str) -> pd.DataFrame:
    """Load labels from raw data or the existing preprocessing metadata artifact."""
    raw_path = dataset_spec.raw_dir / "participants.tsv"
    if raw_path.is_file():
        df = pd.read_csv(raw_path, sep="\t")
        df["Dataset"] = dataset_spec.id
        return df

    metadata_path = experiment_metadata_path(dataset_spec.name, experiment)
    if metadata_path.is_file():
        participant_index = read_json(metadata_path).get("participant_index", [])
        if participant_index:
            df = pd.DataFrame(participant_index)
            df["Dataset"] = dataset_spec.id
            return df
    raise FileNotFoundError(
        f"Participant metadata not found at {raw_path} or in {metadata_path}. "
        "Re-run notebook 01 in full mode so metadata.json includes participant_index."
    )


def save_features_parquet(
    df: pd.DataFrame,
    participant_id: str,
    dataset_name: str,
    experiment: str,
    label: str,
    dataset_id: int,
) -> Path:
    rows = prepare_feature_rows(
        df,
        participant_id,
        dataset_name,
        label,
        dataset_id,
    )
    return merge_features_parquet([rows], dataset_name, experiment)


def prepare_feature_rows(
    df: pd.DataFrame,
    participant_id: str,
    dataset_name: str,
    label: str,
    dataset_id: int,
) -> pd.DataFrame:
    """Return a copy of one subject's features with identity columns."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    else:
        df = df.copy()

    df["participant_id"] = participant_id
    df["dataset_id"] = dataset_id
    df["dataset_name"] = dataset_name
    df["label"] = label
    return df


def merge_features_parquet(
    frames: list[pd.DataFrame],
    dataset_name: str,
    experiment: str,
) -> Path:
    """Merge completed subject frames into the aggregate with one atomic write.

    Callers may compute or serialize subjects in parallel, but only the parent
    process should call this function for a dataset batch.
    """
    path = features_parquet_path(dataset_name, experiment)
    if not frames:
        return path

    incoming = pd.concat(frames, ignore_index=True)
    if "participant_id" not in incoming.columns:
        raise ValueError("Feature frames must include participant_id")

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        participant_ids = set(incoming["participant_id"].astype(str))
        existing = existing[
            ~existing["participant_id"].astype(str).isin(participant_ids)
        ]
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming

    # A failed write must not destroy the last complete aggregate.
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        combined.to_parquet(tmp_path, engine="pyarrow", index=False)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


def load_features_df(
    dataset_name: str | None = None,
    experiment: str = "baseline",
    dataset_id: int | None = None,
) -> pd.DataFrame:
    if dataset_name is not None:
        path = resolve_features_path(dataset_name, experiment, dataset_id)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run extract_features.py for {dataset_name}/{experiment} first."
            )
        return pd.read_parquet(path)

    data_dir = Path(__file__).resolve().parents[1] / "data" / "features"
    if data_dir.exists():
        parquets = list(data_dir.rglob("subject_features.parquet"))
        if len(parquets) == 1:
            return pd.read_parquet(parquets[0])

    legacy = [legacy_parquet_path(i) for i in (2, 3)]
    existing = [p for p in legacy if p.exists()]
    if len(existing) == 1:
        return pd.read_parquet(existing[0])

    raise FileNotFoundError("No feature parquet found. Run extract_features.py first.")
