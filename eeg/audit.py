"""Dataset audit: per-recording metadata and cohort summary statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import ALL_CHANNELS
from eeg.config import DatasetSpec
from eeg.io import list_subjects, read_eeg_data, write_json
from eeg.paths import audit_dir, participants_tsv, raw_eeg_path


@dataclass
class AuditResult:
    metadata: pd.DataFrame
    patient_summary: pd.DataFrame
    dataset_summary: dict[str, Any]
    environment: dict[str, Any] | None = None
    corrupt_files: list[str] = field(default_factory=list)


def _subject_num_from_row(idx: int) -> int:
    return idx + 1


def _diagnosis_label(group: str) -> str:
    mapping = {"A": "AD", "F": "FTD", "C": "HC"}
    return mapping.get(str(group).strip().upper(), str(group))


def _missing_electrodes(ch_names: list[str]) -> str:
    expected = set(ALL_CHANNELS)
    present = set(ch_names)
    missing = sorted(expected - present)
    return ";".join(missing) if missing else ""


def audit_dataset(dataset_spec: DatasetSpec) -> AuditResult:
    """Load raw recordings and build audit tables (no preprocessing)."""
    participants = list_subjects(dataset_spec)
    rows: list[dict[str, Any]] = []
    corrupt: list[str] = []

    for idx, row in participants.iterrows():
        subject_num = _subject_num_from_row(idx)
        participant_id = row.get("participant_id", f"sub-{subject_num:03d}")
        raw_path = raw_eeg_path(dataset_spec, subject_num)
        diagnosis = _diagnosis_label(row.get("Group", row.get("group", "")))
        age = row.get("Age", row.get("age", np.nan))
        sex = row.get("Sex", row.get("sex", np.nan))

        record = {
            "participant_id": participant_id,
            "diagnosis": diagnosis,
            "session": "01",
            "task": dataset_spec.task,
            "age": age,
            "sex": sex,
            "recording_date": np.nan,
            "duration_s": np.nan,
            "sfreq": np.nan,
            "n_channels": np.nan,
            "ch_names": "",
            "missing_electrodes": "",
            "bad_channels": "",
            "file_path": str(raw_path),
            "load_ok": False,
            "load_error": "",
        }

        if not raw_path.exists():
            record["load_error"] = "file not found"
            corrupt.append(str(raw_path))
            rows.append(record)
            continue

        try:
            raw = read_eeg_data(raw_path)
            ch_names = list(raw.ch_names)
            meas_date = raw.info.get("meas_date")
            record.update(
                {
                    "recording_date": meas_date.isoformat() if meas_date else np.nan,
                    "duration_s": float(raw.n_times / raw.info["sfreq"]),
                    "sfreq": float(raw.info["sfreq"]),
                    "n_channels": len(ch_names),
                    "ch_names": ";".join(ch_names),
                    "missing_electrodes": _missing_electrodes(ch_names),
                    "load_ok": True,
                }
            )
        except Exception as exc:
            record["load_error"] = str(exc)
            corrupt.append(str(raw_path))

        rows.append(record)

    metadata = pd.DataFrame(rows)
    patient_summary = _build_patient_summary(metadata)
    dataset_summary = _build_dataset_summary(metadata, participants, corrupt)
    return AuditResult(
        metadata=metadata,
        patient_summary=patient_summary,
        dataset_summary=dataset_summary,
        corrupt_files=corrupt,
    )


def _build_patient_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    agg_spec: dict[str, Any] = {
        "diagnosis": ("diagnosis", "first"),
        "n_recordings": ("participant_id", "count"),
        "mean_duration_s": ("duration_s", "mean"),
        "load_ok": ("load_ok", "all"),
    }
    if "age" in metadata.columns:
        agg_spec["age"] = ("age", "first")
    if "sex" in metadata.columns:
        agg_spec["sex"] = ("sex", "first")

    agg = (
        metadata.groupby("participant_id", as_index=False)
        .agg(**agg_spec)
        .sort_values("participant_id")
    )
    return agg


def _build_dataset_summary(
    metadata: pd.DataFrame,
    participants: pd.DataFrame,
    corrupt: list[str],
) -> dict[str, Any]:
    ok = metadata[metadata["load_ok"]]
    durations = ok["duration_s"].dropna()

    def _class_counts(series: pd.Series) -> dict[str, int]:
        return {str(k): int(v) for k, v in series.value_counts().items()}

    age = participants.get("Age", participants.get("age", pd.Series(dtype=float)))
    sex = participants.get("Sex", participants.get("sex", pd.Series(dtype=float)))

    summary: dict[str, Any] = {
        "n_subjects": int(metadata["participant_id"].nunique()),
        "n_recordings": int(len(metadata)),
        "patients_per_class": _class_counts(metadata.groupby("participant_id")["diagnosis"].first()),
        "recordings_per_class": _class_counts(metadata["diagnosis"]),
        "mean_duration_s": float(durations.mean()) if len(durations) else None,
        "std_duration_s": float(durations.std()) if len(durations) else None,
        "recording_lengths": {
            "min": float(durations.min()) if len(durations) else None,
            "max": float(durations.max()) if len(durations) else None,
            "median": float(durations.median()) if len(durations) else None,
        },
        "channel_count_distribution": {
            str(int(k)): int(v) for k, v in ok["n_channels"].value_counts().items()
        },
        "sfreq": sorted({float(x) for x in ok["sfreq"].dropna().unique()}),
        "missing_values": {
            col: int(participants[col].isna().sum())
            for col in participants.columns
            if participants[col].isna().any()
        },
        "sex_distribution": _class_counts(sex.dropna().astype(str)),
        "age_distribution": {
            "mean": float(age.mean()) if len(age.dropna()) else None,
            "std": float(age.std()) if len(age.dropna()) else None,
            "histogram_bins": np.histogram(age.dropna(), bins=10)[0].tolist()
            if len(age.dropna())
            else [],
        },
        "corrupt_files": {"count": len(corrupt), "paths": corrupt},
    }
    return summary


def write_audit_artifacts(
    result: AuditResult,
    dataset: str,
    environment: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write metadata.csv, patient_summary.csv, dataset_summary.json, environment.json."""
    out = audit_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "metadata": out / "metadata.csv",
        "patient_summary": out / "patient_summary.csv",
        "dataset_summary": out / "dataset_summary.json",
    }
    result.metadata.to_csv(paths["metadata"], index=False)
    result.patient_summary.to_csv(paths["patient_summary"], index=False)
    write_json(paths["dataset_summary"], result.dataset_summary)

    if environment is not None:
        env_path = out / "environment.json"
        write_json(env_path, environment)
        paths["environment"] = env_path

    return paths
