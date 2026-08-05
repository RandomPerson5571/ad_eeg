"""Tests for dataset audit module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg.audit import _build_dataset_summary, _build_patient_summary, write_audit_artifacts
from eeg.repro import attach_repro_metadata, init_repro, snapshot_environment


def test_init_repro_sets_seed():
    r1 = init_repro(123)
    a = np.random.rand()
    init_repro(123)
    b = np.random.rand()
    assert a == b
    assert r1["seed"] == 123


def test_snapshot_environment_has_keys():
    env = snapshot_environment()
    assert "python" in env
    assert "git_sha" in env
    assert "timestamp" in env


def test_attach_repro_metadata():
    meta = attach_repro_metadata({}, {"seed": 42, "experiment": "baseline"})
    assert meta["seed"] == 42
    assert "git_sha" in meta


def test_build_patient_summary():
    metadata = pd.DataFrame(
        [
            {"participant_id": "sub-001", "diagnosis": "AD", "duration_s": 100.0, "load_ok": True, "age": 70, "sex": "M"},
            {"participant_id": "sub-002", "diagnosis": "HC", "duration_s": 200.0, "load_ok": True, "age": 65, "sex": "F"},
        ]
    )
    summary = _build_patient_summary(metadata)
    assert len(summary) == 2
    assert "n_recordings" in summary.columns


def test_build_dataset_summary():
    metadata = pd.DataFrame(
        [
            {"participant_id": "sub-001", "diagnosis": "AD", "duration_s": 100.0, "load_ok": True, "n_channels": 19, "sfreq": 500.0},
            {"participant_id": "sub-002", "diagnosis": "HC", "duration_s": 200.0, "load_ok": True, "n_channels": 19, "sfreq": 500.0},
        ]
    )
    participants = pd.DataFrame({"Age": [70, 65], "Sex": ["M", "F"]})
    summary = _build_dataset_summary(metadata, participants, [])
    assert summary["n_subjects"] == 2
    assert summary["n_recordings"] == 2
    assert summary["corrupt_files"]["count"] == 0


def test_write_audit_artifacts(tmp_path, monkeypatch):
    import eeg.audit as audit_mod

    monkeypatch.setattr(audit_mod, "audit_dir", lambda d: tmp_path / d)

    metadata = pd.DataFrame([{"participant_id": "sub-001", "diagnosis": "AD", "load_ok": True, "duration_s": 1.0}])
    from eeg.audit import AuditResult

    result = AuditResult(
        metadata=metadata,
        patient_summary=_build_patient_summary(metadata),
        dataset_summary={"n_subjects": 1},
    )
    paths = write_audit_artifacts(result, "eyesclosed", environment={"test": True})
    assert paths["metadata"].exists()
    assert (tmp_path / "eyesclosed" / "environment.json").exists()
