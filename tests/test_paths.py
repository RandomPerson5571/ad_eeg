"""Tests for dataset alias resolution and path helpers."""

from pathlib import Path

import pytest

from eeg.config import config_fingerprint, load_experiment, resolve_dataset
from eeg.paths import (
    checkpoint_path,
    features_parquet_path,
    preprocessed_dir,
    subject_log_path,
)


def test_resolve_dataset_numeric_alias():
    specs = resolve_dataset("2")
    assert len(specs) == 1
    assert specs[0].name == "eyesclosed"
    assert specs[0].id == 2


def test_resolve_dataset_semantic_alias():
    specs = resolve_dataset("photomark")
    assert specs[0].name == "photomark"
    assert specs[0].id == 3


def test_resolve_dataset_all():
    specs = resolve_dataset("all")
    assert len(specs) == 2
    names = {s.name for s in specs}
    assert names == {"eyesclosed", "photomark"}


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        resolve_dataset("nonexistent")


def test_checkpoint_paths():
    p = checkpoint_path("eyesclosed", "baseline", "sub-001", "epochs")
    assert p.name == "sub-001_epo.fif"
    assert "preprocessed" in str(p)
    assert "eyesclosed" in str(p)
    assert "baseline" in str(p)


def test_features_parquet_path():
    p = features_parquet_path("eyesclosed", "baseline")
    assert p.name == "subject_features.parquet"


def test_subject_log_path():
    p = subject_log_path("eyesclosed", "baseline", "sub-001")
    assert p.name == "sub-001.json"
    assert p.parent.name == "logs"


def test_config_fingerprint_stable():
    cfg = load_experiment("baseline")
    fp1 = config_fingerprint(cfg)
    fp2 = config_fingerprint(cfg)
    assert fp1 == fp2
    assert len(fp1) == 64


def test_audit_dir():
    from eeg.paths import audit_dir, epochs_npy_dir, selected_features_path

    assert audit_dir("eyesclosed").name == "eyesclosed"
    assert "audit" in str(audit_dir("eyesclosed"))
    assert epochs_npy_dir("eyesclosed", "baseline").name == "epochs"
    assert selected_features_path("eyesclosed", "baseline").name == "selected_features.parquet"
