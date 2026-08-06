"""Tests for SHA256 hashing and config fingerprinting."""

import hashlib
from pathlib import Path

from eeg.config import config_fingerprint, load_experiment
from eeg.io import sha256_file


def test_sha256_file(tmp_path):
    p = tmp_path / "test.bin"
    p.write_bytes(b"hello eeg")
    expected = hashlib.sha256(b"hello eeg").hexdigest()
    assert sha256_file(p) == expected


def test_config_fingerprint_changes_with_experiment():
    baseline = config_fingerprint(load_experiment("baseline"))
    fast = config_fingerprint(load_experiment("fast"))
    assert baseline != fast


def test_config_fingerprint_changes_with_pipeline_version():
    cfg = load_experiment("baseline")
    cfg_v1 = {**cfg, "experiment": {**cfg["experiment"], "pipeline_version": 1}}
    cfg_v2 = {**cfg, "experiment": {**cfg["experiment"], "pipeline_version": 2}}
    assert config_fingerprint(cfg_v1) != config_fingerprint(cfg_v2)
