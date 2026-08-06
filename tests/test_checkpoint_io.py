"""Tests for checkpoint save/load roundtrip."""

import mne
import numpy as np
import pytest

from eeg.config import load_base_configs
import json

from eeg.io import load_checkpoint, read_json, save_checkpoint, try_load_checkpoint, write_json


def _synthetic_raw(n_channels=4, n_times=1000):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_channels, n_times)) * 1e-6
    info = mne.create_info(
        ch_names=[f"EEG{i}" for i in range(n_channels)],
        sfreq=500,
        ch_types=["eeg"] * n_channels,
    )
    return mne.io.RawArray(data, info)


def test_raw_checkpoint_roundtrip(tmp_path):
    raw = _synthetic_raw()
    path = tmp_path / "sub-001_raw.fif"
    save_checkpoint(raw, path)
    loaded = load_checkpoint(path, "raw")
    assert loaded.n_times == raw.n_times


def test_epochs_checkpoint_roundtrip(tmp_path):
    raw = _synthetic_raw(n_times=4000)
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, overlap=1.0, preload=True)
    path = tmp_path / "sub-001_epo.fif"
    save_checkpoint(epochs, path)
    loaded = load_checkpoint(path, "epochs")
    assert len(loaded) == len(epochs)


def test_try_load_missing_returns_none(tmp_path):
    assert try_load_checkpoint(tmp_path / "missing.fif", "raw") is None


def test_try_load_corrupt_returns_none(tmp_path):
    bad = tmp_path / "bad.fif"
    bad.write_text("not a fif file")
    assert try_load_checkpoint(bad, "raw") is None


def test_write_json_handles_numpy_scalars(tmp_path):
    path = tmp_path / "log.json"
    write_json(
        path,
        {
            "ica_excluded_indices": [np.int64(0), np.int64(3)],
            "pct_epochs_rejected": np.float64(12.5),
            "flag": np.bool_(True),
        },
    )
    loaded = read_json(path)
    assert loaded == {
        "ica_excluded_indices": [0, 3],
        "pct_epochs_rejected": 12.5,
        "flag": True,
    }
    assert all(isinstance(v, int) for v in loaded["ica_excluded_indices"])
