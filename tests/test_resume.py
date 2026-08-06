"""Tests for auto-resume logic."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from eeg.config import config_fingerprint, load_experiment
from eeg.io import write_json
from eeg.paths import checkpoint_path, subject_log_path
from eeg.preprocessing import _furthest_valid_stage, preprocess_subject


def test_furthest_valid_stage_none_when_no_log(tmp_path):
    with patch("eeg.preprocessing.subject_log_path") as mock_log:
        mock_log.return_value = tmp_path / "missing.json"
        assert _furthest_valid_stage("eyesclosed", "baseline", "sub-001", "abc", "def") is None


def test_preprocess_skips_when_epochs_complete(tmp_path):
    """If valid log + epochs checkpoint exist, subject is skipped."""
    config = load_experiment("fast")
    fp = config_fingerprint(config)
    dataset = "eyesclosed"
    experiment = "baseline"
    participant = "sub-001"

    log_path = tmp_path / "sub-001.json"
    write_json(
        log_path,
        {
            "participant_id": participant,
            "raw_sha256": "deadbeef",
            "config_fingerprint": fp,
            "status": "ok",
            "stages_completed": ["raw", "filtered", "ica", "clean", "epochs"],
        },
    )

    epochs_cp = tmp_path / "sub-001_epo.fif"
    # Create minimal epochs file via synthetic raw
    import mne
    import numpy as np

    data = np.random.randn(2, 2000) * 1e-6
    info = mne.create_info(["E1", "E2"], 500, "eeg")
    raw = mne.io.RawArray(data, info)
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=True)
    epochs.save(str(epochs_cp), overwrite=True)

    from eeg.paths import STAGE_SUFFIX

    with patch("eeg.preprocessing.subject_log_path", return_value=log_path), patch(
        "eeg.preprocessing.checkpoint_path",
        side_effect=lambda ds, exp, pid, stage: tmp_path / f"{pid}{STAGE_SUFFIX[stage]}",
    ), patch(
        "eeg.preprocessing.resolve_checkpoint_path",
        side_effect=lambda ds, exp, pid, stage: tmp_path / f"{pid}{STAGE_SUFFIX[stage]}",
    ), patch("eeg.preprocessing.sha256_file", return_value="deadbeef"):
        result = preprocess_subject(
            raw_path=tmp_path / "raw.set",
            participant_id=participant,
            dataset_name=dataset,
            experiment=experiment,
            config=config,
            config_fp=fp,
            force=False,
        )

    assert result.status == "skipped"
