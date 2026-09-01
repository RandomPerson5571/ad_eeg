"""Tests for auto-resume logic."""

import json
from contextlib import ExitStack
from unittest.mock import patch

from eeg.config import config_fingerprint, load_experiment
from eeg.io import read_json, write_json
from eeg.preprocessing import _furthest_valid_stage, _log_is_valid, preprocess_subject


def test_furthest_valid_stage_none_when_no_log(tmp_path):
    with patch("eeg.preprocessing.subject_log_path") as mock_log:
        mock_log.return_value = tmp_path / "missing.json"
        assert _furthest_valid_stage("eyesclosed", "baseline", "sub-001", "abc", "def") is None


def test_legacy_skipped_log_remains_resume_eligible():
    assert _log_is_valid(
        {
            "raw_sha256": "raw-hash",
            "config_fingerprint": "config-hash",
            "status": "skipped",
        },
        "raw-hash",
        "config-hash",
    )


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
    ), patch("eeg.preprocessing.sha256_file", return_value="deadbeef"):
        results = [
            preprocess_subject(
                raw_path=tmp_path / "raw.set",
                participant_id=participant,
                dataset_name=dataset,
                experiment=experiment,
                config=config,
                config_fp=fp,
                force=False,
            )
            for _ in range(3)
        ]

    assert [result.status for result in results] == ["skipped"] * 3
    assert read_json(log_path)["status"] == "ok"


def test_custom_output_uses_canonical_checkpoint_names(tmp_path):
    participant = "sub-007"
    output_dir = tmp_path / "eyesclosed" / "baseline"
    saved_paths = []
    sentinel = object()

    def stage(*args, **kwargs):
        return sentinel, {}

    patches = [
        patch(
            "eeg.preprocessing.load_base_configs",
            return_value={"features": {"sampling_rate": 500}},
        ),
        patch("eeg.preprocessing.sha256_file", return_value="raw-hash"),
        patch("eeg.preprocessing.stage_raw", side_effect=stage),
        patch("eeg.preprocessing.stage_filtered", side_effect=stage),
        patch("eeg.preprocessing.stage_ica", side_effect=stage),
        patch("eeg.preprocessing.stage_clean", side_effect=stage),
        patch("eeg.preprocessing.stage_epochs", side_effect=stage),
        patch(
            "eeg.preprocessing.save_checkpoint",
            side_effect=lambda obj, path: saved_paths.append(path),
        ),
    ]
    with ExitStack() as stack:
        for active_patch in patches:
            stack.enter_context(active_patch)
        result = preprocess_subject(
            raw_path=tmp_path / "raw.set",
            participant_id=participant,
            dataset_name="eyesclosed",
            experiment="baseline",
            config={},
            config_fp="config-hash",
            force=True,
            output_dir=output_dir,
        )

    assert result.status == "ok"
    assert [path.name for path in saved_paths] == [
        "sub-007_raw.fif",
        "sub-007_filtered_raw.fif",
        "sub-007_ica_raw.fif",
        "sub-007_clean_raw.fif",
        "sub-007_epo.fif",
    ]
    assert read_json(output_dir / "logs" / "sub-007.json")["status"] == "ok"
