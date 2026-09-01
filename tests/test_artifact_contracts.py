"""Regression tests for the Kaggle stage-to-stage artifact contracts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from eeg.contracts import (
    ArtifactContractError,
    validate_epoch_exports,
    validate_preprocessed_artifacts,
)
from eeg.export import export_all_epochs_npy
from eeg.io import list_preprocessed_subjects


def test_preprocessed_contract_accepts_epoch_checkpoints_and_participant_index(
    tmp_path, monkeypatch
):
    root = tmp_path / "preprocessed" / "eyesclosed" / "baseline"
    root.mkdir(parents=True)
    (root / "sub-001_epo.fif").touch()
    (root / "metadata.json").write_text(
        json.dumps(
            {"participant_index": [{"participant_id": "sub-001", "Group": "A"}]}
        )
    )
    monkeypatch.setattr("eeg.contracts.preprocessed_dir", lambda *_: root)

    result = validate_preprocessed_artifacts("eyesclosed", "baseline")

    assert result["participants"] == 1
    assert result["epoch_checkpoints"] == 1


def test_preprocessed_contract_allows_step_02_metadata_without_participant_index(
    tmp_path, monkeypatch
):
    root = tmp_path / "preprocessed" / "eyesclosed" / "baseline"
    root.mkdir(parents=True)
    (root / "sub-001_epo.fif").touch()
    (root / "metadata.json").write_text("{}")
    monkeypatch.setattr("eeg.contracts.preprocessed_dir", lambda *_: root)

    result = validate_preprocessed_artifacts(
        "eyesclosed", "baseline", require_participants=False
    )

    assert result["participants"] is None
    assert result["epoch_checkpoints"] == 1


def test_epoch_export_contract_requires_consistent_float32_3d_arrays(tmp_path):
    first = tmp_path / "sub-001.npy"
    second = tmp_path / "sub-002.npy"
    np.save(first, np.zeros((3, 19, 2000), dtype=np.float32))
    np.save(second, np.zeros((5, 19, 2000), dtype=np.float32))

    result = validate_epoch_exports([first, second])

    assert result["subjects"] == 2
    assert result["n_channels"] == 19
    assert result["n_samples"] == 2000


def test_epoch_export_contract_rejects_cross_subject_shape_mismatch(tmp_path):
    first = tmp_path / "sub-001.npy"
    second = tmp_path / "sub-002.npy"
    np.save(first, np.zeros((3, 19, 2000), dtype=np.float32))
    np.save(second, np.zeros((5, 18, 2000), dtype=np.float32))

    with pytest.raises(ArtifactContractError, match="Incompatible epoch shape"):
        validate_epoch_exports([first, second])


def test_export_all_epochs_discovers_checkpoints_without_raw_participants(
    tmp_path, monkeypatch
):
    root = tmp_path / "preprocessed" / "eyesclosed" / "baseline"
    root.mkdir(parents=True)
    (root / "sub-002_epo.fif").touch()
    (root / "sub-001_epo.fif").touch()
    exported = []

    monkeypatch.setattr("eeg.export.preprocessed_dir", lambda *_: root)

    def fake_export(dataset, experiment, participant_id, normalize=None):
        exported.append(participant_id)
        return root / "epochs" / f"{participant_id}.npy"

    monkeypatch.setattr("eeg.export.export_epochs_npy", fake_export)

    paths = export_all_epochs_npy("eyesclosed", "baseline")

    assert exported == ["sub-001", "sub-002"]
    assert [path.name for path in paths] == ["sub-001.npy", "sub-002.npy"]


def test_feature_extraction_labels_fall_back_to_preprocessing_metadata(
    tmp_path, monkeypatch
):
    raw_root = tmp_path / "raw"
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {"participant_index": [{"participant_id": "sub-001", "Group": "A"}]}
        )
    )
    spec = SimpleNamespace(name="eyesclosed", id=2, raw_dir=raw_root)
    monkeypatch.setattr("eeg.io.experiment_metadata_path", lambda *_: metadata_path)

    participants = list_preprocessed_subjects(spec, "baseline")

    assert participants.loc[0, "participant_id"] == "sub-001"
    assert participants.loc[0, "Group"] == "A"
    assert participants.loc[0, "Dataset"] == 2
