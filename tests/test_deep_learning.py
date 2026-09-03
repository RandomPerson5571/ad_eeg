"""Tests for leakage-safe deep-learning data preparation."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from eeg.training.deep_learning import (
    EpochArrayDataset,
    _channel_statistics,
    _resolve_training_device,
    _subject_folds,
    _train_validation_split,
)


def _subjects(tmp_path, per_class=5):
    rows = []
    subject_number = 0
    for label_id in range(3):
        for _ in range(per_class):
            path = tmp_path / f"sub-{subject_number:03d}.npy"
            np.save(
                path,
                np.full((label_id + 1, 2, 8), subject_number, dtype=np.float32),
            )
            rows.append(
                {
                    "participant_id": path.stem,
                    "Group": ["A", "C", "F"][label_id],
                    "label_id": label_id,
                    "path": path,
                }
            )
            subject_number += 1
    return pd.DataFrame(rows)


def test_subject_folds_never_mix_participants(tmp_path):
    subjects = _subjects(tmp_path)
    folds = list(_subject_folds(subjects, n_splits=5, seed=42))

    test_ids = []
    for train, test in folds:
        assert set(train["participant_id"]).isdisjoint(test["participant_id"])
        assert set(test["label_id"]) == {0, 1, 2}
        test_ids.extend(test["participant_id"])
    assert sorted(test_ids) == sorted(subjects["participant_id"])


def test_validation_split_is_subject_level_and_stratified(tmp_path):
    subjects = _subjects(tmp_path, per_class=10)

    train, validation = _train_validation_split(subjects, validation_size=0.2, seed=42)

    assert set(train["participant_id"]).isdisjoint(validation["participant_id"])
    assert validation["label_id"].value_counts().to_dict() == {0: 2, 1: 2, 2: 2}


def test_normalization_and_sampling_weights_use_subjects_not_epoch_counts(tmp_path):
    subjects = _subjects(tmp_path, per_class=2)
    mean, std = _channel_statistics(subjects)
    dataset = EpochArrayDataset(subjects, mean, std)
    weights = dataset.balanced_sample_weights()

    assert mean.shape == (2,)
    assert std.shape == (2,)
    mass_by_subject = {}
    for weight, (_, _, _, participant_id) in zip(weights, dataset.records):
        mass_by_subject[participant_id] = mass_by_subject.get(participant_id, 0.0) + weight
    assert len({round(value, 12) for value in mass_by_subject.values()}) == 1


def test_explicit_cpu_skips_cuda_probe():
    fake_torch = SimpleNamespace(device=lambda value: value)

    assert _resolve_training_device(fake_torch, "cpu") == "cpu"


def test_incompatible_cuda_fails_with_accelerator_guidance(monkeypatch):
    fake_torch = SimpleNamespace(
        device=lambda value: value,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _: "Tesla P100",
        ),
    )
    monkeypatch.setattr(
        "eeg.training.deep_learning._probe_cuda_device",
        lambda *_: (_ for _ in ()).throw(RuntimeError("no kernel image")),
    )

    with pytest.raises(RuntimeError, match="Pascal-compatible"):
        _resolve_training_device(fake_torch, None)
