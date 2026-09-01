"""Regression tests for preprocessing output path routing."""

from types import SimpleNamespace

import pandas as pd

from scripts import preprocess_dataset


def _patch_preprocess_run(monkeypatch, tmp_path):
    dataset = SimpleNamespace(name="eyesclosed", id=2, task="eyesclosed")
    monkeypatch.setattr(preprocess_dataset, "resolve_datasets_arg", lambda tag: [dataset])
    monkeypatch.setattr(preprocess_dataset, "load_experiment", lambda experiment: {})
    monkeypatch.setattr(preprocess_dataset, "config_fingerprint", lambda config: "fp")
    monkeypatch.setattr(
        preprocess_dataset,
        "list_subjects",
        lambda ds: pd.DataFrame(
            [{"participant_id": "sub-001", "Group": "A"}]
        ),
    )
    monkeypatch.setattr(
        preprocess_dataset,
        "raw_eeg_path",
        lambda ds, subject_num: tmp_path / "sub-001.set",
    )
    metadata_calls = []
    monkeypatch.setattr(
        preprocess_dataset,
        "update_experiment_metadata",
        lambda *args, **kwargs: metadata_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        preprocess_dataset,
        "write_preprocess_report",
        lambda *a, **k: {"summary_csv": tmp_path / "summary.csv"},
    )
    captured = []

    def fake_parallel(tasks, worker, workers):
        captured.extend(tasks)
        return []

    monkeypatch.setattr(preprocess_dataset, "run_parallel", fake_parallel)
    return captured, metadata_calls


def test_default_output_uses_canonical_preprocessing_paths(tmp_path, monkeypatch):
    canonical = tmp_path / "data" / "preprocessed" / "eyesclosed" / "baseline"
    monkeypatch.setattr(preprocess_dataset, "preprocessed_dir", lambda ds, exp: canonical)
    captured, metadata_calls = _patch_preprocess_run(monkeypatch, tmp_path)

    preprocess_dataset.run_preprocess("eyesclosed", "baseline")

    assert captured[0][-1] is None
    assert not (canonical / "participants.tsv").exists()
    assert metadata_calls[0][1]["extra_metadata"] == {
        "participant_index": [{"participant_id": "sub-001", "Group": "A"}]
    }


def test_custom_output_is_dataset_and_experiment_scoped(tmp_path, monkeypatch):
    captured, _ = _patch_preprocess_run(monkeypatch, tmp_path)
    custom_root = tmp_path / "pipeline_output"

    preprocess_dataset.run_preprocess(
        "eyesclosed",
        "baseline",
        output_dir=custom_root,
    )

    assert captured[0][-1] == str(custom_root / "eyesclosed" / "baseline")
    assert not (
        custom_root / "eyesclosed" / "baseline" / "participants.tsv"
    ).exists()
