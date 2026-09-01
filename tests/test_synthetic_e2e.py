"""Synthetic EEG-to-features-to-nested-evaluation regression test."""

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from eeg.features import extract_from_epochs
from eeg.training.benchmark import run_benchmark


def test_synthetic_feature_and_evaluation_pipeline(tmp_path, monkeypatch):
    """Exercise entropy, PSD, connectivity, parquet, and subject OOF evaluation."""
    rng = np.random.default_rng(2026)
    sfreq = 500
    ch_names = ["P3", "Pz", "P4", "O1", "O2", "Cz"]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    labels = ["A", "F", "C"]
    class_frequency = {"A": 6.0, "F": 10.0, "C": 20.0}
    subject_frames = []

    for subject_idx in range(9):
        label = labels[subject_idx % len(labels)]
        t = np.arange(400) / sfreq
        epochs = []
        for epoch_idx in range(2):
            channel_data = []
            for channel_idx in range(len(ch_names)):
                phase = 0.15 * channel_idx + 0.05 * epoch_idx
                signal = np.sin(2 * np.pi * class_frequency[label] * t + phase)
                signal += 0.08 * rng.standard_normal(len(t))
                channel_data.append(signal * 1e-6)
            epochs.append(channel_data)
        mne_epochs = mne.EpochsArray(np.asarray(epochs), info, verbose=False)
        features = pd.DataFrame(extract_from_epochs(mne_epochs))
        features["participant_id"] = f"sub-{subject_idx:03d}"
        features["label"] = label
        features["dataset_id"] = 2
        features["dataset_name"] = "eyesclosed"
        subject_frames.append(features)

    feature_frame = pd.concat(subject_frames, ignore_index=True)
    parquet = tmp_path / "features" / "subject_features.parquet"
    parquet.parent.mkdir(parents=True)
    feature_frame.to_parquet(parquet, engine="pyarrow", index=False)

    import eeg.training.benchmark as benchmark_module

    result_dir = tmp_path / "results"
    model_dir = tmp_path / "models"
    monkeypatch.setattr(
        benchmark_module,
        "load_features_df",
        lambda dataset, experiment, dataset_id: pd.read_parquet(parquet),
    )
    monkeypatch.setattr(benchmark_module, "results_dir", lambda d, e: result_dir)
    monkeypatch.setattr(benchmark_module, "figures_dir", lambda d, e: result_dir / "figures")
    monkeypatch.setattr(benchmark_module, "models_dir", lambda d, e: model_dir)

    result = run_benchmark(
        "eyesclosed",
        "synthetic-ci",
        models=["logistic_regression"],
        config={
            "seed": 7,
            "cv_folds": 3,
            "inner_cv_folds": 2,
            "bootstrap_iterations": 20,
            "feature_selection": {"top_k": 6, "correlation_threshold": 0.98},
        },
    )

    predictions = pd.read_csv(result_dir / "predictions.csv")
    epoch_predictions = pd.read_csv(result_dir / "epoch_predictions.csv")
    assert Path(result["benchmark_csv"]).exists()
    assert (model_dir / "logistic_regression.joblib").exists()
    assert len(predictions) == 9
    assert predictions["participant_id"].nunique() == 9
    assert len(epoch_predictions) == 18
    assert result["rows"][0]["n_oof_subjects"] == 9
