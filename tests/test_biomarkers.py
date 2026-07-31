import importlib
import sys

import mne
import numpy as np
import pandas as pd
import pytest

from biomarkers import (
    compute_band_power,
    compute_connectivity,
    compute_regional_complexity,
    normalize_signal,
)
from biomarkers.complexity import lempel_ziv_complexity
from classifier_models.train_utils import validate_feature_schema
from config import ALL_CHANNELS, FEATURE_COLUMNS, REGIONAL_CHANNELS, SAMPLING_RATE, parquet_path
from util.extract_features import extract_eeg_features


def _synthetic_epochs(n_epochs=3, n_channels=4, n_times=500, ch_names=None):
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_epochs, n_channels, n_times))
    ch_names = ch_names or [f"EEG{i}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_RATE, ch_types=["eeg"] * n_channels)
    return mne.EpochsArray(data, info=info)


def test_band_power_ratios_are_scalars():
    epoch_mne = _synthetic_epochs(n_epochs=1)
    features = compute_band_power(epoch_mne, target_channels=None)

    for key, value in features.items():
        assert np.isscalar(value), f"{key} should be scalar, got {type(value)}"

    for key in ("theta_alpha_ratio", "theta_beta_ratio", "slow_fast_ratio"):
        assert key in features
        assert np.isfinite(features[key])


def test_all_channels_not_mutated_on_import():
    expected_len = len(ALL_CHANNELS)
    assert "Fp1" in ALL_CHANNELS

    if "util.preprocessing" in sys.modules:
        importlib.reload(sys.modules["util.preprocessing"])
    else:
        import util.preprocessing  # noqa: F401

    assert len(ALL_CHANNELS) == expected_len
    assert "Fp1" in ALL_CHANNELS


def test_connectivity_returns_scalar_bands():
    epochs = _synthetic_epochs(n_epochs=3)
    features = compute_connectivity(epochs)

    assert set(features) == {"theta_wpli", "alpha_wpli"}
    for key, value in features.items():
        assert np.isscalar(value), f"{key} should be scalar"
        assert np.isfinite(value), f"{key} should be finite"


def test_normalize_signal_zero_std():
    signal = np.ones(100)
    normalized = normalize_signal(signal)
    assert normalized.shape == signal.shape
    assert np.allclose(normalized, 0.0)


def test_regional_complexity_keys_and_scalars():
    posterior = REGIONAL_CHANNELS["posterior"]
    epoch = np.random.default_rng(0).standard_normal((len(ALL_CHANNELS), 500))
    features = compute_regional_complexity(epoch, ALL_CHANNELS, posterior, region_name="posterior")

    assert set(features) == {"lzc_posterior", "mse_posterior"}
    for key, value in features.items():
        assert np.isscalar(value), f"{key} should be scalar"
        assert np.isfinite(value), f"{key} should be finite"


def test_regional_complexity_uses_subset():
    posterior = REGIONAL_CHANNELS["posterior"]
    n_times = 500
    t = np.arange(n_times) / SAMPLING_RATE
    epoch = np.random.default_rng(7).standard_normal((len(ALL_CHANNELS), n_times)) * 0.01
    posterior_idx = [ALL_CHANNELS.index(ch) for ch in posterior]
    epoch[posterior_idx, :] = np.sin(2 * np.pi * 10 * t)

    regional = compute_regional_complexity(epoch, ALL_CHANNELS, posterior, region_name="posterior")
    global_lzc = lempel_ziv_complexity(normalize_signal(epoch.mean(axis=0)))

    assert regional["lzc_posterior"] != global_lzc


def test_extract_eeg_features_schema():
    ch_names = ALL_CHANNELS
    n_times = int(SAMPLING_RATE * 4)
    data = np.random.default_rng(1).standard_normal((2, len(ch_names), n_times))
    features = extract_eeg_features(data, ch_names=ch_names)

    assert len(features) == 2
    for epoch_features in features:
        for key in FEATURE_COLUMNS:
            assert key in epoch_features, f"missing {key}"
            assert np.isscalar(epoch_features[key]), f"{key} should be scalar"


def test_parquet_path_per_dataset():
    assert parquet_path(2) != parquet_path(3)
    assert parquet_path(2).endswith("features_dataset2.parquet")


def test_load_features_df_per_dataset(tmp_path, monkeypatch):
    import config
    import util.io as io_module

    monkeypatch.setattr(config, "PARQUET_DIR", str(tmp_path))

    df2 = pd.DataFrame({"lzc_posterior": [0.5], "mse_posterior": [1.0], "participant_id": ["sub-001"]})
    path2 = config.parquet_path(2)
    df2.to_parquet(path2, engine="pyarrow")

    loaded = io_module.load_features_df(dataset_id=2)
    assert len(loaded) == 1
    assert loaded.iloc[0]["lzc_posterior"] == 0.5


def test_validate_feature_schema_detects_stale_columns():
    df = pd.DataFrame({"lzc": [0.5], "mse_mean": [1.0]})
    with pytest.raises(ValueError, match="Re-ingest required"):
        validate_feature_schema(df)


def test_theta_dominant_sine_has_higher_theta_alpha_ratio():
    t = np.arange(2000) / SAMPLING_RATE
    theta_signal = np.sin(2 * np.pi * 6 * t)
    alpha_signal = np.sin(2 * np.pi * 10 * t)
    info = mne.create_info(ch_names=["Cz"], sfreq=SAMPLING_RATE, ch_types=["eeg"])

    theta_epoch = mne.EpochsArray(theta_signal[np.newaxis, np.newaxis, :], info)
    alpha_epoch = mne.EpochsArray(alpha_signal[np.newaxis, np.newaxis, :], info)

    theta_features = compute_band_power(theta_epoch, target_channels=None)
    alpha_features = compute_band_power(alpha_epoch, target_channels=None)

    assert theta_features["theta_alpha_ratio"] > alpha_features["theta_alpha_ratio"]
