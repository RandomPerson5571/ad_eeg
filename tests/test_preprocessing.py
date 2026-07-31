import importlib
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from config import ALL_CHANNELS, SAMPLING_RATE
from util.preprocessing import convert_to_epochs, detect_bad_channels, preprocess_EEG


def _synthetic_raw(n_channels=4, n_times=2000, flat_channel=None):
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_times)) * 1e-6
    if flat_channel is not None:
        data[flat_channel, :] = 0.0
    ch_names = [f"EEG{i}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_RATE, ch_types=["eeg"] * n_channels)
    return mne.io.RawArray(data, info)


def test_detect_bad_channels_flat():
    raw = _synthetic_raw(flat_channel=0)
    bads = detect_bad_channels(raw)
    assert "EEG0" in bads


def test_preprocess_fast_mode_returns_epochs():
    raw = _synthetic_raw(n_times=4000)
    clean, epochs = preprocess_EEG(
        raw,
        freq_filter=True,
        notch_filter=False,
        bad_channels=False,
        referencing=False,
        asr=False,
        run_ica=False,
        AR=False,
        fle=True,
    )
    assert clean is not None
    data = epochs.get_data()
    assert data.ndim == 3
    assert data.shape[0] >= 1


def test_preprocess_does_not_plot_by_default():
    raw = _synthetic_raw(n_channels=19, n_times=int(35 * SAMPLING_RATE))
    raw.rename_channels({f"EEG{i}": ch for i, ch in enumerate(ALL_CHANNELS[:19])})

    mock_ica = MagicMock()
    mock_ica.exclude = [0]
    mock_ica.find_bads_eog.return_value = ([0], np.array([4.0]))
    mock_ica.find_bads_ecg.return_value = ([], np.array([]))

    with patch("util.preprocessing.ICA", return_value=mock_ica) as ica_cls:
        preprocess_EEG(
            raw,
            freq_filter=False,
            notch_filter=False,
            bad_channels=False,
            referencing=False,
            asr=False,
            run_ica=True,
            AR=False,
            fle=True,
            verbose_plots=False,
        )

    ica_cls.assert_called_once()
    mock_ica.fit.assert_called_once()
    mock_ica.plot_components.assert_not_called()
    mock_ica.plot_sources.assert_not_called()


def test_all_channels_not_mutated_after_preprocessing_import():
    expected_len = len(ALL_CHANNELS)
    importlib.reload(importlib.import_module("util.preprocessing"))
    assert len(ALL_CHANNELS) == expected_len
    assert "Fp1" in ALL_CHANNELS
