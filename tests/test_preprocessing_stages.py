"""Non-regression tests for staged preprocessing pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from config import ALL_CHANNELS, SAMPLING_RATE
from eeg.preprocessing import (
    apply_standard_montage,
    preprocess_EEG,
    stage_clean,
    stage_filtered,
    stage_raw,
)
from eeg.qc import BAND_RANGES, compute_band_power_table


def _synthetic_raw(n_channels=4, n_times=2000, flat_channel=None, ch_names=None):
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_times)) * 1e-6
    if flat_channel is not None:
        data[flat_channel, :] = 0.0
    if ch_names is None:
        ch_names = [f"EEG{i}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_RATE, ch_types=["eeg"] * n_channels)
    return mne.io.RawArray(data, info)


def _config(**prep_overrides):
    return {
        "experiment": {
            "preprocessing": {
                "freq_filter": True,
                "notch_filter": False,
                "bad_channels": True,
                "referencing": False,
                "asr": False,
                "run_ica": False,
                "AR": False,
                **prep_overrides,
            },
            "filtering": {"l_freq": 0.5, "h_freq": 40, "notch_freq": [50]},
            "bad_channels": {"flat_std": 1e-15, "noisy_z": 5.0},
        },
        "features": {"sampling_rate": SAMPLING_RATE},
    }


def test_filter_excludes_bad_channels():
    raw = _synthetic_raw(flat_channel=0)
    before = raw.get_data(picks=[0]).copy()
    filtered, meta = stage_filtered(raw, _config())
    after = filtered.get_data(picks=[0])
    assert meta["bad_channels_detected"] == ["EEG0"]
    assert "EEG0" in filtered.info["bads"]
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-20)
    good_before = raw.get_data(picks=[1])
    good_after = filtered.get_data(picks=[1])
    assert not np.allclose(good_before, good_after)


def test_bad_channels_preserved_in_meta_before_interpolation():
    raw = _synthetic_raw(flat_channel=0, ch_names=ALL_CHANNELS[:4])
    apply_standard_montage(raw)
    filtered, _ = stage_filtered(raw, _config())
    clean, meta = stage_clean(filtered, _config())
    assert meta["bad_channels"] == ["Fp1"]
    assert clean.info["bads"] == []
    assert meta["bad_channels_interpolated"] == ["Fp1"]


def test_montage_applied_in_stage_raw(monkeypatch):
    raw = _synthetic_raw(n_channels=19, ch_names=ALL_CHANNELS[:19])
    apply_standard_montage(raw)

    def _read_eeg_data(path, sfreq):
        return raw.copy()

    monkeypatch.setattr("eeg.preprocessing.read_eeg_data", _read_eeg_data)
    out, meta = stage_raw(Path("sub-001.set"), SAMPLING_RATE)
    assert meta["montage_set"] is True
    assert meta["n_eeg_channels"] == 19
    assert out.get_montage() is not None


def test_apply_standard_montage_sets_dig():
    raw = _synthetic_raw(n_channels=19, ch_names=ALL_CHANNELS[:19])
    out, meta = apply_standard_montage(raw)
    assert meta["montage_set"] is True
    montage = out.get_montage()
    assert montage is not None
    assert len(montage.dig) > 0


def test_stage_clean_order_asr_before_car():
    raw = _synthetic_raw(ch_names=ALL_CHANNELS[:4])
    apply_standard_montage(raw)
    config = _config(referencing=True, asr=True)
    call_order: list[str] = []

    mock_asr = MagicMock()
    mock_asr.fit.return_value = mock_asr
    mock_asr.transform.side_effect = lambda r: (call_order.append("asr"), r)[1]

    with patch("eeg.preprocessing.ASR", return_value=mock_asr), patch.object(
        mne.io.BaseRaw,
        "set_eeg_reference",
        autospec=True,
        side_effect=lambda self, *args, **kwargs: (call_order.append("car"), self)[1],
    ):
        _, meta = stage_clean(raw, config)

    assert call_order == ["asr", "car"]
    assert meta["asr_before_car"] is True


def test_preprocess_eeg_matches_stages():
    raw = _synthetic_raw(n_times=4000)
    call_order: list[str] = []

    def _montage(raw_in):
        call_order.append("montage")
        return raw_in, {}

    def _filtered(raw_in, config):
        call_order.append("filtered")
        return raw_in, {}

    def _ica(raw_in, config):
        call_order.append("ica")
        return raw_in, {}

    def _clean(raw_in, config, raw_before=None):
        call_order.append("clean")
        return raw_in, {}

    with patch("eeg.preprocessing.apply_standard_montage", side_effect=_montage), patch(
        "eeg.preprocessing.stage_filtered", side_effect=_filtered
    ), patch("eeg.preprocessing.stage_ica", side_effect=_ica), patch(
        "eeg.preprocessing.stage_clean", side_effect=_clean
    ), patch(
        "eeg.preprocessing.stage_epochs",
        return_value=(MagicMock(), {}),
    ):
        preprocess_EEG(
            raw,
            freq_filter=True,
            notch_filter=False,
            bad_channels=False,
            referencing=False,
            asr=False,
            run_ica=False,
            AR=False,
        )

    assert call_order == ["montage", "filtered", "clean"]


def test_compute_band_power_table_uv2_keys():
    raw_a = _synthetic_raw()
    raw_b = _synthetic_raw()
    result = compute_band_power_table(raw_a, raw_b)
    legacy_suffixes = ("_before", "_after", "_delta")
    for key in result:
        if key == "band_power":
            continue
        assert key.endswith("_uv2") or key.endswith("_log10"), f"unexpected key: {key}"
        for band in BAND_RANGES:
            for suffix in legacy_suffixes:
                assert f"{band}{suffix}" not in result
    assert result["alpha_before_uv2"] > 0
