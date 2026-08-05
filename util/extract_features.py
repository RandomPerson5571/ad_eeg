"""Backward-compat shim — use eeg.features instead."""

import warnings

from eeg.features import connectivity_for_epochs, extract_eeg_features, extract_from_epochs  # noqa: F401

warnings.warn(
    "util.extract_features is deprecated; import from eeg.features",
    DeprecationWarning,
    stacklevel=2,
)
