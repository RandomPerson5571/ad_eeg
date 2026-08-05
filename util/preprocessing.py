"""Backward-compat shim — use eeg.preprocessing instead."""

import warnings

from eeg.preprocessing import (  # noqa: F401
    convert_to_epochs,
    detect_bad_channels,
    preprocess_EEG,
    preprocess_subject,
)

warnings.warn(
    "util.preprocessing is deprecated; import from eeg.preprocessing",
    DeprecationWarning,
    stacklevel=2,
)
