import warnings

import numpy as np
from mne_connectivity import spectral_connectivity_epochs


def _band_wpli(epochs, fmin, fmax):
    n_channels = len(epochs.ch_names)
    if n_channels < 2:
        return 0.0
    # Request each undirected edge exactly once. Averaging a dense matrix would
    # otherwise include the undefined diagonal and count symmetric edges twice.
    sources, targets = np.triu_indices(n_channels, k=1)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="There were no Annotations stored.*",
            category=RuntimeWarning,
        )
        con = spectral_connectivity_epochs(
            data=epochs,
            method="wpli",
            indices=(sources, targets),
            mode="multitaper",
            sfreq=epochs.info["sfreq"],
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            verbose=False,
        )
    values = np.asarray(con.get_data()).reshape(len(sources), -1)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else 0.0


def compute_connectivity(epochs):
    theta_wpli = _band_wpli(epochs, 4, 8)
    alpha_wpli = _band_wpli(epochs, 8, 13)

    return {
        "theta_wpli": theta_wpli,
        "alpha_wpli": alpha_wpli,
    }
