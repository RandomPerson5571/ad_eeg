from mne_connectivity import spectral_connectivity_epochs
import numpy as np


def _band_wpli(epochs, fmin, fmax):
    con = spectral_connectivity_epochs(
        data=epochs,
        method="wpli",
        mode="multitaper",
        sfreq=epochs.info["sfreq"],
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose=False,
    )
    return float(np.mean(con.get_data(output="dense")))


def compute_connectivity(epochs):
    theta_wpli = _band_wpli(epochs, 4, 8)
    alpha_wpli = _band_wpli(epochs, 8, 13)

    return {
        "theta_wpli": theta_wpli,
        "alpha_wpli": alpha_wpli,
    }
