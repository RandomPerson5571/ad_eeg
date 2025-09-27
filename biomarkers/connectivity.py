import mne
from mne_connectivity import spectral_connectivity_epochs
import numpy as np

def compute_connectivity(epochs, fmin=4, fmax=12):
    con = spectral_connectivity_epochs(
    data=epochs,             
    method='wpli',
    mode='multitaper',
    sfreq=epochs.info['sfreq'],
    fmin=fmin,
    fmax=fmax,
    faverage=False,
    verbose=False
    )

    conn_data = con.get_data(output='dense')

    theta_wpli = np.mean(conn_data[..., 0])
    alpha_wpli = np.mean(conn_data[..., 1])

    connectivity_features = {
        "theta_wpli": theta_wpli,
        "alpha_wpli": alpha_wpli
    }

    return connectivity_features