import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

from config import SAMPLING_RATE

def plot_power_spectrum(eeg_signal, target_channels):

    if eeg_signal.ndim == 1:
        eeg_signal = eeg_signal[np.newaxis, :]  # make 2D

    plt.figure(figsize=(10, 6))
    
    for ch_idx, ch_data in enumerate(eeg_signal):
        f, psd = welch(ch_data, SAMPLING_RATE, nperseg=SAMPLING_RATE*2)
        plt.semilogy(f, psd, label=target_channels[ch_idx] if target_channels else f"Ch{ch_idx+1}")
    
    plt.title("EEG Power Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (V²/Hz)")
    plt.xlim(0, 45)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_eeg(epoch_mne, target_channels):

    epoch_mne.plot(
        picks=target_channels,
        scalings='auto',
        title='EEG Epoch',
        show=True,
        block=True
    )