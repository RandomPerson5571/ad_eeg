import matplotlib.pyplot as plt

def plot_power_spectrum(epoch_mne, target_channels):

    psd, freqs = epoch_mne.compute_psd(
        method="welch",
        fmin=1,
        fmax=40,
        n_fft=256,
        picks=target_channels,
        verbose=False
    ).get_data(return_freqs=True)

    psd = psd[0]

    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, psd.T)
    plt.title("Power Spectral Density (PSD)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.xlim(1, 40)
    plt.grid()
    plt.show()

def plot_eeg(epoch_mne, target_channels):

    epoch_mne.plot(
        picks=target_channels,
        scalings='auto',
        title='EEG Epoch',
        show=True,
        block=True
    )