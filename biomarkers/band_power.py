import numpy as np

def compute_band_power(epoch_mne, target_channels):
    eps = 1e-12
    
    psd, freqs = epoch_mne.compute_psd(
    method="welch",
    fmin=1,
    fmax=40,
    n_fft=256,
    picks=target_channels,
    verbose=False
    ).get_data(return_freqs=True)

    psd = psd[0]

    delta_power = psd[:, (freqs >= 1) & (freqs < 4)].mean(axis=1)
    theta_power = psd[:, (freqs >= 4) & (freqs < 8)].mean(axis=1)
    alpha_power = psd[:, (freqs >= 8) & (freqs < 13)].mean(axis=1)
    beta_power  = psd[:, (freqs >= 13) & (freqs < 30)].mean(axis=1)

    
    theta_mean = theta_power.mean()
    delta_mean = delta_power.mean()
    alpha_mean = alpha_power.mean()
    beta_mean = beta_power.mean()

    theta_alpha_ratio = (theta_mean / (alpha_power + eps))
    theta_beta_ratio = (theta_mean / (beta_power + eps))
    slow_fast_ratio = (theta_mean + delta_mean) / (alpha_mean + beta_mean + eps)

    total_mean = delta_mean + theta_mean + alpha_mean + beta_mean


    rel_alpha = alpha_mean / (total_mean + eps)
    rel_beta = beta_mean / (total_mean + eps)
    rel_theta = theta_mean / (total_mean + eps)
    rel_delta = delta_mean / (total_mean + eps)

    alpha_band = (freqs >= 8) & (freqs <= 13)
    mean_spectrum = psd.mean(axis=0)
    if np.any(alpha_band):
        peak_id = np.argmax(mean_spectrum[alpha_band])
        alpha_peak_freq = freqs[alpha_band][peak_id]
    else:
        alpha_peak_freq = np.nan

    band_power_features = {
        "rel_alpha": rel_alpha,
        "rel_beta": rel_beta,
        "rel_theta": rel_theta,
        "rel_delta": rel_delta,
        "alpha_peak_freq": alpha_peak_freq,
        "theta_alpha_ratio": theta_alpha_ratio,
        "theta_beta_ratio": theta_beta_ratio,
        "slow_fast_ratio": slow_fast_ratio
    }

    return band_power_features