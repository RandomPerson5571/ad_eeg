import mne
import numpy as np

from asrpy import ASR
from mne.preprocessing import ICA

import autoreject

from config import (
    ASR_CUTOFF,
    BAD_CHANNEL_FLAT_STD,
    BAD_CHANNEL_NOISY_Z,
    EPOCH_LENGTH,
    EPOCH_OVERLAP,
    ICA_N_COMPONENTS,
    ICA_RANDOM_STATE,
    NOTCH_FREQ,
    SAMPLING_RATE,
)


def detect_bad_channels(raw, flat_std=BAD_CHANNEL_FLAT_STD, noisy_z=BAD_CHANNEL_NOISY_Z):
    """Return EEG channel names that are flat or excessively noisy."""
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        return []

    data = raw.get_data(picks=picks)
    ch_names = [raw.ch_names[i] for i in picks]
    stds = np.std(data, axis=1)
    med_std = np.median(stds)
    if med_std == 0:
        med_std = 1e-15

    bads = []
    for ch, std in zip(ch_names, stds):
        if std < flat_std or std > med_std * noisy_z:
            bads.append(ch)
    return bads


def preprocess_EEG(
    eeg_raw,
    freq_filter=True,
    notch_filter=False,
    bad_channels=False,
    referencing=False,
    asr=False,
    asr_cutoff=None,
    run_ica=False,
    AR=True,
    erp=False,
    fle=True,
    verbose_plots=False,
    return_reject_log=False,
):
    if asr_cutoff is None:
        asr_cutoff = ASR_CUTOFF

    bad_channels_found = []

    if freq_filter:
        eeg_raw.filter(l_freq=0.5, h_freq=40, fir_design="firwin", filter_length="auto")

    if notch_filter and NOTCH_FREQ:
        eeg_raw.notch_filter(freqs=NOTCH_FREQ)

    if bad_channels:
        bad_channels_found = detect_bad_channels(eeg_raw)
        if bad_channels_found:
            eeg_raw.info["bads"] = list(set(eeg_raw.info.get("bads", []) + bad_channels_found))
            eeg_raw.interpolate_bads(reset_bads=True)

    if referencing:
        eeg_raw.set_eeg_reference("average")

    if asr:
        asr_model = ASR(sfreq=SAMPLING_RATE, cutoff=asr_cutoff)
        asr_model.fit(eeg_raw)
        eeg_raw = asr_model.transform(eeg_raw)

    if run_ica:
        sfreq = eeg_raw.info["sfreq"]
        seconds = eeg_raw.n_times / sfreq
        print(f"Recording length: {seconds:.2f} seconds")

        if seconds >= 30:
            raw_ica = eeg_raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
            ica = ICA(
                n_components=ICA_N_COMPONENTS,
                method="infomax",
                random_state=ICA_RANDOM_STATE,
            )
            ica.fit(raw_ica)

            eog_inds, eog_scores = ica.find_bads_eog(
                eeg_raw, ch_name=["Fp1", "Fp2"], threshold=3.0
            )
            ica.exclude = list(eog_inds)
            print("EOG indices:", eog_inds)
            if len(eog_scores) > 0:
                print("Max EOG score:", np.max(eog_scores))

            if "ECG" in eeg_raw.ch_names:
                ecg_inds, _ = ica.find_bads_ecg(eeg_raw, threshold=3.0, filterlength="auto")
                ica.exclude = list(set(ica.exclude + list(ecg_inds)))

            if verbose_plots and ica.exclude:
                ica.plot_components(picks=ica.exclude)
                ica.plot_sources(eeg_raw, picks=ica.exclude)

            ica.apply(eeg_raw)
        else:
            print("Recording too short for ICA, skipping this step.")

    eeg_clean = eeg_raw.copy()

    if erp:
        epochs = extract_erp_epochs(eeg_clean, "highstim", "lowstim")
    elif fle:
        epochs = convert_to_epochs(eeg_clean)
    else:
        raise ValueError("Either erp or fle must be enabled.")

    epochs_out = epochs
    reject_log = None
    if AR:
        ar = autoreject.AutoReject(
            n_interpolate=[1, 2, 3, 4], random_state=11, n_jobs=1, verbose=True
        )
        ar.fit(epochs[:20])
        epochs_ar, reject_log = ar.transform(epochs, return_log=True)
        n_dropped = reject_log.bad_epochs.sum()
        print(f"Dropped {n_dropped} epochs out of {len(epochs_ar)}")
        p_dropped = n_dropped / len(epochs_ar) * 100
        print(f"Percent dropped : {p_dropped}%")
        epochs_out = epochs_ar

    meta = {"bad_channels": bad_channels_found}

    if return_reject_log:
        return eeg_clean, epochs_out, reject_log, meta
    return eeg_clean, epochs_out


def extract_erp_epochs(clean_eeg, cond1, cond2, epochLowLim=-0.3, epochHiLim=0.7):
    events_from_annot, event_dict = mne.events_from_annotations(clean_eeg)
    epochs_all = mne.Epochs(
        clean_eeg,
        events_from_annot,
        tmin=epochLowLim,
        tmax=epochHiLim,
        event_id=event_dict,
        preload=True,
        event_repeated="drop",
    )
    epochs = epochs_all[cond1, cond2]
    return epochs


def convert_to_epochs(clean_eeg):
    epochs = mne.make_fixed_length_epochs(
        clean_eeg, duration=EPOCH_LENGTH, overlap=EPOCH_OVERLAP, preload=True
    )
    return epochs
