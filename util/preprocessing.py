import mne

import numpy as np

from asrpy import ASR
from mne.preprocessing import ICA
from config import SAMPLING_RATE, EPOCH_LENGTH, EPOCH_OVERLAP, NOTCH_FREQ, ALL_CHANNELS, TARGET_CHANNELS

import autoreject

def preprocess_EEG(eeg_raw, freq_filter=True, notch_filter=False, asr=False, asr_cutoff=10, referencing=False, ar=True, erp=False, fle=True):
    rm_ch = ALL_CHANNELS
    for target_ch in TARGET_CHANNELS:
        rm_ch.remove(target_ch)
    
    # remove unwanted channels
    eeg_raw.drop_channels(rm_ch)

    # filter the data
    if freq_filter:
        eeg_raw.filter(l_freq=0.5, h_freq=None, fir_design='firwin')
    
    if notch_filter:
        eeg_raw.notch_filter(freqs=NOTCH_FREQ)

    # artificial subspace removal
    if asr:
        asr = ASR(sfreq=SAMPLING_RATE, cutoff=asr_cutoff)
        asr.fit(eeg_raw)
        eeg_raw = asr.transform(eeg_raw)


    # seconds = eeg_raw.n_times / SAMPLING_RATE
    # print(f"Recording length: {seconds:.2f} seconds")

    # if seconds >= 30:

    #     eeg_raw.resample(256, npad="auto")

    #     ica = ICA(n_components=0.95, method="infomax", random_state=97)
    #     ica.fit(eeg_raw)

    #     eog_inds, eog_scores = ica.find_bads_eog(eeg_raw, ch_name=['Fp1', 'Fp2'], threshold=3.0)
    #     print("EOG indices:", eog_inds)
    #     print("EOG scores:", eog_scores)
    #     print("Max EOG score:", np.max(eog_scores))
    #     ica.exclude.extend(eog_inds)

    #     if "ECG" in eeg_raw.ch_names:
    #         ecg_inds, ecg_scores = ica.find_bads_ecg(eeg_raw, threshold=3.0, filterlength='auto')
    #         ica.exclude.extend(ecg_inds)

    #     if ica.exclude:
    #         ica.plot_components(picks=ica.exclude)
    #         ica.plot_sources(eeg_raw, picks=ica.exclude)

    #     ica.apply(eeg_raw)
    # else:
    #     print("Recording too short for ICA, skipping this step.")

    # clean_eeg_path = os.path.join(base_dir, "..", "EEG_clean_data", "dataset2", "cleaned_sub-001_raw.fif")
    # eeg_raw.save(clean_eeg_path, overwrite=True)

    if referencing:
        eeg_raw.set_eeg_reference('average')

    eeg_clean = eeg_raw.copy() # copy clean eeg

    if erp:
        epochs = extract_erp_epochs(eeg_clean, "highstim", "lowstim")
    
    if fle:
        epochs = convert_to_epochs(eeg_clean)

    ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,n_jobs=1, verbose=True)
    ar.fit(epochs[:20])
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)

    n_dropped = reject_log.bad_epochs.sum()
    print(f"Dropped {n_dropped} epochs out of {len(epochs_ar)}")
    p_retained = n_dropped/len(epochs_ar)
    print(f"Percent retained : {p_retained}")

    return eeg_clean, epochs_ar # clean eeg & epochs

def extract_erp_epochs(clean_eeg, cond1, cond2, epochLowLim=-0.3, epochHiLim=0.7):
    events_from_annot, event_dict = mne.events_from_annotations(clean_eeg)
    epochs_all = mne.Epochs(clean_eeg, events_from_annot, tmin=epochLowLim, tmax=epochHiLim, event_id=event_dict, preload=True, event_repeated='drop')
    epochs = epochs_all[cond1, cond2]

def convert_to_epochs(clean_eeg):
    epochs = mne.make_fixed_length_epochs(clean_eeg, duration=EPOCH_LENGTH, overlap=EPOCH_OVERLAP, preload=True)

    return epochs