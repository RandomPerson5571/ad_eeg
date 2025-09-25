import mne

import numpy as np

from asrpy import ASR
from mne.preprocessing import ICA
from config import SAMPLING_RATE, EPOCH_LENGTH, EPOCH_OVERLAP, HP_FILTER, LP_FILTER, NOTCH_FREQ

def preprocess_EEG(eeg_raw):
    eeg_raw.set_montage('standard_1020')

    original = eeg_raw.copy() # for testing

    eeg_raw.notch_filter(freqs=[50, 100, 150])
    eeg_raw.filter(l_freq=1., h_freq=None, fir_design='firwin')

    # artifact subspace removal (ASR)
    # basically identifies artifacts

    asr = ASR(sfreq=eeg_raw.info['sfreq'], cutoff=10)
    asr.fit(eeg_raw)
    eeg_raw = asr.transform(eeg_raw)

    # eeg.raw.plot(n_channels=19, duration=10, scalings='auto', show=True, title='After ASR')
    # original.plot(n_channels=19, duration=10, scalings='auto', show=True, title='Before ASR')

    # independant component analysis (ICA)
    # removes the identified artifacts in ASR

    seconds = eeg_raw.n_times / eeg_raw.info['sfreq']
    print(f"Recording length: {seconds:.2f} seconds")

    if seconds >= 30:

        eeg_raw.resample(256, npad="auto")

        ica = ICA(n_components=0.95, method="infomax", random_state=97)
        ica.fit(eeg_raw)

        eog_inds, eog_scores = ica.find_bads_eog(eeg_raw, ch_name=['Fp1', 'Fp2'], threshold=3.0)
        print("EOG indices:", eog_inds)
        print("EOG scores:", eog_scores)
        print("Max EOG score:", np.max(eog_scores))
        ica.exclude.extend(eog_inds)

        if "ECG" in eeg_raw.ch_names:
            ecg_inds, ecg_scores = ica.find_bads_ecg(eeg_raw, threshold=3.0, filterlength='auto')
            ica.exclude.extend(ecg_inds)

        if ica.exclude:
            ica.plot_components(picks=ica.exclude)
            ica.plot_sources(eeg_raw, picks=ica.exclude)

        ica.apply(eeg_raw)
    else:
        print("Recording too short for ICA, skipping this step.")

    # clean_eeg_path = os.path.join(base_dir, "..", "EEG_clean_data", "dataset2", "cleaned_sub-001_raw.fif")
    # eeg_raw.save(clean_eeg_path, overwrite=True)

    # epoching

    eeg_raw.set_eeg_reference('average')
    epochs = mne.make_fixed_length_epochs(eeg_raw, duration=EPOCH_LENGTH, overlap=EPOCH_OVERLAP, preload=True)

    return epochs