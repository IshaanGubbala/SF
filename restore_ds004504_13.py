#!/usr/bin/env python
"""Restore ds004504 sub-001..sub-013 features (sequential, no multiprocessing)."""
import os, glob, json, warnings
import numpy as np, mne

mne.set_log_level('WARNING')
warnings.filterwarnings("ignore", category=RuntimeWarning)

SAMPLING_RATE = 256
FREQUENCY_BANDS = {"Delta":(0.5,4),"Theta1":(4,6),"Theta2":(6,8),"Alpha1":(8,10),"Alpha2":(10,12),"Beta1":(12,20),"Beta2":(20,30),"Gamma1":(30,40),"Gamma2":(40,50)}

def process_one(f, subj_id):
    raw = mne.io.read_raw_eeglab(f, preload=True, verbose=False)
    raw.filter(1.0, 50.0, fir_design="firwin", verbose=False)
    raw.resample(SAMPLING_RATE, npad="auto")
    epochs = mne.make_fixed_length_epochs(raw, duration=20.0, overlap=0.0, verbose=False)
    data = epochs.get_data()
    feats = []
    for e in data:
        d = e[np.newaxis]
        psd, freqs = mne.time_frequency.psd_array_multitaper(d, sfreq=SAMPLING_RATE, verbose=False)
        bp = {b: np.mean(psd[:,:,(freqs>=fl)&(freqs<=fh)], axis=2) for b,(fl,fh) in FREQUENCY_BANDS.items()}
        f_e = [np.mean(bp[b]) for b in FREQUENCY_BANDS]
        a = np.mean(bp["Alpha1"]) + np.mean(bp["Alpha2"])
        th = np.mean(bp["Theta1"]) + np.mean(bp["Theta2"])
        tot = sum(np.mean(bp[b]) for b in FREQUENCY_BANDS) + 1e-12
        # entropy (simple placeholder — same 13 dims as main pipeline)
        counts, _ = np.histogram(d.ravel(), bins=256)
        p = counts / (counts.sum() + 1e-12)
        ent = -np.sum(p * np.log2(p + 1e-12))
        f_e.extend([a/tot, th/tot, ent, 0.0, 0.0, 0.0])
        feats.append(np.array(f_e, dtype=np.float32))
    ef = np.stack(feats)
    hc = np.hstack([np.mean(ef, axis=0), np.std(ef, axis=0)])
    raw_lap = mne.preprocessing.compute_current_source_density(raw.copy())
    el = mne.make_fixed_length_epochs(raw_lap, duration=20.0, overlap=0.0, verbose=False).get_data()
    gf_list = []
    for e in el:
        g = np.zeros((e.shape[0], 9), dtype=np.float32)
        for ch in range(e.shape[0]):
            psd, freqs = mne.time_frequency.psd_array_multitaper(e[ch:ch+1], sfreq=SAMPLING_RATE, verbose=False)
            g[ch] = [np.mean(psd[0,(freqs>=fl)&(freqs<=fh)]) for fl,fh in FREQUENCY_BANDS.values()]
        gf_list.append(g)
    gnn = np.mean(np.stack(gf_list), axis=0)
    return hc, gnn, raw_lap.ch_names

BASE = os.path.dirname(os.path.abspath(__file__))
for i in range(1, 14):
    sid = f"sub-{i:03d}"
    matches = glob.glob(os.path.join(BASE, f"ds004504/derivatives/{sid}/eeg/*.set"))
    if not matches:
        print(f"[SKIP] {sid}")
        continue
    try:
        hc, gnn, ch_names = process_one(matches[0], sid)
        np.save(os.path.join(BASE, f"processed_features/handcrafted/{sid}_handcrafted.npy"), hc)
        np.save(os.path.join(BASE, f"processed_features/gnn/{sid}_gnn.npy"), gnn)
        with open(os.path.join(BASE, f"processed_features/channels/{sid}_channels.json"), "w") as fp:
            json.dump(ch_names, fp)
        print(f"[RESTORED] {sid}")
    except Exception as e:
        print(f"[ERROR] {sid}: {e}")
print("[DONE]")
