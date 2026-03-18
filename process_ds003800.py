#!/usr/bin/env python
"""
process_ds003800.py

Processes ds003800 (all-AD resting EEG) into processed_features/ with a
"b" prefix on subject IDs to avoid collision with ds004504 (both use sub-001..sub-013).

Output IDs: bsub001, bsub002, ... bsub013
Labels: all → 1 (Alzheimer's Disease)
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import mne
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from numba import njit

mne.set_log_level('WARNING')
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DS003800_PATH   = os.path.join(BASE_DIR, "ds003800")
PARTICIPANTS_TSV= os.path.join(BASE_DIR, "ds003800", "participants.tsv")

BASE_OUTPUT_DIR = "processed_features"
HANDCRAFTED_DIR = os.path.join(BASE_OUTPUT_DIR, "handcrafted")
GNN_DIR         = os.path.join(BASE_OUTPUT_DIR, "gnn")
CHANNELS_DIR    = os.path.join(BASE_OUTPUT_DIR, "channels")
for d in [BASE_OUTPUT_DIR, HANDCRAFTED_DIR, GNN_DIR, CHANNELS_DIR]:
    os.makedirs(d, exist_ok=True)

SAMPLING_RATE = 256
FREQUENCY_BANDS = {
    "Delta":  (0.5, 4), "Theta1": (4, 6),   "Theta2": (6, 8),
    "Alpha1": (8, 10),  "Alpha2": (10, 12),
    "Beta1":  (12, 20), "Beta2":  (20, 30),
    "Gamma1": (30, 40), "Gamma2": (40, 50),
}


def compute_band_powers(data, sfreq):
    psd, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=sfreq, verbose=False)
    return {band: np.mean(psd[:, :, (freqs >= fmin) & (freqs <= fmax)], axis=2)
            for band, (fmin, fmax) in FREQUENCY_BANDS.items()}


def compute_shannon_entropy(data):
    def entropy_fn(x):
        counts, _ = np.histogram(x, bins=256)
        p = counts / np.sum(counts)
        return -np.sum(p * np.log2(p + 1e-12))
    return np.apply_along_axis(entropy_fn, 1, data)


@njit
def compute_hjorth_numba(data):
    n_epochs, n_channels, n_times = data.shape
    activities = np.empty(n_epochs, dtype=np.float64)
    mobilities = np.empty(n_epochs, dtype=np.float64)
    complexities = np.empty(n_epochs, dtype=np.float64)
    for i in range(n_epochs):
        var0 = np.empty(n_channels, dtype=np.float64)
        for j in range(n_channels):
            s = 0.0
            for k in range(n_times): s += data[i, j, k]
            mv = s / n_times
            s2 = 0.0
            for k in range(n_times):
                d = data[i, j, k] - mv; s2 += d * d
            var0[j] = max(s2 / n_times, 1e-12)
        mobility = np.empty(n_channels, dtype=np.float64)
        for j in range(n_channels):
            s2 = 0.0
            for k in range(1, n_times):
                d = data[i, j, k] - data[i, j, k-1]; s2 += d * d
            mobility[j] = np.sqrt(s2 / ((n_times - 1) * var0[j]))
        complexity = np.empty(n_channels, dtype=np.float64)
        for j in range(n_channels):
            s2 = 0.0
            for k in range(2, n_times):
                d2 = (data[i, j, k] - data[i, j, k-1]) - (data[i, j, k-1] - data[i, j, k-2])
                s2 += d2 * d2
            complexity[j] = np.sqrt(max(s2, 1e-12) / (n_times - 2)) / (mobility[j] + 1e-12)
        activities[i] = np.mean(var0)
        mobilities[i] = np.mean(mobility)
        complexities[i] = np.mean(complexity)
    return activities, mobilities, complexities


def extract_features_epoch(epoch_data):
    data = epoch_data[np.newaxis, :, :]
    bp = compute_band_powers(data, SAMPLING_RATE)
    feats = [np.mean(bp[b]) for b in FREQUENCY_BANDS]
    alpha = np.mean(bp["Alpha1"]) + np.mean(bp["Alpha2"])
    theta = np.mean(bp["Theta1"]) + np.mean(bp["Theta2"])
    total = sum(np.mean(bp[b]) for b in FREQUENCY_BANDS) + 1e-12
    feats.extend([alpha / total, theta / total])
    feats.append(np.mean(compute_shannon_entropy(data)))
    act, mob, comp = compute_hjorth_numba(data)
    feats.extend([np.mean(act), np.mean(mob), np.mean(comp)])
    return np.array(feats, dtype=np.float32)


def extract_channel_features_gnn(epoch_data, sfreq):
    n_ch = epoch_data.shape[0]
    feats = np.zeros((n_ch, len(FREQUENCY_BANDS)), dtype=np.float32)
    for ch in range(n_ch):
        ts = epoch_data[ch, :]
        psd, freqs = mne.time_frequency.psd_array_multitaper(
            ts[np.newaxis, :], sfreq=sfreq, verbose=False)
        feats[ch] = [np.mean(psd[0, (freqs >= fmin) & (freqs <= fmax)])
                     for fmin, fmax in FREQUENCY_BANDS.values()]
    return feats


def process_subject(args):
    filepath, subj_id = args
    try:
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        raw.filter(1.0, 50.0, fir_design="firwin", verbose=False)
        raw.resample(SAMPLING_RATE, npad="auto")

        epochs = mne.make_fixed_length_epochs(raw, duration=20.0, overlap=0.0, verbose=False)
        data = epochs.get_data()
        if data.size == 0:
            raise ValueError("No epochs extracted")

        epoch_feats = np.array([extract_features_epoch(e) for e in data])
        handcrafted_global = np.hstack([np.mean(epoch_feats, axis=0),
                                         np.std(epoch_feats, axis=0)])

        raw_lap = raw.copy()
        raw_lap = mne.preprocessing.compute_current_source_density(raw_lap)
        epochs_lap = mne.make_fixed_length_epochs(raw_lap, duration=20.0, overlap=0.0, verbose=False)
        data_lap = epochs_lap.get_data()
        gnn_feats = np.array([extract_channel_features_gnn(e, SAMPLING_RATE) for e in data_lap])
        gnn_aggregated = np.mean(gnn_feats, axis=0)
        ch_names = raw_lap.ch_names

        return handcrafted_global, gnn_aggregated, ch_names
    except Exception as e:
        print(f"[ERROR] {subj_id}: {e}")
        return None, None, None


def main():
    set_files = sorted(glob.glob(
        os.path.join(DS003800_PATH, "**", "*_task-Rest_eeg.set"), recursive=True))
    print(f"[INFO] Found {len(set_files)} ds003800 Rest EEG files")

    tasks = []
    for f in set_files:
        raw_id = os.path.basename(f).split('_')[0]    # "sub-001"
        num = int(raw_id.split('-')[1])
        subj_id = f"bsub{num:03d}"                    # "bsub001"
        tasks.append((f, subj_id))

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(
            executor.map(process_subject, tasks), total=len(tasks)))

    saved = 0
    for (_, subj_id), (hc, gnn, ch_names) in zip(tasks, results):
        if hc is None:
            continue
        np.save(os.path.join(HANDCRAFTED_DIR, f"{subj_id}_handcrafted.npy"), hc)
        np.save(os.path.join(GNN_DIR,         f"{subj_id}_gnn.npy"), gnn)
        with open(os.path.join(CHANNELS_DIR, f"{subj_id}_channels.json"), "w") as fp:
            json.dump(ch_names, fp)
        saved += 1

    print(f"[DONE] Saved {saved} ds003800 subjects with 'bsub' prefix.")


if __name__ == "__main__":
    main()
