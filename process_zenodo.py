#!/usr/bin/env python
"""
process_zenodo.py

Processes Zenodo 4316608 HD-EEG dataset (.mat files) into the same
processed_features/ format used by the main pipeline.

Label mapping (binary, matching existing dataset):
  HC  (Healthy Control)          → 0
  SCD (Subjective Cognitive Decline) → excluded by default (see INCLUDE_SCD)
  MCI (Mild Cognitive Impairment)    → excluded by default (see INCLUDE_MCI)
  AD  (Alzheimer's Disease)         → 1

Each .mat file contains:
  - Category_1_Segment1/2/3: (257, 150000) float64 — 256 EEG ch + 1 ref
  - samplingRate: scalar (250 Hz)
  - Good_Regions: HC only — artifact-free region markers

Output: processed_features/handcrafted/<subj>_handcrafted.npy
        processed_features/gnn/<subj>_gnn.npy
        processed_features/channels/<subj>_channels.json
"""

import os
import glob
import json
import warnings
import numpy as np
import scipy.io
import mne
from numba import njit

mne.set_log_level('WARNING')
warnings.filterwarnings("ignore", category=RuntimeWarning)

ZENODO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zenodo_sample")
BASE_OUTPUT_DIR = "processed_features"
HANDCRAFTED_DIR = os.path.join(BASE_OUTPUT_DIR, "handcrafted")
GNN_DIR         = os.path.join(BASE_OUTPUT_DIR, "gnn")
CHANNELS_DIR    = os.path.join(BASE_OUTPUT_DIR, "channels")

SAMPLING_RATE = 256   # resample to match main pipeline
ORIG_SRATE    = 250   # Zenodo dataset native sampling rate

FREQUENCY_BANDS = {
    "Delta":  (0.5, 4), "Theta1": (4, 6),   "Theta2": (6, 8),
    "Alpha1": (8, 10),  "Alpha2": (10, 12),
    "Beta1":  (12, 20), "Beta2":  (20, 30),
    "Gamma1": (30, 40), "Gamma2": (40, 50),
}

# Label mapping — exclude SCD/MCI from binary training by default
# Set to True to include them as AD (class 1)
INCLUDE_SCD = False
INCLUDE_MCI = False

LABEL_MAP = {
    "HC":  0,
    "SCD": 1 if INCLUDE_SCD else None,
    "MCI": 1 if INCLUDE_MCI else None,
    "AD":  1,
}

# File → (subject_id, class_label) mapping
# Files are named <subj_id>_<CLASS>.mat (after our renaming on download)
def discover_files(zenodo_dir):
    tasks = []
    for f in sorted(glob.glob(os.path.join(zenodo_dir, "*.mat"))):
        fname = os.path.basename(f)
        # Expected pattern: <subj>_<LABEL>.mat  e.g. S055_HC.mat
        parts = fname.replace(".mat", "").split("_")
        if len(parts) < 2:
            print(f"[SKIP] Cannot parse label from {fname}")
            continue
        subj_id = parts[0]
        class_key = parts[-1].upper()
        label = LABEL_MAP.get(class_key, None)
        if label is None:
            print(f"[SKIP] {fname} — class '{class_key}' excluded or unknown")
            continue
        tasks.append((f, subj_id, label))
    return tasks


def load_eeg_from_mat(filepath):
    """Load all Category_1_Segment* arrays and concatenate along time axis."""
    try:
        mat = scipy.io.loadmat(filepath)
    except Exception as e:
        raise RuntimeError(f"loadmat failed: {e}")

    sr = float(mat["samplingRate"].flat[0])
    segments = []
    for key in sorted(mat.keys()):
        if key.startswith("Category_1_Segment"):
            seg = mat[key].astype(np.float64)  # (257, T)
            segments.append(seg)

    if not segments:
        raise ValueError("No Category_1_Segment* found in .mat file")

    data = np.concatenate(segments, axis=1)   # (257, total_T)
    data = data[:256, :]                       # drop channel 257 (ref/event)
    return data, sr


def make_mne_raw(data, sfreq):
    """Wrap numpy array as MNE RawArray with generic channel names."""
    n_ch = data.shape[0]
    ch_names = [f"EEG{i:03d}" for i in range(n_ch)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)  # assume µV input
    return raw


# ── Feature extraction (shared with process_server.py) ────────────────────────

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
    activities   = np.empty(n_epochs, dtype=np.float64)
    mobilities   = np.empty(n_epochs, dtype=np.float64)
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
        activities[i]   = np.mean(var0)
        mobilities[i]   = np.mean(mobility)
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


def process_subject(filepath, subj_id, label):
    print(f"[INFO] Processing {subj_id} (label={label}) ...")
    try:
        raw_data, orig_sr = load_eeg_from_mat(filepath)
        raw = make_mne_raw(raw_data, orig_sr)
        raw.filter(1.0, 50.0, fir_design="firwin", verbose=False)
        raw.resample(SAMPLING_RATE, npad="auto")

        epochs = mne.make_fixed_length_epochs(raw, duration=20.0, overlap=0.0, verbose=False)
        data = epochs.get_data()
        if data.size == 0:
            raise ValueError("No epochs extracted")

        # Handcrafted features
        epoch_feats = np.array([extract_features_epoch(e) for e in data])
        handcrafted_global = np.hstack([np.mean(epoch_feats, axis=0),
                                         np.std(epoch_feats, axis=0)])

        # GNN features — apply CSD Laplacian (MNE requires 3D montage for 256-ch EGI)
        # Fall back to standard montage if CSD fails
        try:
            raw_lap = raw.copy()
            raw_lap = mne.preprocessing.compute_current_source_density(raw_lap)
        except Exception:
            raw_lap = raw.copy()  # skip CSD if montage unavailable

        epochs_lap = mne.make_fixed_length_epochs(raw_lap, duration=20.0, overlap=0.0, verbose=False)
        data_lap = epochs_lap.get_data()
        gnn_feats = np.array([extract_channel_features_gnn(e, SAMPLING_RATE) for e in data_lap])
        gnn_aggregated = np.mean(gnn_feats, axis=0)
        ch_names = raw_lap.ch_names

        return handcrafted_global, gnn_aggregated, label, ch_names
    except Exception as e:
        print(f"[ERROR] {subj_id}: {e}")
        return None, None, None, None


def main():
    tasks = discover_files(ZENODO_DIR)
    print(f"[INFO] Found {len(tasks)} subjects to process")

    for filepath, subj_id, label in tasks:
        hc_global, gnn_agg, lbl, ch_names = process_subject(filepath, subj_id, label)
        if hc_global is None:
            continue
        np.save(os.path.join(HANDCRAFTED_DIR, f"{subj_id}_handcrafted.npy"), hc_global)
        np.save(os.path.join(GNN_DIR, f"{subj_id}_gnn.npy"), gnn_agg)
        with open(os.path.join(CHANNELS_DIR, f"{subj_id}_channels.json"), "w") as fp:
            json.dump(ch_names, fp)
        print(f"[DONE] Saved features for {subj_id} (label={lbl}, "
              f"handcrafted={hc_global.shape}, gnn={gnn_agg.shape})")

    print("[DONE] All Zenodo subjects processed.")


if __name__ == "__main__":
    main()
