#!/usr/bin/env python
"""
EEG Processing Script

This script loads EEG .set files from the specified directories, computes:
  1. Handcrafted global features (mean and standard deviation of per‐epoch handcrafted features)
  2. GNN aggregated features (channel-level band-power features, averaged over epochs)
  3. Channel names (from the montage after applying current source density)

It then saves each subject’s outputs in dedicated directories.
Parallel processing is used via ProcessPoolExecutor.
"""

#############################################
# CONFIGURATION & IMPORTS
#############################################
import os, glob, json, warnings, time
import numpy as np
import pandas as pd
import mne
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from imblearn.over_sampling import SMOTE
from numba import njit

# Suppress warnings and logs
mne.set_log_level('WARNING')
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#############################################
# CONFIGURATION (Update paths as needed)
#############################################
DS004504_PATH = "/Users/ishaangubbala/Documents/SF/ds004504/derivatives"
DS003800_PATH = "/Users/ishaangubbala/Documents/SF/ds003800/"
PARTICIPANTS_FILE_DS004504 = "/Users/ishaangubbala/Documents/SF/ds004504/participants.tsv"
PARTICIPANTS_FILE_DS003800 = "/Users/ishaangubbala/Documents/SF/ds003800/participants.tsv"
SAMPLING_RATE = 256  # Hz

FREQUENCY_BANDS = {
    "Delta": (0.5, 4), "Theta1": (4, 6), "Theta2": (6, 8),
    "Alpha1": (8, 10), "Alpha2": (10, 12),
    "Beta1": (12, 20), "Beta2": (20, 30),
    "Gamma1": (30, 40), "Gamma2": (40, 50)
}

# Output directories for saving processed features
BASE_OUTPUT_DIR = "processed_features"
HANDCRAFTED_DIR = os.path.join(BASE_OUTPUT_DIR, "handcrafted")
GNN_DIR = os.path.join(BASE_OUTPUT_DIR, "gnn")
CHANNELS_DIR = os.path.join(BASE_OUTPUT_DIR, "channels")
for d in [BASE_OUTPUT_DIR, HANDCRAFTED_DIR, GNN_DIR, CHANNELS_DIR]:
    os.makedirs(d, exist_ok=True)

#############################################
# 1) LOAD PARTICIPANT LABELS
#############################################
def load_participant_labels(ds004504_file, ds003800_file):
    label_dict = {}
    # Only include groups "A" and "C"
    group_map_ds004504 = {"A": 1, "C": 0, "F": 1}
    df_ds004504 = pd.read_csv(ds004504_file, sep="\t")
    #df_ds004504 = df_ds004504[df_ds004504['Group'] != 'F']
    df_ds004504 = df_ds004504[df_ds004504['Group'].isin(group_map_ds004504.keys())]
    labels_ds004504 = df_ds004504.set_index("participant_id")["Group"].map(group_map_ds004504).to_dict()
    label_dict.update(labels_ds004504)
    df_ds003800 = pd.read_csv(ds003800_file, sep="\t")
    labels_ds003800 = df_ds003800.set_index("participant_id")["Group"].apply(lambda x: 1).to_dict()
    label_dict.update(labels_ds003800)
    return label_dict

participant_labels = load_participant_labels(PARTICIPANTS_FILE_DS004504, PARTICIPANTS_FILE_DS003800)

#############################################
# 2) HANDCRAFTED FEATURE EXTRACTION FUNCTIONS
#############################################
def compute_band_powers(data, sfreq):
    psd, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=sfreq, verbose=False)
    band_powers = {band: np.mean(psd[:, :, (freqs >= fmin) & (freqs <= fmax)], axis=2)
                   for band, (fmin, fmax) in FREQUENCY_BANDS.items()}
    return band_powers

def compute_shannon_entropy(data):
    def entropy_fn(x):
        counts, _ = np.histogram(x, bins=256)
        p = counts / np.sum(counts)
        return -np.sum(p * np.log2(p + 1e-12))
    return np.apply_along_axis(entropy_fn, 1, data)

@njit
def compute_hjorth_parameters_numba(data):
    n_epochs, n_channels, n_times = data.shape
    activities = np.empty(n_epochs, dtype=np.float64)
    mobilities = np.empty(n_epochs, dtype=np.float64)
    complexities = np.empty(n_epochs, dtype=np.float64)
    for i in range(n_epochs):
        var0 = np.empty(n_channels, dtype=np.float64)
        for j in range(n_channels):
            s = 0.0
            for k in range(n_times):
                s += data[i, j, k]
            mean_val = s / n_times
            s2 = 0.0
            for k in range(n_times):
                diff = data[i, j, k] - mean_val
                s2 += diff * diff
            var0[j] = s2 / n_times
            if var0[j] < 1e-12:
                var0[j] = 1e-12
        mobility = np.empty(n_channels, dtype=np.float64)
        for j in range(n_channels):
            s2 = 0.0
            for k in range(1, n_times):
                diff = data[i, j, k] - data[i, j, k-1]
                s2 += diff * diff
            mobility[j] = np.sqrt(s2 / ((n_times - 1) * var0[j]))
        complexity = np.empty(n_channels, dtype=np.float64)
        for j in range(n_channels):
            s2 = 0.0
            for k in range(2, n_times):
                diff1 = data[i, j, k-1] - data[i, j, k-2]
                diff2 = data[i, j, k] - data[i, j, k-1]
                second_diff = diff2 - diff1
                s2 += second_diff * second_diff
            if s2 < 1e-12:
                s2 = 1e-12
            complexity[j] = np.sqrt(s2 / (n_times - 2)) / (mobility[j] + 1e-12)
        activities[i] = np.mean(var0)
        mobilities[i] = np.mean(mobility)
        complexities[i] = np.mean(complexity)
    return activities, mobilities, complexities

def extract_features_epoch(epoch_data):
    data = epoch_data[np.newaxis, :, :]
    band_powers = compute_band_powers(data, SAMPLING_RATE)
    features = [np.mean(band_powers[band]) for band in FREQUENCY_BANDS.keys()]
    alpha_power = np.mean(band_powers["Alpha1"]) + np.mean(band_powers["Alpha2"])
    theta_power = np.mean(band_powers["Theta1"]) + np.mean(band_powers["Theta2"])
    total_power = sum(np.mean(band_powers[band]) for band in FREQUENCY_BANDS.keys()) + 1e-12
    features.extend([alpha_power/total_power, theta_power/total_power])
    sh_entropy = np.mean(compute_shannon_entropy(data))
    features.append(sh_entropy)
    act, mob, comp = compute_hjorth_parameters_numba(data)
    features.extend([np.mean(act), np.mean(mob), np.mean(comp)])
    return np.array(features, dtype=np.float32)

#############################################
# 3) GNN CHANNEL-FEATURE EXTRACTION
#############################################
def extract_channel_features_GNN_epoch(epoch_data, sfreq):
    n_channels = epoch_data.shape[0]
    features = np.zeros((n_channels, len(FREQUENCY_BANDS)), dtype=np.float32)
    for ch in range(n_channels):
        ts = epoch_data[ch, :]
        psd, freqs = mne.time_frequency.psd_array_multitaper(
            ts[np.newaxis, :], sfreq=sfreq, verbose=False
        )
        band_vals = []
        for (fmin, fmax) in FREQUENCY_BANDS.values():
            idx = (freqs >= fmin) & (freqs <= fmax)
            band_vals.append(np.mean(psd[0, idx]))
        features[ch, :] = np.array(band_vals, dtype=np.float32)
    return features

#############################################
# 4) COMBINED PROCESSING (HANDCRAFTED + GNN)
#############################################
def process_subject_combined(args):
    file, label = args
    try:
        raw = mne.io.read_raw_eeglab(file, preload=True, verbose=False)
        raw.filter(1.0, 50.0, fir_design="firwin", verbose=False)
        raw.resample(SAMPLING_RATE, npad="auto")
        # Create fixed-length epochs (adjust overlap if needed)
        epochs = mne.make_fixed_length_epochs(raw, duration=20.0, overlap=0.0, verbose=False)
        data = epochs.get_data()
        if data.size == 0:
            raise ValueError("No data extracted from epochs.")

        # Handcrafted branch: compute per-epoch features then aggregate (mean & std)
        epoch_hand_features = np.array([extract_features_epoch(epoch) for epoch in data])
        mean_features = np.mean(epoch_hand_features, axis=0)
        std_features = np.std(epoch_hand_features, axis=0)
        handcrafted_global = np.hstack((mean_features, std_features))

        # GNN branch: apply Laplacian montage then extract channel-level features per epoch
        raw_lap = raw.copy()
        raw_lap = mne.preprocessing.compute_current_source_density(raw_lap)
        epochs_lap = mne.make_fixed_length_epochs(raw_lap, duration=20.0, overlap=0.0, verbose=False)
        data_lap = epochs_lap.get_data()
        gnn_epoch_features = np.array([extract_channel_features_GNN_epoch(epoch, SAMPLING_RATE) for epoch in data_lap])
        # Aggregate GNN features over epochs (by taking the mean)
        gnn_aggregated = np.mean(gnn_epoch_features, axis=0)
        ch_names = raw_lap.ch_names

        return handcrafted_global, epoch_hand_features, gnn_aggregated, label, ch_names
    except Exception as e:
        print(f"[ERROR] Processing failed for {file}: {e}")
        return None, None, None, None, None

#############################################
# 5) LOAD DATASET IN PARALLEL & SAVE OUTPUTS
#############################################
def main():
    # Collect .set files from both datasets
    dataset_paths = [DS004504_PATH, DS003800_PATH]
    all_files = []
    for path in dataset_paths:
        pattern = os.path.join(path, "**", "*_task-Rest_eeg.set") if 'ds003800' in path else os.path.join(path, "**", "*.set")
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)

    # Build tasks: only include files for which the subject ID is found in participant_labels.
    tasks = []
    for f in all_files:
        subj = os.path.basename(f).split('_')[0]
        label = participant_labels.get(subj, None)
        if label is None:
            print(f"[DEBUG] Skipping {f} because subject {subj} not found in labels.")
        else:
            tasks.append((f, label))
    print(f"[INFO] Found {len(tasks)} files to process.")  # Expected: 76 subjects

    processed_subjects = []
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_subject_combined, tasks), total=len(tasks)))
    for res in results:
        if res[0] is not None and res[2] is not None:
            processed_subjects.append(res)
    print(f"[INFO] Successfully processed {len(processed_subjects)} subjects.")

    # Save outputs for each processed subject
    for res, task in zip(processed_subjects, tasks):
        file, _ = task
        subj_id = os.path.basename(file).split('_')[0]
        handcrafted_global, epoch_features, gnn_aggregated, label, ch_names = res
        np.save(os.path.join(HANDCRAFTED_DIR, f"{subj_id}_handcrafted.npy"), handcrafted_global)
        np.save(os.path.join(GNN_DIR, f"{subj_id}_gnn.npy"), gnn_aggregated)
        with open(os.path.join(CHANNELS_DIR, f"{subj_id}_channels.json"), "w") as fp:
            json.dump(ch_names, fp)
        print(f"[INFO] Saved features for subject {subj_id}")

    print(f"[DONE] Processed and saved features for {len(processed_subjects)} subjects.")

if __name__ == "__main__":
    main()
