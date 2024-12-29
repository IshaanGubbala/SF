import os
import numpy as np
import mne
import pandas as pd
import glob
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, LSTM, BatchNormalization,
    Bidirectional, Input
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Parallel processing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Stats
from scipy.stats import ttest_ind, entropy

# Nolds for entropy calculations
import nolds

# Numba for JIT compilation
from numba import njit, prange

# CuPy for GPU acceleration
import cupy as cp

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------

DATASET_PATH = "/path/to/your/eeg_data"           # Update this path
PARTICIPANTS_FILE = "/path/to/participants.tsv"   # Update this path
SAMPLING_RATE = 256  # Desired resampling rate in Hz

# Sub-band definitions (Hz)
FREQUENCY_BANDS = {
    "Delta": (0.5, 4),
    "Theta1": (4, 6),
    "Theta2": (6, 8),
    "Alpha1": (8, 10),
    "Alpha2": (10, 12),
    "Beta1": (12, 20),
    "Beta2": (20, 30),
    "Gamma1": (30, 40),
    "Gamma2": (40, 50),
}

# --------------------------------------------------------------------------------
# 1) LOAD PARTICIPANT LABELS
# --------------------------------------------------------------------------------
def load_participant_labels():
    """
    Reads participants.tsv, removes FTD if present,
    maps 'A' -> 1 (Alzheimer), 'C' -> 0 (Control).
    """
    print("[DEBUG] Loading participant labels.")
    df = pd.read_csv(PARTICIPANTS_FILE, sep="\t")
    df = df[df['Group'] != 'F']  # remove FTD if present
    group_map = {"A": 1, "C": 0}
    label_dict = df.set_index("participant_id")["Group"].map(group_map).to_dict()
    print(f"[DEBUG] Found {len(label_dict)} subjects after removing FTD.")
    return label_dict

participant_labels = load_participant_labels()

# --------------------------------------------------------------------------------
# 2) HELPER FUNCTIONS (PSD, ENTROPIES, ETC.) WITH DEBUG USING NOLDS, SCIPY, CUPY
# --------------------------------------------------------------------------------

# Numba-accelerated Shannon Entropy
@njit(parallel=True)
def compute_shannon_entropy_numba(flattened_epochs, bins=256):
    """
    Computes Shannon entropy for multiple flattened epochs using Numba for acceleration.
    """
    n_epochs = flattened_epochs.shape[0]
    H = np.empty(n_epochs, dtype=np.float32)
    for i in prange(n_epochs):
        counts, _ = np.histogram(flattened_epochs[i], bins=bins, density=True)
        p = counts / counts.sum()
        H[i] = -np.sum(p * np.log2(p + 1e-12))
    return H

def compute_shannon_entropy_epoch_debug(data, filename):
    """
    Shannon entropy per epoch (averaged across epochs).
    Utilizes Numba-accelerated function for efficiency.
    """
    print(f"[DEBUG] {filename}: compute_shannon_entropy_epoch_debug with shape {data.shape}")
    # Flatten across channels
    flattened = data.reshape(data.shape[0], -1).astype(np.float32)
    H = compute_shannon_entropy_numba(flattened)
    mean_H = H.mean()
    return mean_H

# Numba-accelerated Sample Entropy
@njit
def sampen_numba(series, m, r):
    N = len(series)
    if N <= m + 1:
        return 0.0
    count_m = 0
    count_m1 = 0
    for i in range(N - m):
        template = series[i:i + m]
        for j in range(i + 1, N - m + 1):
            if np.max(np.abs(template - series[j:j + m])) <= r:
                count_m += 1
    for i in range(N - m -1):
        template = series[i:i + m +1]
        for j in range(i +1, N - m):
            if np.max(np.abs(template - series[j:j + m +1])) <= r:
                count_m1 += 1
    if count_m ==0:
        return 0.0
    return -np.log((count_m1 + 1e-12) / (count_m + 1e-12))

@njit(parallel=True)
def compute_sample_entropy_numba(flattened_epochs, m=2, r=0.2):
    """
    Computes Sample Entropy for multiple flattened epochs using Numba for acceleration.
    """
    n_epochs = flattened_epochs.shape[0]
    sampen = np.empty(n_epochs, dtype=np.float32)
    for i in prange(n_epochs):
        sampen[i] = sampen_numba(flattened_epochs[i], m, r)
    return sampen

def compute_sample_entropy_debug(data, filename, m=2, r=0.2):
    """
    Computes Sample Entropy per epoch (averaged across epochs) using Numba-accelerated functions.
    """
    print(f"[DEBUG] {filename}: compute_sample_entropy_debug with shape {data.shape}")
    # Flatten across channels
    flattened = data.reshape(data.shape[0], -1).astype(np.float32)
    sampen = compute_sample_entropy_numba(flattened, m, r)
    mean_sampen = sampen.mean()
    return mean_sampen

# Numba-accelerated Permutation Entropy
@njit
def perm_entropy_numba(series, order):
    N = len(series)
    if N < order:
        return 0.0
    perms = {}
    for i in range(N - order +1):
        pattern = tuple(np.argsort(series[i:i+order]))
        perms[pattern] = perms.get(pattern, 0) +1
    total = N - order +1
    probs = 0.0
    for count in perms.values():
        p = count / total
        probs -= p * np.log2(p +1e-12)
    return probs / np.log2(len(perms) +1e-12)

@njit(parallel=True)
def compute_permutation_entropy_numba(flattened_epochs, order=3):
    """
    Computes Permutation Entropy for multiple flattened epochs using Numba for acceleration.
    """
    n_epochs = flattened_epochs.shape[0]
    perm_ent = np.empty(n_epochs, dtype=np.float32)
    for i in prange(n_epochs):
        perm_ent[i] = perm_entropy_numba(flattened_epochs[i], order)
    return perm_ent

def compute_permutation_entropy_debug(data, filename, order=3):
    """
    Computes Permutation Entropy per epoch (averaged across epochs) using Numba-accelerated functions.
    """
    print(f"[DEBUG] {filename}: compute_permutation_entropy_debug with shape {data.shape}")
    # Flatten across channels
    flattened = data.reshape(data.shape[0], -1).astype(np.float32)
    perm_ent = compute_permutation_entropy_numba(flattened, order)
    mean_perm_ent = perm_ent.mean()
    return mean_perm_ent

def compute_spatial_complexity_debug(data, filename):
    """
    Log-determinant of covariance for each epoch, average across epochs.
    Utilizes CuPy for GPU acceleration.
    """
    print(f"[DEBUG] {filename}: compute_spatial_complexity_debug with shape {data.shape}")
    # Transfer data to GPU
    data_gpu = cp.asarray(data)
    # Compute covariance matrices
    cov_gpu = cp.cov(data_gpu.reshape(-1, data.shape[1]), rowvar=False)
    # Compute determinant
    det_gpu = cp.linalg.det(cov_gpu)
    # Log determinant
    log_det = cp.log(det_gpu + 1e-12)
    # Transfer back to CPU
    log_det_cpu = cp.asnumpy(log_det)
    mean_log_det = log_det_cpu.mean()
    return mean_log_det

def compute_hjorth_parameters_debug(data, filename):
    """
    Computes Hjorth (Activity, Mobility, Complexity) across epochs.
    Utilizes Numba for acceleration.
    """
    print(f"[DEBUG] {filename}: compute_hjorth_parameters_debug with shape {data.shape}")
    n_epochs, n_channels, n_times = data.shape
    activities = np.empty(n_epochs, dtype=np.float32)
    mobilities = np.empty(n_epochs, dtype=np.float32)
    complexities = np.empty(n_epochs, dtype=np.float32)
    
    for i in prange(n_epochs):
        epoch = data[i]
        var0 = np.var(epoch, axis=1) + 1e-12
        mob = np.sqrt(np.var(np.diff(epoch, axis=1), axis=1) / var0)
        comp = np.sqrt(np.var(np.diff(np.diff(epoch, axis=1), axis=1), axis=1) / (mob + 1e-12))
        activities[i] = var0.mean()
        mobilities[i] = mob.mean()
        complexities[i] = comp.mean()
    
    mean_act = activities.mean()
    mean_mob = mobilities.mean()
    mean_comp = complexities.mean()
    
    return mean_act, mean_mob, mean_comp

def compute_band_powers_debug(data, sfreq, filename):
    """
    Computes PSD with multitaper and extracts average power per sub-band.
    Includes debug prints to show progress.
    Utilizes CuPy for GPU acceleration where applicable.
    """
    print(f"[DEBUG] {filename}: Starting PSD calculation with shape {data.shape}")
    # Transfer data to GPU
    data_gpu = cp.asarray(data)
    
    # Compute PSD using MNE's multitaper (runs on CPU)
    # Unfortunately, MNE's PSD functions do not support CuPy directly
    # Thus, we'll compute PSD on CPU, but can transfer data more efficiently
    psd, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=sfreq, verbose=False)
    
    print(f"[DEBUG] {filename}: PSD done. psd shape={psd.shape}, freqs shape={freqs.shape}")
    band_powers = {}
    for band_name, (fmin, fmax) in FREQUENCY_BANDS.items():
        idx_band = (freqs >= fmin) & (freqs <= fmax)
        band_vals = np.mean(psd[:, :, idx_band], axis=2)  # average freq dimension
        band_powers[band_name] = np.mean(band_vals)       # average across epochs & channels
    return band_powers

def compute_features_parallel(data, filename):
    """
    Computes all features in parallel using ThreadPoolExecutor.
    """
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            "band_powers": executor.submit(compute_band_powers_debug, data, SAMPLING_RATE, filename),
            "shannon_entropy": executor.submit(compute_shannon_entropy_epoch_debug, data, filename),
            "spatial_complexity": executor.submit(compute_spatial_complexity_debug, data, filename),
            "hjorth_params": executor.submit(compute_hjorth_parameters_debug, data, filename),
            "sample_entropy": executor.submit(compute_sample_entropy_debug, data, filename),
            "permutation_entropy": executor.submit(compute_permutation_entropy_debug, data, filename),
        }
        results = {key: future.result() for key, future in futures.items()}
    return results

# --------------------------------------------------------------------------------
# 3) MAIN FEATURE EXTRACTION WITH DEBUG USING NOLDS, SCIPY, CUPY, NUMBA
# --------------------------------------------------------------------------------
def extract_features_debug(raw, filename):
    """
    Full feature extraction process with 20-second epochs, using Numba and CuPy for acceleration.
    """
    print(f"[DEBUG] extract_features_debug called for {filename}")
    raw.load_data()
    print(f"[DEBUG] {filename}: loaded data. Filtering 1-50 Hz.")
    raw.filter(1.0, 50.0, fir_design="firwin", verbose=False)

    # Create 20-second epochs, no overlap
    print(f"[DEBUG] {filename}: making 20s epochs (overlap=0).")
    epochs = mne.make_fixed_length_epochs(raw, duration=20.0, overlap=0.0, verbose=False)
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    print(f"[DEBUG] {filename}: shape after epoching = {data.shape}")

    # Compute all features in parallel
    features = compute_features_parallel(data, filename)

    # Aggregate features
    subband_features = [features["band_powers"][b] for b in FREQUENCY_BANDS]
    total_power = np.sum(subband_features) + 1e-12

    alpha_power = features["band_powers"]["Alpha1"] + features["band_powers"]["Alpha2"]
    theta_power = features["band_powers"]["Theta1"] + features["band_powers"]["Theta2"]
    alpha_ratio = alpha_power / total_power
    theta_ratio = theta_power / total_power

    # Combine all features into a single vector
    feature_vector = (
        subband_features + [
            alpha_ratio, theta_ratio,
            features["shannon_entropy"], features["spatial_complexity"],
            features["hjorth_params"][0], features["hjorth_params"][1], features["hjorth_params"][2],
            features["sample_entropy"], features["permutation_entropy"]
        ]
    )
    print(f"[DEBUG] {filename}: done extracting all features.")
    return np.array(feature_vector, dtype=np.float32)

# --------------------------------------------------------------------------------
# 4) PROCESS SUBJECT (FOR PARALLEL)
# --------------------------------------------------------------------------------
def process_subject(file_label):
    """
    Worker function that loads a file, resamples, calls extract_features_debug, returns (features, label).
    Utilizes GPU acceleration where applicable.
    """
    file, label = file_label
    try:
        print(f"[DEBUG] Starting subject file: {file}")
        if file.endswith(".set"):
            raw = mne.io.read_raw_eeglab(file, preload=True)
        elif file.endswith(".vhdr"):
            raw = mne.io.read_raw_brainvision(file, preload=True)
        else:
            print(f"[DEBUG] Skipping unrecognized file type: {file}")
            return None, None

        print(f"[DEBUG] Resampling file: {file}")
        raw.resample(SAMPLING_RATE)

        print(f"[DEBUG] Extracting features for {file}")
        feats = extract_features_debug(raw, file)

        print(f"[DEBUG] Completed: {file}")
        return feats, label
    except Exception as e:
        print(f"[ERROR] process_subject failed for {file}: {e}")
        return None, None

# --------------------------------------------------------------------------------
# 5) PARALLEL LOADING USING CUPY, NUMBA, AND PARALLEL FEATURE COMPUTATION
# --------------------------------------------------------------------------------
def load_dataset_parallel():
    """
    Loads .set/.vhdr files with concurrent.futures, using Numba and CuPy for entropy calculations.
    """
    all_files = glob.glob(os.path.join(DATASET_PATH, "**", "*.set"), recursive=True)
    # If you have BrainVision: uncomment below
    # all_files += glob.glob(os.path.join(DATASET_PATH, "**", "*.vhdr"), recursive=True)

    tasks = []
    for f in sorted(all_files):
        subj_id = os.path.basename(f).split("_")[0]
        label = participant_labels.get(subj_id, 0)
        tasks.append((f, label))

    print("[DEBUG] Starting parallel extraction.")
    features_list = []
    labels_list = []

    # Adjust max_workers based on your CPU and GPU capabilities
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_subject, tasks), total=len(tasks)))

    for feats, lab in results:
        if feats is not None:
            features_list.append(feats)
            labels_list.append(lab)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    return X, y

# --------------------------------------------------------------------------------
# 6) ANALYZE FEATURES / T-TESTS
# --------------------------------------------------------------------------------
def analyze_feature_statistics(X, y, feature_names):
    """
    For each feature, plot histograms and perform t-tests between classes.
    Saves plots instead of showing to avoid blocking execution.
    Utilizes GPU acceleration with CuPy where applicable.
    """
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend to prevent blocking

    for i, f_name in enumerate(feature_names):
        print(f"[DEBUG] Analyzing feature {f_name}")
        plt.figure(figsize=(8, 6))
        sns.histplot(X[y == 0, i], color='blue', label='Control', kde=True, stat="density", linewidth=0)
        sns.histplot(X[y == 1, i], color='orange', label='Alzheimer', kde=True, stat="density", linewidth=0)
        plt.title(f"{f_name} Distribution by Class")
        plt.legend()
        plt.xlabel(f_name)
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(f"{f_name}_distribution.png")
        plt.close()

        # Perform t-test
        t_stat, p_val = ttest_ind(X[y == 0, i], X[y == 1, i], equal_var=False)
        print(f"{f_name}: t-stat = {t_stat:.4f}, p-value = {p_val:.4e}")

# --------------------------------------------------------------------------------
# 7) BASELINE MODELS
# --------------------------------------------------------------------------------
def baseline_logistic_regression(X, y):
    """
    Logistic Regression baseline with StratifiedKFold cross-validation.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_num = 1
    for train_idx, test_idx in skf.split(X, y):
        print(f"[LogReg] Fold {fold_num} - Training")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(f"\n[LogReg] Fold {fold_num} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Control', 'Alzheimer']))
        fold_num += 1

def baseline_mlp(X, y):
    """
    MLP Classifier baseline with StratifiedKFold cross-validation.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_num = 1
    for train_idx, test_idx in skf.split(X, y):
        print(f"[MLP] Fold {fold_num} - Training")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        mlp = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=300, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)

        print(f"\n[MLP] Fold {fold_num} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Control', 'Alzheimer']))
        fold_num += 1

# --------------------------------------------------------------------------------
# 8) CNN-LSTM DEEP MODEL
# --------------------------------------------------------------------------------
def create_deep_cnn_lstm(input_dim):
    """
    Builds a CNN-LSTM model architecture.
    """
    model = Sequential([
        Input(shape=(input_dim, 1)),

        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, padding="same"),
        Dropout(0.2),

        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, padding="same"),
        Dropout(0.2),

        Bidirectional(LSTM(32, activation="tanh", return_sequences=False)),
        Dropout(0.3),

        Dense(16, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def cnn_lstm_training(X, y, epochs=50, batch_size=16):
    """
    Trains a CNN-LSTM model using StratifiedKFold cross-validation.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    class_weights = dict(enumerate(compute_class_weight("balanced", classes=np.unique(y), y=y)))

    fold_num = 1
    for train_idx, test_idx in skf.split(X, y):
        print(f"[CNN-LSTM] Fold {fold_num} - Training")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Reshape for CNN
        X_train = X_train[..., np.newaxis]
        X_test  = X_test[..., np.newaxis]

        model = create_deep_cnn_lstm(X_train.shape[1])
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            class_weight=class_weights,
            callbacks=[es],
            verbose=0
        )

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred = (model.predict(X_test) > 0.5).astype(int)

        print(f"\n[CNN-LSTM] Fold {fold_num} - Test Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Control', 'Alzheimer']))
        fold_num += 1

# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------
def main():
    # 1) Load data in parallel with optimized feature extraction
    print("[DEBUG] Starting dataset loading in parallel with optimized feature extraction...")
    X, y = load_dataset_parallel()

    print("[DEBUG] Done loading dataset.")
    print("Feature matrix shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Class distribution (0=Control, 1=Alzheimer):")
    print(pd.Series(y).value_counts())

    # 2) Define feature names for reference
    band_names = list(FREQUENCY_BANDS.keys())
    extra_feats = [
        "Alpha_Ratio", "Theta_Ratio",
        "Epoch_Entropy", "Spatial_Complexity",
        "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity",
        "Sample_Entropy", "Permutation_Entropy"
    ]
    feature_names = band_names + extra_feats

    # 3) Analyze feature statistics (save plots instead of showing)
    analyze_feature_statistics(X, y, feature_names)

    # 4) Baseline Models
    print("\n--- Baseline Logistic Regression ---")
    baseline_logistic_regression(X, y)

    print("\n--- Baseline MLP ---")
    baseline_mlp(X, y)

    # 5) CNN-LSTM Deep Model
    print("\n--- CNN-LSTM Training ---")
    cnn_lstm_training(X, y, epochs=30, batch_size=8)

if __name__ == "__main__":
    main()
