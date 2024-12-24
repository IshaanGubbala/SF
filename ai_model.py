import os
import mne
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, Conv1D, MaxPooling1D, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import glob
import warnings
from sklearn.utils.class_weight import compute_class_weight

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

# Configuration
DATASET_PATHS = ["/Users/ishaangubbala/Documents/SF/ds004504/derivatives", "/Users/ishaangubbala/Documents/SF/ds004796"]
PARTICIPANTS_FILES = {
    "ds004504": "/Users/ishaangubbala/Documents/SF/ds004504/participants.tsv",
    "ds004796": "/Users/ishaangubbala/Documents/SF/ds004796/participants.tsv",
}
SAMPLING_RATE = 256
FREQUENCY_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
    "Gamma": (30, 100),
}

FEATURE_LABELS = [f"{band}_Power" for band in FREQUENCY_BANDS] + ["Entropy", "Spatial Complexity", "Temporal Complexity"]

# Load participant metadata
def load_participant_labels():
    participant_labels = {}
    for dataset, file_path in PARTICIPANTS_FILES.items():
        participants_df = pd.read_csv(file_path, sep="\t")
        print(f"Available columns in {file_path}: {participants_df.columns.tolist()}")

        # Map participant IDs to labels
        if dataset == "ds004504":
            group_mapping = {
                "A": 1,        # Alzheimer's
                "C": 0,        # Healthy control
                "F": 2,        # Frontotemporal dementia
            }
            participant_labels.update(participants_df.set_index("participant_id")["Group"].map(group_mapping).to_dict())
        elif dataset == "ds004796":
            participant_labels.update({row["participant_id"]: 0 for _, row in participants_df.iterrows()})
    return participant_labels

participant_labels = load_participant_labels()

# Feature extraction functions (same as before)
def compute_band_powers(data, sampling_rate):
    psd, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=sampling_rate, verbose=False)
    assert len(psd.shape) == 3, f"Unexpected PSD shape: {psd.shape}"
    band_powers = {}
    for band, (fmin, fmax) in FREQUENCY_BANDS.items():
        idx_band = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(idx_band):
            raise ValueError(f"No frequencies found in range {fmin}-{fmax} Hz for band {band}.")
        band_powers[band] = np.mean(psd[:, :, idx_band], axis=2)
    band_powers_mean = {band: np.mean(band_powers[band], axis=0) for band in band_powers}
    return band_powers_mean

def compute_epoch_entropy(data):
    entropies = []
    for epoch in data:
        p = np.abs(epoch) / np.sum(np.abs(epoch))  # Normalize signal
        entropy = -np.sum(p * np.log2(p + 1e-12))  # Compute entropy
        entropies.append(entropy)
    return np.mean(entropies)  # Return average entropy for all epochs

def compute_spatial_complexity(data):
    channel_variances = np.var(data, axis=-1)  # Variance of each channel
    return np.var(channel_variances)  # Variance of variances

def compute_temporal_complexity(data):
    temporal_variances = np.var(data, axis=1)  # Variance over time for each channel
    return np.mean(temporal_variances)  # Average temporal variance

def extract_features(raw):
    raw.load_data()
    raw.filter(0.5, 100, fir_design="firwin", verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, overlap=0.5, verbose=False)
    data = epochs.get_data()

    # Extract features
    band_powers = compute_band_powers(data, SAMPLING_RATE)
    entropy = compute_epoch_entropy(data)
    spatial_complexity = compute_spatial_complexity(data)
    temporal_complexity = compute_temporal_complexity(data)

    features = np.hstack([
        np.mean(band_powers[band]) for band in FREQUENCY_BANDS
    ] + [entropy, spatial_complexity, temporal_complexity])
    return features

# Parallel processing
def process_subject(file, participant_label):
    try:
        # Read the EEG file
        print(f"Reading file: {file}")
        raw = mne.io.read_raw_brainvision(file, preload=True)

        # Preprocess the data (e.g., filtering, downsampling, etc.)
        print(f"Preprocessing data for {file}")
        raw.filter(1., 30., fir_design='firwin')  # Example bandpass filter
        raw.resample(128)  # Example resampling

        # Extract epochs or features
        data = raw.get_data()
        features = extract_features(data)  # Replace with your feature extraction function

        # Return features and participant label
        return features, participant_label

    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None, None
def process_subject_wrapper(args):
    file, participant_labels = args
    try:
        return process_subject(file, participant_labels)
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None

def load_dataset_parallel():
    """
    Load the dataset using parallel processing.
    """
    # Generate file paths for EEG files
    files = [
        (file, participant_labels)
        for file in sorted(glob.glob("./*/sub-*/*_task-rest_eeg.*"))
    ]

    if not files:
        raise ValueError("No valid EEG files found. Check the file paths.")

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_subject_wrapper, files),
                total=len(files),
                desc="Processing EEG Files",
            )
        )

    # Debug output to ensure results are valid
    if not any(results):
        raise ValueError("No valid results from processing. Check `process_subject`.")

    # Combine all the features and labels
    features = np.concatenate([res[0] for res in results if res is not None], axis=0)
    labels = np.concatenate([res[1] for res in results if res is not None], axis=0)

    return features, labels


# TensorFlow model
def create_cnn_lstm_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim, 1)),
        Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, padding="same"),
        Dropout(0.3),
        LSTM(64, return_sequences=True, activation="tanh", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32, activation="tanh", kernel_regularizer=l2(0.001)),
        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(1, activation="sigmoid")  # Output layer
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Train and evaluate model
def train_model():
    features, labels = load_dataset_parallel()

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    print("Starting 5-Fold Cross-Validation...")

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weights = dict(enumerate(class_weights))

    for fold, (train_idx, test_idx) in enumerate(kfold.split(features), 1):
        print(f"Training Fold {fold}/5...")
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Reshape for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = create_cnn_lstm_model(input_dim=X_train.shape[1])
        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=20, batch_size=32, verbose=1,
            class_weight=class_weights
        )

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Fold {fold} Accuracy: {accuracy:.4f}")
        fold_accuracies.append(accuracy)

    print(f"Average Cross-Validation Accuracy: {np.mean(fold_accuracies):.4f}")

if __name__ == "__main__":
    file = "./ds004504/sub-01/eeg/sub-01_task-rest_eeg.vhdr"  # Adjust path as needed
    participant_label = participant_labels["sub-01"]  # Replace with actual label if available
    print(process_subject(file, participant_label))

