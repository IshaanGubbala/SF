import os
import numpy as np
import pandas as pd
import mne
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

# Paths to data and model files
DATA_PATH = "./ds004504/derivatives"  # Update if dataset is located elsewhere
PARTICIPANTS_FILE = "./ds004504/participants.tsv"
SCALER_PATH = "/Users/ishaangubbala/Documents/SF/ai_scaler.pkl"
MODEL_PATH = "/Users/ishaangubbala/Documents/SF/deep_ai_model.h5"

# Channels to use
TARGET_CHANNELS = ['Fp1', 'Fp2', 'C3', 'C4']

# Frequency bands
THETA_BAND = (4, 8)
ALPHA_BAND = (8, 12)

# Function to load and preprocess EEG data
def load_eeg_data(subject_dir):
    eeg_file = os.path.join(subject_dir, "eeg", f"{os.path.basename(subject_dir)}_task-eyesclosed_eeg.set")
    if os.path.exists(eeg_file):
        print(f"Loading EEG data from {eeg_file}...")
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
        raw.filter(1., 50., fir_design='firwin')  # Band-pass filter from 1-50Hz

        # Pick only the desired channels
        available_channels = raw.ch_names
        channels_to_pick = [ch for ch in TARGET_CHANNELS if ch in available_channels]
        if not channels_to_pick:
            print(f"Warning: None of the target channels {TARGET_CHANNELS} found in {eeg_file}.")
            return None
        raw.pick_channels(channels_to_pick)
        print(f"Selected channels: {channels_to_pick}")
        return raw
    return None

def extract_features(raw):
    print("Extracting features...")

    # Extract EEG data as a NumPy array
    data = raw.get_data()  # shape: (n_channels, n_samples)

    # Compute PSD using the updated `psd_array_welch`
    psds = []
    for ch_data in data:
        psd, freqs = mne.time_frequency.psd_array_welch(
            ch_data, sfreq=raw.info['sfreq'], fmin=1, fmax=50, n_fft=256, n_overlap=128
        )
        psds.append(psd)

    psds = np.array(psds)  # shape: (n_channels, n_freqs)
    psds /= np.sum(psds, axis=-1, keepdims=True)  # Normalize

    # Calculate theta and alpha power
    theta_power = psds[:, (freqs >= THETA_BAND[0]) & (freqs < THETA_BAND[1])].mean(axis=-1)
    alpha_power = psds[:, (freqs >= ALPHA_BAND[0]) & (freqs < ALPHA_BAND[1])].mean(axis=-1)

    # Spatial factor (variance of the data)
    spatial_factor = np.var(data, axis=1)

    # Complexity factor (entropy)
    complexity_factor = -np.sum(psds * np.log(psds + 1e-8), axis=-1)

    # Hemisphere synchronicity (difference in power between left and right hemispheres)
    left_channels = data[:2, :]  # First two channels
    right_channels = data[2:, :]  # Last two channels
    left_power = np.mean(left_channels)  # Average across samples
    right_power = np.mean(right_channels)
    hemisphere_sync = np.abs(left_power - right_power)  # Single value

    # Broadcast hemisphere_sync to match the shape of other features
    hemisphere_sync = np.full_like(theta_power, hemisphere_sync)

    # Combine features
    features = np.column_stack([theta_power, alpha_power, spatial_factor, complexity_factor, hemisphere_sync])
    print(f"Features shape: {features.shape}")
    return features


# Function to load participant labels
def load_labels():
    print("Loading labels from participants file...")
    participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t')
    group_mapping = {'A': 0, 'C': 1, 'F': 2}  # A=Alzheimer's, C=Control, F=FTD
    labels = participants['Group'].map(group_mapping).values
    print("Label mapping: A=0 (Alzheimer's), C=1 (Control), F=2 (FTD)")
    return participants['participant_id'].values, labels

# Train and save scaler
def train_and_save_scaler(features):
    print("Training and saving scaler...")
    scaler = StandardScaler()
    scaler.fit(features)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved at {SCALER_PATH}.")
    return scaler

# Train and save AI model
def train_and_save_ai_model(features_scaled, labels):
    print("Training AI model...")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(features_scaled.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(features_scaled, labels, epochs=50, batch_size=32, verbose=1, validation_split=0.2)
    
    # Save model in native Keras format
    keras_model_path = MODEL_PATH.replace('.h5', '.keras')
    model.save(keras_model_path)
    print(f"AI model saved at {keras_model_path}.")
    return model


# Main function
def main():
    # Load participant labels
    participant_ids, labels = load_labels()

    all_features, all_labels = [], []

    # Process each subject
    for subject_id, label in zip(participant_ids, labels):
        subject_dir = os.path.join(DATA_PATH, subject_id)
        raw = load_eeg_data(subject_dir)
        if raw:
            features = extract_features(raw)
            all_features.append(features)
            all_labels.extend([label] * features.shape[0])

    # Combine all features and labels
    if not all_features:
        print("No valid data found. Exiting.")
        return

    X = np.vstack(all_features)
    y = np.array(all_labels)
    print(f"Total features shape: {X.shape}, Total labels shape: {y.shape}")

    # Train scaler
    scaler = train_and_save_scaler(X)

    # Scale features
    X_scaled = scaler.transform(X)

    # Train and save AI model
    train_and_save_ai_model(X_scaled, y)

if __name__ == "__main__":
    main()

