import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from collections import deque
import time
import pandas as pd
import matplotlib.pyplot as plt
import mne
from pyprep.prep_pipeline import PrepPipeline
from tabulate import tabulate

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Configuration
CHANNELS = ["Fp1", "Fp2", "C3", "C4"]
SAMPLING_RATE = 500
BUFFER_LENGTH = SAMPLING_RATE * 10  # 10 seconds buffer
ANALYSIS_INTERVAL = 2 * 60  # 2 minutes
DATA_INTERVAL = 0.005  # 5 ms
FREQUENCY_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
    "Gamma": (30, 100),
}

FEATURE_LABELS = [
    f"{band}_Power" for band in FREQUENCY_BANDS
] + ["Entropy", "Spatial Complexity", "Temporal Complexity"]

# Visualization setup
plt.ion()
fig, axes = plt.subplots(len(CHANNELS), 1, sharex=True, figsize=(8, 6))
for i, ch in enumerate(CHANNELS):
    axes[i].set_xlim(0, BUFFER_LENGTH)
    axes[i].set_ylim(-1.5, 1.5)
    axes[i].set_title(ch)
    axes[i].set_xlabel("Samples")
    axes[i].set_ylabel("Amplitude")
lines = [axes[i].plot([], [])[0] for i in range(len(CHANNELS))]

# Initialize buffer
buffer = deque(maxlen=BUFFER_LENGTH)

# Load the model and scaler
model = load_model("model.keras")
scaler = pd.read_pickle("scaler.pkl")

# Feature extraction functions
def compute_band_powers(data, sampling_rate):
    psd, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=sampling_rate, verbose=False)
    band_powers = {}
    for band, (fmin, fmax) in FREQUENCY_BANDS.items():
        idx_band = (freqs >= fmin) & (freqs <= fmax)
        band_powers[band] = np.mean(psd[:, :, idx_band], axis=2)
    band_powers_mean = {band: np.mean(band_powers[band], axis=0) for band in band_powers}
    return band_powers_mean

def compute_epoch_entropy(data):
    entropies = []
    for epoch in data:
        p = np.abs(epoch) / np.sum(np.abs(epoch))
        entropy = -np.sum(p * np.log2(p + 1e-12))
        entropies.append(entropy)
    return np.mean(entropies)

def compute_spatial_complexity(data):
    channel_variances = np.var(data, axis=-1)
    return np.var(channel_variances)

def compute_temporal_complexity(data):
    temporal_variances = np.var(data, axis=1)
    return np.mean(temporal_variances)

def create_custom_montage(channel_names):
    """
    Create a custom montage for the specified channel names.
    """
    from mne.channels import make_dig_montage

    # Define approximate positions for your channels (example values in meters)
    channel_positions = {
        "Fp1": [-0.03, 0.08, 0],  # Example 3D coordinates
        "Fp2": [0.03, 0.08, 0],
        "C3": [-0.05, 0, 0],
        "C4": [0.05, 0, 0],
    }

    montage = make_dig_montage(ch_pos=channel_positions, coord_frame="head")
    return montage

def apply_prep_pipeline(raw):
    """
    Apply the PREP pipeline for preprocessing EEG data.
    """
    try:
        # Create and set a custom montage
        montage = create_custom_montage(CHANNELS)
        raw.set_montage(montage)

        # Define PREP parameters
        prep_params = {
            "ref_chs": "average",  # Use average reference
            "reref_chs": "all",   # Channels to re-reference
            "line_freqs": [60],    # Line noise frequency for filtering
        }

        prep = PrepPipeline(raw, prep_params, montage=None)
        prep.fit()

        return prep.raw
    except ValueError as e:
        print(f"Error in PREP pipeline: {e}")
        return raw  # Return the raw data without PREP processing

def extract_features(raw):
    raw.load_data()

    # Apply PREP pipeline
    raw = apply_prep_pipeline(raw)

    # Create fixed-length epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, overlap=0.5, verbose=False)
    data = epochs.get_data()

    # Compute features
    band_powers = compute_band_powers(data, SAMPLING_RATE)
    entropy = compute_epoch_entropy(data)
    spatial_complexity = compute_spatial_complexity(data)
    temporal_complexity = compute_temporal_complexity(data)

    features = np.hstack([
        np.mean(band_powers[band]) for band in FREQUENCY_BANDS
    ] + [
        entropy, spatial_complexity, temporal_complexity
    ]) # Scale all features down by 100
    return features

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# Main function
def main():
    params = BrainFlowInputParams()
    params.serial_port = "COM3"  # Replace with your board's serial port
    board_id = BoardIds.SYNTHETIC_BOARD  # Replace with your actual board ID if different
    board = BoardShim(board_id, params)

    board.prepare_session()
    board.start_stream()
    print("Starting real-time EEG analysis...")

    try:
        last_analysis_time = time.time()

        # Retrieve EEG channel names from the board
        eeg_channel_names = [f"EEG {ch}" for ch in CHANNELS]

        while True:
            data = board.get_current_board_data(SAMPLING_RATE)
            if data.shape[1] == 0:
                continue

            # Scale raw data by dividing it by 100
            data[:len(CHANNELS), :] /= 100

            selected_data = data[:len(CHANNELS), :].T  # Shape: (samples, channels)
            buffer.extend(selected_data)

            # Update visualization
            for i, line in enumerate(lines):
                channel_data = [row[i] for row in buffer]
                line.set_data(range(len(buffer)), channel_data)
                axes[i].relim()
                axes[i].autoscale_view()
            plt.pause(0.001)

            # Short-term analysis
            if len(buffer) >= BUFFER_LENGTH:
                short_term_data = np.array(buffer).T  # Shape: (channels, samples)
                raw = mne.io.RawArray(short_term_data, mne.create_info(CHANNELS, SAMPLING_RATE, ch_types="eeg"))

                features = extract_features(raw)
                if features is not None:
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    prediction = model.predict(features_scaled)[0, 0]
                    clear_console()

                    # Prepare table for display
                    table_data = [[label, value] for label, value in zip(FEATURE_LABELS, features)]
                    table_data.append(["Prediction", prediction])
                    print(tabulate(table_data, headers=["Feature", "Value"], tablefmt="grid"))

            if time.time() - last_analysis_time >= ANALYSIS_INTERVAL:
                long_term_data = np.array(buffer).T  # Shape: (channels, samples)
                raw = mne.io.RawArray(long_term_data, mne.create_info(CHANNELS, SAMPLING_RATE, ch_types="eeg"))

                features = extract_features(raw)
                if features is not None:
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    prediction = model.predict(features_scaled)[0, 0]
                    clear_console()

                    # Prepare table for display
                    table_data = [[label, value] for label, value in zip(FEATURE_LABELS, features)]
                    table_data.append(["Prediction", prediction])
                    print(tabulate(table_data, headers=["Feature", "Value"], tablefmt="grid"))
                last_analysis_time = time.time()

            time.sleep(DATA_INTERVAL)

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        board.stop_stream()
        board.release_session()
        print("EEG session closed.")

if __name__ == "__main__":
    main()
