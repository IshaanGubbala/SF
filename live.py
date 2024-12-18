import os
import numpy as np
import joblib
import tensorflow as tf
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
import matplotlib.pyplot as plt
from collections import deque
import time

# File paths for AI model and scaler
AI_MODEL_PATH = "/Users/ishaangubbala/Documents/SF/deep_ai_model.keras"
AI_SCALER_PATH = "/Users/ishaangubbala/Documents/SF/ai_scaler.pkl"

# Visualization settings
PLOT_WINDOW = 2000  # Number of data points to display in the plot
PLOT_CHANNELS = 4  # Number of channels to visualize

# Channels to use
TARGET_CHANNELS = [1, 2, 3, 4]
WINDOW_LENGTH = 2  # seconds
SAMPLING_RATE = 256  # Hz
EPOCH_SIZE = WINDOW_LENGTH * SAMPLING_RATE

# Labels for predictions
LABEL_MAPPING = {0: "Alzheimer's", 1: "Healthy"}

# Pre-defined accuracies (from training)
MODEL_ACCURACIES = {
    "AI Model": 0.7104   # Replace with actual AI model training accuracy
}

# Add a buffer to store features from multiple intervals
FEATURE_BUFFER_SIZE = 10  # Number of intervals to store
feature_buffer = deque(maxlen=FEATURE_BUFFER_SIZE)

# Load AI model and scaler
print("Loading AI model and scaler...")
ai_model = tf.keras.models.load_model(AI_MODEL_PATH)
ai_scaler = joblib.load(AI_SCALER_PATH)
print("AI model and scaler loaded.")

# Function to clean extracted features
def clean_features(features):
    """
    Replace NaNs or infinite values with zeros in the feature set.
    """
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# Function to extract AI features
def extract_ai_features(data):
    psds = np.abs(np.fft.fft(data, n=256))[:, :(256 // 2)]
    psds /= np.sum(psds, axis=1, keepdims=True)

    freqs = np.fft.fftfreq(256, d=1 / SAMPLING_RATE)[:256 // 2]
    theta_power = psds[:, (freqs >= 4) & (freqs < 8)].mean(axis=1)
    alpha_power = psds[:, (freqs >= 8) & (freqs < 12)].mean(axis=1)

    spatial_factor = np.var(data, axis=1)
    complexity_factor = -np.sum(psds * np.log(psds + 1e-8), axis=1)

    left_power = data[:2, :].mean()
    right_power = data[2:, :].mean()
    hemisphere_sync = np.abs(left_power - right_power)

    # Add a placeholder for missing feature (if any)
    placeholder_feature = 0.0

    features = np.column_stack([theta_power, alpha_power, spatial_factor, complexity_factor, [hemisphere_sync] * len(data), [placeholder_feature] * len(data)])
    return clean_features(features.mean(axis=0))
# Function to aggregate features with buffer
def evaluate_with_feature_buffer(features):
    feature_buffer.append(features)
    # Average features across the buffer
    aggregated_features = np.mean(feature_buffer, axis=0)
    return aggregated_features

# Function to evaluate with AI model
def evaluate_with_ai_model(model, scaler, features):
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(features_scaled)
    result = "Alzheimer's" if prediction[0][0] > 0.5 else "Healthy"
    confidence = prediction[0][0] if result == "Alzheimer's" else 1 - prediction[0][0]
    return result, confidence

# Function to display results
def display_results(ai_result, ai_confidence, ai_features):
    print("\033c", end="")  # Clear console
    print(f"{'Model':<20} {'Prediction':<15} {'Accuracy (%)':<15} {'Confidence (%)':<15}")
    print("-" * 65)

    ai_accuracy = MODEL_ACCURACIES.get("AI Model", "N/A") * 100
    print(f"{'AI Model':<20} {ai_result:<15} {ai_accuracy:<15.2f} {ai_confidence * 100:<15.2f}")

    # Display factors for AI model
    print("\nFactors (AI Model):")
    print(f"Theta Power: {ai_features[0]:.4f}")
    print(f"Alpha Power: {ai_features[1]:.4f}")
    print(f"Spatial Factor: {ai_features[2]:.4f}")
    print(f"Complexity Factor: {ai_features[3]:.4f}")
    print(f"Hemisphere Synchrony: {ai_features[4]:.4f}")

    # Display confidence breakdown
    print(f"\nConfidence Breakdown: Alzheimer's={ai_confidence:.4f}, Healthy={1 - ai_confidence:.4f}")

# Function to visualize data
def setup_visualization():
    fig, ax = plt.subplots(PLOT_CHANNELS, 1, figsize=(10, 8))
    lines = []
    data_buffers = [deque([0] * PLOT_WINDOW, maxlen=PLOT_WINDOW) for _ in range(PLOT_CHANNELS)]

    for i in range(PLOT_CHANNELS):
        line, = ax[i].plot(range(PLOT_WINDOW), data_buffers[i])
        lines.append(line)
        ax[i].set_ylim(-100, 100)
        ax[i].set_title(f"Channel {i + 1}")
    plt.tight_layout()
    plt.ion()
    plt.show()
    return fig, ax, lines, data_buffers

def update_visualization(lines, data_buffers, new_data):
    for i, (line, buffer, channel_data) in enumerate(zip(lines, data_buffers, new_data)):
        buffer.extend(channel_data)
        line.set_ydata(buffer)
    plt.pause(0.01)

# Main function to stream data
def main():
    print("Setting up BrainFlow for OpenBCI Ganglion...")
    params = BrainFlowInputParams()
    params.serial_port = ""  # Auto-detect
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)

    board.prepare_session()
    board.start_stream()
    print("EEG stream started. Press Ctrl+C to stop.")

    fig, ax, lines, data_buffers = setup_visualization()

    last_update_time = time.time()
    forced_update_time = time.time()

    try:
        while True:
            data = board.get_current_board_data(EPOCH_SIZE)
            eeg_data = data[TARGET_CHANNELS, :]

            # Visualize data
            update_visualization(lines, data_buffers, eeg_data)

            # Update every 2.5 seconds, forced every 10 seconds
            current_time = time.time()
            if current_time - last_update_time >= 0.1 or current_time - forced_update_time >= 10:
                last_update_time = current_time

                # Extract features
                ai_features = extract_ai_features(eeg_data)

                # Aggregate features with the buffer
                aggregated_features = evaluate_with_feature_buffer(ai_features)

                # Evaluate AI model with aggregated features
                ai_result, ai_confidence = evaluate_with_ai_model(ai_model, ai_scaler, aggregated_features)

                # Display results
                display_results(ai_result, ai_confidence, aggregated_features)

                if current_time - forced_update_time >= 10:
                    forced_update_time = current_time

    except KeyboardInterrupt:
        print("\nStopping EEG stream.")
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
