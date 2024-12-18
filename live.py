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

# File paths for ML models and scaler
ML_MODELS_PATHS = {
    "RandomForest": "randomforest_model.pkl",
    "GradientBoosting": "gradientboosting_model.pkl",
    #"SVM": "svm_model.pkl"
}
ML_SCALER_PATH = "scaler.pkl"

# Visualization settings
PLOT_WINDOW = 500  # Number of data points to display in the plot
PLOT_CHANNELS = 4  # Number of channels to visualize

# Channels to use
TARGET_CHANNELS = [1, 2, 3, 4]
WINDOW_LENGTH = 2  # seconds
SAMPLING_RATE = 200  # Hz
EPOCH_SIZE = WINDOW_LENGTH * SAMPLING_RATE

# Labels for predictions
LABEL_MAPPING = {0: "Alzheimer's", 1: "Healthy"}

# Pre-defined accuracies (from training)
MODEL_ACCURACIES = {
    "RandomForest": 0.5823,
    "GradientBoosting": 0.5581,
    #"SVM": 0.5215,
    "AI Model": 0.7104   # Replace with actual AI model training accuracy
}

# Load ML models and scalers
print("Loading ML models and scalers...")
ml_models = {name: joblib.load(path) for name, path in ML_MODELS_PATHS.items()}
ml_scaler = joblib.load(ML_SCALER_PATH)
print("ML models and scaler loaded.")

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

    features = np.column_stack([theta_power, alpha_power, spatial_factor, complexity_factor, [hemisphere_sync] * len(data)])
    return clean_features(features.mean(axis=0))

# Function to extract ML features
def extract_ml_features(data):
    psds = np.abs(np.fft.fft(data, n=256))[:, :(256 // 2)]
    psds /= np.sum(psds, axis=1, keepdims=True)

    freqs = np.fft.fftfreq(256, d=1 / SAMPLING_RATE)[:256 // 2]
    delta = psds[:, (freqs >= 1) & (freqs < 4)].mean(axis=1)
    theta = psds[:, (freqs >= 4) & (freqs < 8)].mean(axis=1)
    alpha = psds[:, (freqs >= 8) & (freqs < 12)].mean(axis=1)
    beta = psds[:, (freqs >= 12) & (freqs < 30)].mean(axis=1)
    gamma = psds[:, (freqs >= 30) & (freqs <= 50)].mean(axis=1)

    entropy = -np.sum(psds * np.log(psds + 1e-8), axis=1)

    features = np.hstack([delta, theta, alpha, beta, gamma, entropy])
    return clean_features(features)

# Function to evaluate with ML models
def evaluate_with_ml_models(models, scaler, features):
    features_scaled = scaler.transform(features.reshape(1, -1))
    predictions = {}
    confidences = {}

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_scaled)
            pred_class = np.argmax(proba, axis=1)[0]
            predictions[name] = pred_class
            confidences[name] = proba[0][pred_class]
        else:
            pred_class = model.predict(features_scaled)[0]
            predictions[name] = pred_class
            confidences[name] = 1.0

    return predictions, confidences

# Function to evaluate with AI model
def evaluate_with_ai_model(model, scaler, features):
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(features_scaled)
    result = "Alzheimer's" if prediction[0][0] > 0.5 else "Healthy"
    confidence = prediction[0][0] if result == "Alzheimer's" else 1 - prediction[0][0]
    return result, confidence

# Function to display results
def display_results(ml_predictions, ml_confidences, ai_result, ai_confidence, ai_features):
    print("\033c", end="")  # Clear console
    print(f"{'Model':<20} {'Prediction':<15} {'Accuracy (%)':<15} {'Confidence (%)':<15}")
    print("-" * 65)

    for name, pred in ml_predictions.items():
        accuracy = MODEL_ACCURACIES.get(name, "N/A") * 100
        confidence = ml_confidences.get(name, 0) * 100
        print(f"{name:<20} {LABEL_MAPPING.get(pred, 'Unknown'):<15} {accuracy:<15.2f} {confidence:<15.2f}")

    ai_accuracy = MODEL_ACCURACIES.get("AI Model", "N/A") * 100
    print(f"\n{'AI Model':<20} {ai_result:<15} {ai_accuracy:<15.2f} {ai_confidence * 100:<15.2f}")

    # Display factors for AI model
    print("\nFactors (AI Model):")
    print(f"Theta Power: {ai_features[0]:.4f}")
    print(f"Alpha Power: {ai_features[1]:.4f}")
    print(f"Spatial Factor: {ai_features[2]:.4f}")
    print(f"Complexity Factor: {ai_features[3]:.4f}")
    print(f"Hemisphere Synchrony: {ai_features[4]:.4f}")

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
                ml_features = extract_ml_features(eeg_data)
                ai_features = extract_ai_features(eeg_data)

                # Evaluate models
                ml_predictions, ml_confidences = evaluate_with_ml_models(ml_models, ml_scaler, ml_features)
                ai_result, ai_confidence = evaluate_with_ai_model(ai_model, ai_scaler, ai_features)

                # Display results
                display_results(ml_predictions, ml_confidences, ai_result, ai_confidence, ai_features)

                if current_time - forced_update_time >= 10:
                    forced_update_time = current_time

    except KeyboardInterrupt:
        print("\nStopping EEG stream.")
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
