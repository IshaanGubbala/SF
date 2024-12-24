"""
server.py

This script implements a Flask web server that:
  - Receives incoming raw EEG data from board_client.py via HTTP POST (/predict).
  - Preprocesses and cleans data (pyprep, MNE).
  - Extracts features and runs a pre-trained Keras model for prediction.
  - Displays the latest prediction results at the root endpoint (/).
"""

import os
import numpy as np
import pandas as pd
import mne
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from pyprep.prep_pipeline import PrepPipeline
from tabulate import tabulate

# ------------------------------
# Flask App Initialization
# ------------------------------
app = Flask(__name__)

# ------------------------------
# Configuration
# ------------------------------
CHANNELS = ["Fp1", "Fp2", "C3", "C4"]
SAMPLING_RATE = 500
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

# ------------------------------
# Load Model & Scaler Once
# ------------------------------
# Make sure model.keras & scaler.pkl are in the same folder or provide full paths
model_path = "model.keras"
scaler_path = "scaler.pkl"

print("Loading model...")
model = load_model(model_path)
print("Loading scaler...")
scaler = pd.read_pickle(scaler_path)
print("Model and scaler loaded successfully.")

# Store the latest prediction table in memory to display at the homepage
latest_prediction_table_html = ""


# ------------------------------
# Helper Functions
# ------------------------------
def create_custom_montage(channel_names):
    """
    Create a custom montage for MNE using approximate 3D positions 
    for the specified channels.
    """
    from mne.channels import make_dig_montage

    # Example approximate 3D positions in meters
    channel_positions = {
        "Fp1": [-0.03, 0.08, 0],
        "Fp2": [0.03, 0.08, 0],
        "C3":  [-0.05, 0.0, 0],
        "C4":  [0.05, 0.0, 0],
    }

    montage = make_dig_montage(ch_pos=channel_positions, coord_frame="head")
    return montage

def apply_prep_pipeline(raw):
    """
    Apply PREP pipeline (pyprep) to clean EEG data.
    """
    try:
        montage = create_custom_montage(CHANNELS)
        raw.set_montage(montage)

        prep_params = {
            "ref_chs": "average",  # Use average reference
            "reref_chs": "all",    # Rereference all channels
            "line_freqs": [60],    # Notch filter at 60 Hz (USA power line)
        }
        prep = PrepPipeline(raw, prep_params, montage=None)
        prep.fit()
        return prep.raw
    except ValueError as e:
        print(f"Error in PREP pipeline: {e}")
        # If pipeline fails, just return raw unmodified
        return raw

def compute_band_powers(data, sampling_rate):
    """
    Compute average band powers for each band in FREQUENCY_BANDS.
    data shape: (n_epochs, n_channels, n_times)
    """
    psd, freqs = mne.time_frequency.psd_array_multitaper(
        data, sfreq=sampling_rate, verbose=False
    )
    band_powers = {}
    for band, (fmin, fmax) in FREQUENCY_BANDS.items():
        idx_band = (freqs >= fmin) & (freqs <= fmax)
        # Mean PSD in the frequency band, shape = (n_epochs, n_channels)
        band_powers[band] = np.mean(psd[:, :, idx_band], axis=2)

    # Average across epochs and channels if desired
    band_powers_mean = {band: np.mean(band_powers[band], axis=0) for band in band_powers}
    return band_powers_mean

def compute_epoch_entropy(data):
    """
    Compute Shannon entropy across epochs.
    data shape: (n_epochs, n_channels, n_times)
    """
    entropies = []
    for epoch in data:
        # Flatten channels, or treat separately. Here we flatten all channels/time
        flattened = epoch.flatten()
        # Probability distribution
        p = np.abs(flattened) / np.sum(np.abs(flattened))
        # Avoid log(0) by adding a small constant
        entropy = -np.sum(p * np.log2(p + 1e-12))
        entropies.append(entropy)
    # Return mean of entropies across all epochs
    return np.mean(entropies)

def compute_spatial_complexity(data):
    """
    Spatial complexity: variance of channel variances.
    data shape: (n_epochs, n_channels, n_times)
    We'll flatten across epochs and times for each channel.
    """
    # Combine all epochs into one array for each channel
    combined = data.reshape(data.shape[0]*data.shape[2], data.shape[1])  # shape: (total_samples, n_channels)
    channel_variances = np.var(combined, axis=0)  # variance for each channel
    return np.var(channel_variances)

def compute_temporal_complexity(data):
    """
    Temporal complexity: mean variance across channels over time.
    data shape: (n_epochs, n_channels, n_times)
    We'll consider each channel's variance across time, average over channels.
    """
    # Combine all epochs for each channel
    combined = data.reshape(data.shape[0]*data.shape[2], data.shape[1]).T  # shape: (n_channels, total_samples)
    temporal_variances = np.var(combined, axis=1)  # variance for each channel over time
    return np.mean(temporal_variances)

def extract_features(raw):
    """
    Convert Raw data into epochs, then compute band powers, entropy, 
    and complexity measures.
    """
    # Ensure data is loaded
    raw.load_data()

    # Apply PREP pipeline
    raw_cleaned = apply_prep_pipeline(raw)

    # Create fixed-length epochs of 1 second with 0.5s overlap
    epochs = mne.make_fixed_length_epochs(
        raw_cleaned, duration=1.0, overlap=0.5, verbose=False
    )
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

    # Compute band powers
    band_powers_mean = compute_band_powers(data, SAMPLING_RATE)

    # Compute advanced features
    entropy = compute_epoch_entropy(data)
    spatial_complexity = compute_spatial_complexity(data)
    temporal_complexity = compute_temporal_complexity(data)

    # Flatten into a single feature vector
    features = np.hstack([
        np.mean(band_powers_mean[band]) for band in FREQUENCY_BANDS
    ] + [
        entropy, 
        spatial_complexity, 
        temporal_complexity
    ])

    return features


# ------------------------------
# Flask Routes
# ------------------------------
@app.route("/")
def index():
    """
    Display the latest prediction table (HTML) if available, 
    or a message if no prediction has been made yet.
    """
    global latest_prediction_table_html

    if latest_prediction_table_html:
        return f"<h1>Latest Prediction</h1>{latest_prediction_table_html}"
    else:
        return "<h1>No predictions made yet.</h1>"


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive raw EEG data (JSON), convert to MNE Raw, extract features, 
    scale them, run them through the model, and store/display results.
    """
    global latest_prediction_table_html

    # Validate JSON
    content = request.json
    if not content or "data" not in content:
        return jsonify({"error": "No data found in request"}), 400

    # Convert data to numpy array
    try:
        eeg_data = np.array(content["data"], dtype=np.float32)  # shape: (channels, samples)
    except Exception as e:
        return jsonify({"error": f"Failed to parse EEG data: {e}"}), 400

    # Construct MNE Raw object
    info = mne.create_info(ch_names=CHANNELS, sfreq=SAMPLING_RATE, ch_types="eeg")
    raw = mne.io.RawArray(eeg_data, info)

    # Extract features
    features = extract_features(raw)
    if features is None:
        return jsonify({"error": "Feature extraction returned None"}), 500

    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Run model prediction
    prediction_value = model.predict(features_scaled)[0, 0]

    # Build table data for display
    table_data = [
        [label, round(val, 5)] for label, val in zip(FEATURE_LABELS, features)
    ]
    table_data.append(["Prediction", round(prediction_value, 5)])

    # Convert table to HTML using tabulate
    latest_prediction_table_html = tabulate(
        table_data, headers=["Feature", "Value"], tablefmt="html"
    )

    return jsonify({
        "prediction": float(prediction_value),
        "table": latest_prediction_table_html
    }), 200


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    # You can customize the host/port here
    app.run(debug=True, host="192.168.5.71", port=5000)
