"""
eeg_server.py

Run the server with:
  uvicorn eeg_server:app --host 0.0.0.0 --port 8000
"""

import os
import numpy as np
import mne
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler
import uvicorn
import warnings
import asyncio
from collections import deque

# For CORS
from fastapi.middleware.cors import CORSMiddleware

# ------------------ CONFIGURATION -------------------------------------------

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings if any

# Frequency bands
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

# Feature Names
FEATURE_NAMES = list(FREQUENCY_BANDS.keys()) + [
    "Alpha_Ratio", "Theta_Ratio",
    "Shannon_Entropy",
    "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity",
    "Spatial_Complexity"
]

# Paths
MODELS_DIR = "trained_models"
LOGREG_MODEL_PATH = os.path.join(MODELS_DIR, "logreg_retrained.joblib")
LOGREG_SCALER_PATH = os.path.join(MODELS_DIR, "logreg_retrained_scaler.joblib")

# EEG streaming config
SAMPLING_RATE = 200
WINDOW_SIZE = 20
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SIZE

# ------------------ FEATURE EXTRACTION --------------------------------------

def compute_band_powers(data: np.ndarray, sfreq: float) -> Dict[str, float]:
    """
    Compute band power for each frequency band using multitaper PSD.
    """
    psd, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=sfreq, verbose=False)
    band_powers = {}
    for band, (fmin, fmax) in FREQUENCY_BANDS.items():
        idx_band = (freqs >= fmin) & (freqs <= fmax)
        # Mean over channels and frequencies
        band_powers[band] = np.mean(psd[:, idx_band], axis=1).mean()
    return band_powers

def compute_shannon_entropy(data: np.ndarray) -> float:
    """
    Compute Shannon Entropy of the data.
    """
    flattened = data.flatten()
    counts, _ = np.histogram(flattened, bins=256, density=True)
    probs = counts / np.sum(counts)
    entropy_val = -np.sum(probs * np.log2(probs + 1e-12))
    return entropy_val

def compute_hjorth_parameters(data: np.ndarray) -> Dict[str, float]:
    """
    Compute Hjorth parameters (Activity, Mobility, Complexity).
    """
    activity = np.var(data) + 1e-12
    first_derivative = np.diff(data, axis=-1)
    mobility = np.sqrt(np.var(first_derivative) / activity)
    second_derivative = np.diff(first_derivative, axis=-1)
    complexity = np.sqrt(np.var(second_derivative) / (np.var(first_derivative) + 1e-12))
    return {
        "Hjorth_Activity": float(activity),
        "Hjorth_Mobility": float(mobility),
        "Hjorth_Complexity": float(complexity)
    }

def extract_features(data: np.ndarray) -> Dict[str, Any]:
    """
    Extract a dictionary of features from EEG data.
    """
    # Frequency-band features
    band_powers = compute_band_powers(data, SAMPLING_RATE)
    alpha_power = band_powers.get("Alpha1", 0.0) + band_powers.get("Alpha2", 0.0)
    theta_power = band_powers.get("Theta1", 0.0) + band_powers.get("Theta2", 0.0)
    total_power = sum(band_powers.values()) + 1e-12
    alpha_ratio = alpha_power / total_power
    theta_ratio = theta_power / total_power

    # Additional features
    shannon_entropy = compute_shannon_entropy(data)
    hjorth_params = compute_hjorth_parameters(data)

    # Example: spatial complexity dummy placeholder (replace with your own logic if needed)
    spatial_complexity = 0.0

    # Build feature dictionary
    features = {band: band_powers.get(band, 0.0) for band in FREQUENCY_BANDS.keys()}
    features["Alpha_Ratio"] = alpha_ratio
    features["Theta_Ratio"] = theta_ratio
    features["Shannon_Entropy"] = shannon_entropy
    features.update(hjorth_params)
    features["Spatial_Complexity"] = spatial_complexity

    return features

def compute_feature_stats(features: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    Create a stats dictionary (mean/std) for each feature. 
    Here, we just fill in the values as the "mean" and zero for "std".
    """
    stats = {}
    for k, v in features.items():
        stats[k] = {
            "mean": v,
            "std": 0.0
        }
    return stats

# ------------------ Pydantic Models -----------------------------------------

class EEGSample(BaseModel):
    channels: List[float]

class EEGData(BaseModel):
    data: List[EEGSample]

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    features: Dict[str, float]
    stats: Dict[str, Dict[str, float]]

# ------------------ FASTAPI APPLICATION -------------------------------------

app = FastAPI(
    title="Continuous EEG Prediction Server",
    description="Streams EEG predictions with extracted features for real-time visualization.",
    version="1.0.0"
)

# CORS - allowing all for demonstration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ LOAD MODELS ON STARTUP ----------------------------------

try:
    logreg_model = joblib.load(LOGREG_MODEL_PATH)
    logreg_scaler = joblib.load(LOGREG_SCALER_PATH)
    print("[INFO] Logistic Regression model and scaler loaded.")
except Exception as e:
    logreg_model = None
    logreg_scaler = None
    print(f"[ERROR] Failed to load model/scaler: {e}")

# ------------------ CONNECTION MANAGER --------------------------------------

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.buffers: Dict[WebSocket, deque] = {}
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections.append(websocket)
            self.buffers[websocket] = deque(maxlen=WINDOW_SAMPLES)
        print(f"[INFO] Client connected: {websocket.client}")

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            self.active_connections.remove(websocket)
            del self.buffers[websocket]
        print(f"[INFO] Client disconnected: {websocket.client}")

    async def receive_data(self, websocket: WebSocket, eeg_data: EEGData):
        buffer = self.buffers.get(websocket)
        if buffer is None:
            print("[WARNING] Received data from unknown websocket.")
            return
        
        # Accumulate the channels
        for sample in eeg_data.data:
            buffer.append(sample.channels)
        
        # Once the buffer has enough samples, process them
        if len(buffer) >= WINDOW_SAMPLES:
            window_data = np.array(buffer).T  # shape: (n_channels, n_times)
            buffer.clear()
            
            feats = extract_features(window_data)
            feat_vec = np.array([feats[f] for f in FEATURE_NAMES]).reshape(1, -1)
            
            # Scale -> Predict
            if logreg_model is not None and logreg_scaler is not None:
                scaled_vec = logreg_scaler.transform(feat_vec)
                prediction = logreg_model.predict(scaled_vec)[0]
                confidence = float(logreg_model.predict_proba(scaled_vec)[0][1])
            else:
                prediction = -1
                confidence = 0.0

            # Compute stats
            stats = compute_feature_stats(feats)
            
            response = PredictionResponse(
                prediction=int(prediction),
                confidence=confidence,
                features=feats,
                stats=stats
            )
            try:
                await websocket.send_json(response.dict())
                print(f"[INFO] Sent prediction to {websocket.client}. "
                      f"Prediction={prediction}, Confidence={confidence:.2f}")
            except Exception as e:
                print(f"[ERROR] Failed to send prediction: {e}")

manager = ConnectionManager()

# ------------------ WEBSOCKET ENDPOINT --------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            try:
                # Parse as EEGData
                data = EEGData.parse_raw(message)
                await manager.receive_data(websocket, data)
            except ValidationError as ve:
                err_msg = {"error": f"Validation Error: {ve}"}
                await websocket.send_json(err_msg)
            except Exception as e:
                err_msg = {"error": f"Failed to process data: {e}"}
                await websocket.send_json(err_msg)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        await manager.disconnect(websocket)
        print(f"[ERROR] Unexpected error: {e}")

# ------------------ MAIN RUN -----------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
