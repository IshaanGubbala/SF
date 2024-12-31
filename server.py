import os
import numpy as np
import mne
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
import uvicorn
import warnings
import asyncio
from collections import deque

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings if any

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------

# Define frequency bands as in the original script
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

# Define feature names as in the original script, including Spatial_Complexity
FEATURE_NAMES = list(FREQUENCY_BANDS.keys()) + [
    "Alpha_Ratio", "Theta_Ratio",
    "Shannon_Entropy",
    "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity",
    "Spatial_Complexity"  # Included as per requirement
]

# Paths to the trained model and scaler
MODELS_DIR = "trained_models"
LOGREG_MODEL_PATH = os.path.join(MODELS_DIR, "logreg_retrained.joblib")
LOGREG_SCALER_PATH = os.path.join(MODELS_DIR, "logreg_retrained_scaler.joblib")

# Sampling rate (Hz)
SAMPLING_RATE = 256

# Window size (seconds)
WINDOW_SIZE = 20  # 20 seconds
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SIZE  # Number of samples per window

# --------------------------------------------------------------------------------
# FEATURE EXTRACTION FUNCTIONS
# --------------------------------------------------------------------------------

def compute_band_powers(data: np.ndarray, sfreq: float) -> Dict[str, float]:
    """
    Compute average power in each frequency band.
    Returns a dictionary with band names as keys and mean power as values.
    """
    psd, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=sfreq, verbose=False)
    band_powers = {}
    for band, (fmin, fmax) in FREQUENCY_BANDS.items():
        idx_band = (freqs >= fmin) & (freqs <= fmax)
        band_powers[band] = np.mean(psd[:, idx_band], axis=1).mean()  # Mean over channels and frequencies
    return band_powers

def compute_shannon_entropy(data: np.ndarray) -> float:
    """
    Compute Shannon entropy for the entire data.
    Returns the entropy value.
    """
    flattened = data.flatten()
    counts, _ = np.histogram(flattened, bins=256, density=True)
    probs = counts / np.sum(counts)
    entropy = -np.sum(probs * np.log2(probs + 1e-12))  # Add epsilon to avoid log(0)
    return entropy

def compute_hjorth_parameters(data: np.ndarray) -> Dict[str, float]:
    """
    Compute Hjorth Activity, Mobility, and Complexity for the entire data.
    Returns a dictionary with the mean of each parameter.
    """
    # Hjorth parameters are computed across all channels and epochs
    activity = np.var(data) + 1e-12  # Avoid division by zero
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
    Extract features from EEG data.
    Args:
        data: NumPy array of shape (n_channels, n_times)
    Returns:
        features: Dictionary of feature names and their computed values.
    """
    # Compute band powers
    band_powers = compute_band_powers(data, SAMPLING_RATE)
    
    # Compute ratios
    alpha_power = band_powers.get("Alpha1", 0.0) + band_powers.get("Alpha2", 0.0)
    theta_power = band_powers.get("Theta1", 0.0) + band_powers.get("Theta2", 0.0)
    total_power = sum(band_powers.values()) + 1e-12  # Avoid division by zero
    alpha_ratio = alpha_power / total_power
    theta_ratio = theta_power / total_power
    
    # Compute Shannon Entropy
    shannon_entropy = compute_shannon_entropy(data)
    
    # Compute Hjorth Parameters
    hjorth_params = compute_hjorth_parameters(data)
    
    # Set Spatial_Complexity to 0 as per user request
    spatial_complexity = 0.0  # Fixed value
    
    # Aggregate features
    features = {band: band_powers.get(band, 0.0) for band in FREQUENCY_BANDS.keys()}
    features["Alpha_Ratio"] = alpha_ratio
    features["Theta_Ratio"] = theta_ratio
    features["Shannon_Entropy"] = shannon_entropy
    features.update(hjorth_params)
    features["Spatial_Complexity"] = spatial_complexity  # Add the fixed Spatial_Complexity
    
    return features

def compute_feature_stats(features: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    Compute statistical summaries of features.
    Args:
        features: Dictionary of feature names and their computed values.
    Returns:
        stats: Dictionary of statistical summaries (mean, std) for each feature.
    """
    stats = {}
    for feature, value in features.items():
        stats[feature] = {
            "mean": value,  # Since features are already aggregated, mean is the value itself
            "std": 0.0       # Standard deviation is not applicable here
        }
    return stats

# --------------------------------------------------------------------------------
# Pydantic Models for WebSocket Messages and Responses
# --------------------------------------------------------------------------------

class EEGSample(BaseModel):
    channels: List[float]  # List of channel values for a single sample

class EEGData(BaseModel):
    data: List[EEGSample]  # List of samples

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    features: Dict[str, float]
    stats: Dict[str, Dict[str, float]]

# --------------------------------------------------------------------------------
# Initialize FastAPI App
# --------------------------------------------------------------------------------

app = FastAPI(
    title="Continuous EEG Prediction Server",
    description="A server that makes real-time predictions on continuous EEG data streams.",
    version="1.0.0"
)

# --------------------------------------------------------------------------------
# Load Trained Model and Scaler at Startup
# --------------------------------------------------------------------------------

try:
    logreg_model = joblib.load(LOGREG_MODEL_PATH)
    logreg_scaler = joblib.load(LOGREG_SCALER_PATH)
    print("[INFO] Logistic Regression model and scaler loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model or scaler: {e}")
    logreg_model = None
    logreg_scaler = None

# --------------------------------------------------------------------------------
# WebSocket Connection Manager
# --------------------------------------------------------------------------------

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

    async def receive_data(self, websocket: WebSocket, data: EEGData):
        buffer = self.buffers.get(websocket)
        if buffer is None:
            print("[WARNING] Received data from unknown connection.")
            return

        for sample in data.data:
            buffer.append(sample.channels)
        
        if len(buffer) >= WINDOW_SAMPLES:
            # Extract the last WINDOW_SAMPLES
            window_data = np.array(buffer).T  # Shape: (n_channels, n_times)
            buffer.clear()  # Reset buffer after processing

            # Feature Extraction
            features = extract_features(window_data)
            
            # Prepare feature vector in the correct order
            feature_vector = np.array([features[feature] for feature in FEATURE_NAMES]).reshape(1, -1)
            
            # Feature Scaling
            scaled_features = logreg_scaler.transform(feature_vector)
            
            # Make Prediction
            prediction = logreg_model.predict(scaled_features)[0]
            confidence = logreg_model.predict_proba(scaled_features)[0][1]  # Probability for class 1
            
            # Compute Stats
            stats = compute_feature_stats(features)
            
            # Prepare response
            response = PredictionResponse(
                prediction=int(prediction),
                confidence=float(confidence),
                features=features,
                stats=stats
            )
            
            # Send response back to client
            try:
                await websocket.send_json(response.dict())
                print(f"[INFO] Sent prediction to {websocket.client}: Prediction={prediction}, Confidence={confidence:.2f}")
            except Exception as e:
                print(f"[ERROR] Failed to send prediction: {e}")

manager = ConnectionManager()

# --------------------------------------------------------------------------------
# WebSocket Endpoint
# --------------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive data from client
            message = await websocket.receive_text()
            try:
                # Parse JSON message to EEGData
                eeg_data = EEGData.parse_raw(message)
                # Process received data
                await manager.receive_data(websocket, eeg_data)
            except ValidationError as ve:
                error_message = {"error": f"Validation Error: {ve}"}
                await websocket.send_json(error_message)
                print(f"[ERROR] Validation error: {ve}")
            except Exception as e:
                error_message = {"error": f"Failed to process data: {e}"}
                await websocket.send_json(error_message)
                print(f"[ERROR] Failed to process data: {e}")
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        await manager.disconnect(websocket)
        print(f"[ERROR] Unexpected error: {e}")

# --------------------------------------------------------------------------------
# CORS Middleware (Optional)
# --------------------------------------------------------------------------------

from fastapi.middleware.cors import CORSMiddleware

# Allow all origins (you can restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------
# Run the Server
# --------------------------------------------------------------------------------

# To run the server, use the following command in your terminal:
# uvicorn eeg_server_continuous_fixed:app --host 0.0.0.0 --port 8000

# Ensure that this script is saved as, for example, 'eeg_server_continuous_fixed.py'
# and run the server using:
# uvicorn eeg_server_continuous_fixed:app --host 0.0.0.0 --port 8000
