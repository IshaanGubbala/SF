"""
eeg_server_energy_landscape.py

Run the server with:
  uvicorn eeg_server_energy_landscape:app --host 0.0.0.0 --port 8000
"""

import os
import numpy as np
import mne
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
import uvicorn
import warnings
import asyncio
from collections import deque
from scipy.interpolate import griddata

# For CORS
from fastapi.middleware.cors import CORSMiddleware

# ------------------ CONFIGURATION -------------------------------------------

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings if any

# Frequency bands (unchanged from your original)
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

# Feature Names (unchanged)
FEATURE_NAMES = list(FREQUENCY_BANDS.keys()) + [
    "Alpha_Ratio", "Theta_Ratio",
    "Shannon_Entropy",
    "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity",
    "Spatial_Complexity"
]

# Paths
MODELS_DIR = "trained_models"
MLP_MODEL_PATH = os.path.join(MODELS_DIR, "mlp_retrained.joblib")
MLP_SCALER_PATH = os.path.join(MODELS_DIR, "mlp_retrained_scaler.joblib")
PCA_MODEL_PATH = os.path.join(MODELS_DIR, "pca_model.joblib")

# EEG streaming config
SAMPLING_RATE = 256
WINDOW_SIZE = 20
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SIZE

# Energy landscape config
GRID_RESOLUTION = 50
ENERGY_RANGE = 1.0

# ------------------ FEATURE EXTRACTION --------------------------------------

def compute_band_powers(data: np.ndarray, sfreq: float) -> Dict[str, float]:
    psd, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=sfreq, verbose=False)
    band_powers = {}
    for band, (fmin, fmax) in FREQUENCY_BANDS.items():
        idx_band = (freqs >= fmin) & (freqs <= fmax)
        # Mean over channels and frequencies
        band_powers[band] = np.mean(psd[:, idx_band], axis=1).mean()
    return band_powers

def compute_shannon_entropy(data: np.ndarray) -> float:
    flattened = data.flatten()
    counts, _ = np.histogram(flattened, bins=256, density=True)
    probs = counts / np.sum(counts)
    entropy_val = -np.sum(probs * np.log2(probs + 1e-12))
    return entropy_val

def compute_hjorth_parameters(data: np.ndarray) -> Dict[str, float]:
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
    band_powers = compute_band_powers(data, SAMPLING_RATE)
    
    alpha_power = band_powers.get("Alpha1", 0.0) + band_powers.get("Alpha2", 0.0)
    theta_power = band_powers.get("Theta1", 0.0) + band_powers.get("Theta2", 0.0)
    total_power = sum(band_powers.values()) + 1e-12
    alpha_ratio = alpha_power / total_power
    theta_ratio = theta_power / total_power
    
    shannon_entropy = compute_shannon_entropy(data)
    hjorth_params = compute_hjorth_parameters(data)
    
    # If you compute spatial_complexity somewhere, put code here. Otherwise, 0.0 for placeholder
    spatial_complexity = 0.0
    
    features = {band: band_powers.get(band, 0.0) for band in FREQUENCY_BANDS.keys()}
    features["Alpha_Ratio"] = alpha_ratio
    features["Theta_Ratio"] = theta_ratio
    features["Shannon_Entropy"] = shannon_entropy
    features.update(hjorth_params)
    features["Spatial_Complexity"] = spatial_complexity
    
    return features

def compute_feature_stats(features: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for k, v in features.items():
        stats[k] = {
            "mean": v,
            "std": 0.0
        }
    return stats

# ------------------ ENERGY LANDSCAPE ----------------------------------------

def compute_energy_weights(features: List[str]) -> Dict[str, float]:
    # Assign weight=1.0 for every feature
    return {f: 1.0 for f in features}

def compute_energy(features: Dict[str, float], weights: Dict[str, float]) -> float:
    val = 0.0
    for k, w in weights.items():
        val += features.get(k, 0.0) * w
    return val

def load_or_fit_pca(features: np.ndarray, n_components: int = 2) -> PCA:
    """
    Attempts to load an existing PCA model; if none is found, it fits a new one.
    """
    if os.path.exists(PCA_MODEL_PATH):
        pca_model = joblib.load(PCA_MODEL_PATH)
        print("[INFO] Loaded existing PCA model.")
    else:
        pca_model = PCA(n_components=n_components)
        pca_model.fit(features)
        joblib.dump(pca_model, PCA_MODEL_PATH)
        print("[INFO] Fitted and saved new PCA model.")
    return pca_model

def generate_energy_landscape_grid(pca: PCA, principal_components: np.ndarray, energy_vals: np.ndarray) -> Dict[str, Any]:
    """
    Generates grid data for the 2D PCA + energy landscape.
    """
    x_min, x_max = principal_components[:,0].min() - ENERGY_RANGE, principal_components[:,0].max() + ENERGY_RANGE
    y_min, y_max = principal_components[:,1].min() - ENERGY_RANGE, principal_components[:,1].max() + ENERGY_RANGE
    xi = np.linspace(x_min, x_max, GRID_RESOLUTION)
    yi = np.linspace(y_min, y_max, GRID_RESOLUTION)
    xi, yi = np.meshgrid(xi, yi)
    
    zi = griddata(principal_components, energy_vals, (xi, yi), method='cubic')
    # Replace any NaNs with the min value in zi
    zi = np.nan_to_num(zi, nan=np.nanmin(zi))
    
    grid_data = {
        "x": xi.tolist(),
        "y": yi.tolist(),
        "z": zi.tolist()
    }
    return grid_data

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
    pc1: float
    pc2: float
    energy: float

class EnergyLandscape(BaseModel):
    x: List[List[float]]
    y: List[List[float]]
    z: List[List[float]]

# ------------------ FASTAPI APPLICATION -------------------------------------

app = FastAPI(
    title="Continuous EEG Prediction Server with Energy Landscape",
    description="Streams EEG predictions and sends a 3D energy landscape to clients.",
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
    mlp_model = joblib.load(MLP_MODEL_PATH)
    mlp_scaler = joblib.load(MLP_SCALER_PATH)
    print("[INFO] MLP model and scaler loaded.")
except Exception as e:
    mlp_model = None
    mlp_scaler = None
    print(f"[ERROR] Failed to load MLP model/scaler: {e}")

# For PCA, we simulate or load training features to either load or fit PCA
def load_training_features() -> np.ndarray:
    """
    In your real scenario, load actual training features from a file.
    For demonstration, we just generate random data here.
    """
    np.random.seed(42)
    sim_data = np.random.rand(1000, len(FEATURE_NAMES))
    return sim_data

training_features = load_training_features()
pca = load_or_fit_pca(training_features, n_components=2)
weights = compute_energy_weights(FEATURE_NAMES)

# Generate "training_energy" for precomputed energy landscape
training_energy = []
for row in training_features:
    ft_dict = {FEATURE_NAMES[i]: row[i] for i in range(len(FEATURE_NAMES))}
    e_val = compute_energy(ft_dict, weights)
    training_energy.append(e_val)

training_energy = np.array(training_energy)
pc_data = pca.transform(training_features)

# Pre-generate the 2D grid for the energy landscape
energy_landscape_data = generate_energy_landscape_grid(pca, pc_data, training_energy)

# ------------------ CONNECTION MANAGER --------------------------------------

class ConnectionManager:
    """
    Manages client connections, buffering data for each connection,
    and sends predictions back to the client.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.buffers: Dict[WebSocket, deque] = {}
        self.lock = asyncio.Lock()
        # Store our precomputed grid data so we don't compute it repeatedly
        self.grid_data = energy_landscape_data

    async def connect(self, websocket: WebSocket):
        """
        Accepts a WebSocket connection and sends the precomputed energy landscape.
        """
        await websocket.accept()
        async with self.lock:
            self.active_connections.append(websocket)
            self.buffers[websocket] = deque(maxlen=WINDOW_SAMPLES)
        print(f"[INFO] Client connected: {websocket.client}")
        
        # Send the energy landscape to the client immediately
        landscape = EnergyLandscape(**self.grid_data)
        await websocket.send_json(landscape.dict())
        print(f"[INFO] Energy landscape sent to {websocket.client}")

    async def disconnect(self, websocket: WebSocket):
        """
        Removes the WebSocket connection.
        """
        async with self.lock:
            self.active_connections.remove(websocket)
            del self.buffers[websocket]
        print(f"[INFO] Client disconnected: {websocket.client}")

    async def receive_data(self, websocket: WebSocket, eeg_data: EEGData):
        """
        Called whenever new EEG data arrives from the client.
        Buffers the data, and when we have enough samples, makes a prediction.
        """
        buffer = self.buffers.get(websocket)
        if buffer is None:
            print("[WARNING] Received data from unknown websocket.")
            return
        
        # Accumulate new samples
        for sample in eeg_data.data:
            buffer.append(sample.channels)
        
        # If we have enough samples (>= WINDOW_SAMPLES), process them
        if len(buffer) >= WINDOW_SAMPLES:
            window_data = np.array(buffer).T  # shape: (n_channels, n_times)
            buffer.clear()  # reset the buffer
            
            # Extract features
            feats = extract_features(window_data)
            feat_vec = np.array([feats[f] for f in FEATURE_NAMES]).reshape(1, -1)
            
            # Scale features, then predict using MLP
            if mlp_model is not None and mlp_scaler is not None:
                scaled_vec = mlp_scaler.transform(feat_vec)
                
                prediction = mlp_model.predict(scaled_vec)[0]
                # If binary classification, confidence is probability of class=1
                confidence = mlp_model.predict_proba(scaled_vec)[0][1]
            else:
                # If we failed to load the MLP, no predictions can be made
                prediction = 0
                confidence = 0.0
            
            # Compute stats for reference
            stats = compute_feature_stats(feats)
            
            # Compute energy
            e_val = compute_energy(feats, weights)
            
            # Project to PCA space
            pc = pca.transform(feat_vec)[0]
            pc1, pc2 = float(pc[0]), float(pc[1])
            
            # Create the response
            response = PredictionResponse(
                prediction=int(prediction),
                confidence=float(confidence),
                features=feats,
                stats=stats,
                pc1=pc1,
                pc2=pc2,
                energy=e_val
            )
            
            # Send JSON back to the client
            try:
                await websocket.send_json(response.dict())
                print(f"[INFO] Sent prediction to {websocket.client}. Prediction={prediction}, Confidence={confidence:.2f}")
            except Exception as e:
                print(f"[ERROR] Failed to send prediction: {e}")

manager = ConnectionManager()

# ------------------ WEBSOCKET ENDPOINT --------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint that continually listens for EEG data,
    processes it, and sends back predictions.
    """
    await manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            try:
                # Attempt to parse as EEGData
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
