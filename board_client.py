"""
board_client.py

This client:
 • Connects to ws://localhost:8000/ws.
 • Uses BrainFlow to capture 1-second chunks (256 samples) from a Ganglion Native board.
 • Assumes the board returns 5 channels:
      - If 5 channels: channel 0 is the hardware reference and channels 1–4 are EEG (F3, F4, P3, O2).
      - If only 4 channels are available: all channels are EEG.
 • Processing pipeline (original order):
      1. Filter each channel (detrend, bandpass 1–50 Hz using a Butterworth filter of order 2, plus moving-average smoothing).
      2. Compute the reference used for cleaning:
             - If ≥5 channels: use channel 0.
             - If 4 channels: compute synthetic reference as the median.
      3. Compute cleaned EEG = (filtered EEG) – (filtered reference).
      4. For display, compute Total EEG = average(filtered unreferenced EEG) and show it in the top subplot.
         For each electrode (F3, F4, P3, O2), overlay:
             • The raw filtered (unreferenced) signal (opaque).
             • The cleaned signal (translucent),
             with a large red dot at the latest cleaned sample.
      5. Send only the cleaned 4-channel EEG to the server.
 • Also displays a 3D energy attractor (using an 8×8 grid) with a big red dot at the current state.
 • Uses a disappearing prediction console (in the main window) to show model predictions and important features.
 • Runs asynchronous BrainFlow/WS tasks in a background thread while the Tkinter UI remains responsive.

Electrode placements (10–20 system):
  • F3: Left frontal.
  • F4: Right frontal.
  • P3: Left parietal.
  • O2: Right occipital.
  • The “Total EEG” is computed as the average of the filtered unreferenced EEG channels.
"""

import asyncio
import websockets
import json
import time
import numpy as np
import tkinter as tk
import threading
import os

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes, NoiseTypes
from pydantic import BaseModel
from typing import List, Dict, Any

# Configuration
SERVER_URI = "ws://localhost:8000/ws"
MAX_PLOT_SAMPLES = 256 * 5  # 5 seconds
SAMPLE_RATE = 256
CHUNK_SIZE = 256

# Pydantic models
class EEGSample(BaseModel):
    channels: List[float]

class EEGData(BaseModel):
    data: List[EEGSample]

class PredictionResponse(BaseModel):
    prediction_mlp: int
    confidence_mlp: float
    prediction_qsup: int
    confidence_qsup: float
    prediction_gcn: int
    confidence_gcn: float
    features: Dict[str, float]
    stats: Dict[str, Dict[str, float]]
    pc1: float
    pc2: float
    energy: float

#############################
# Main Window for EEG, 3D Attractor, and Prediction Console
#############################
class EEGMainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cleaned EEG & Energy Attractor")
        self.geometry("1300x900")
        
        # Frame for plots
        frame_plots = tk.Frame(self)
        frame_plots.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 2D Plots: one for Total EEG and one per electrode.
        self.electrode_labels = ["F3", "F4", "P3", "O2"]
        self.all_labels = ["Total EEG"] + self.electrode_labels
        self.fig_2d, self.axes_2d = plt.subplots(len(self.all_labels), 1, figsize=(6,10), sharex=True)
        self.fig_2d.tight_layout()
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=frame_plots)
        self.canvas_2d.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 3D Plot for energy attractor
        self.fig_3d = plt.Figure(figsize=(6,8))
        self.ax_3d = self.fig_3d.add_subplot(111, projection="3d")
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=frame_plots)
        self.canvas_3d.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Prediction console at the bottom (disappears after 5 seconds)
        self.prediction_console = tk.Label(self, text="", font=("Arial", 12), bg="lightgrey")
        self.prediction_console.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Data storage for each channel
        self.time_data = np.zeros(0, dtype=np.float32)
        self.channel_data = {"Total EEG": np.zeros(0, dtype=np.float32)}
        for lbl in self.electrode_labels:
            self.channel_data[f"{lbl}_raw"] = np.zeros(0, dtype=np.float32)
            self.channel_data[f"{lbl}_clean"] = np.zeros(0, dtype=np.float32)
        self.total_samples = 0

    def update_signals(self, combined: np.ndarray, raw_eeg: np.ndarray):
        """
        combined: shape (5, n_samples)
          Row 0: filtered reference (used for cleaning; not displayed directly).
          Rows 1-4: cleaned EEG signals.
        raw_eeg: shape (4, n_samples) filtered unreferenced EEG signals.
        For display, Total EEG = average(raw_eeg) across channels.
        """
        n = combined.shape[1]
        t_new = np.arange(self.total_samples, self.total_samples+n, dtype=np.float32) / SAMPLE_RATE
        self.total_samples += n
        self.time_data = np.concatenate((self.time_data, t_new))
        total_eeg = np.mean(raw_eeg, axis=0, keepdims=True)
        self.channel_data["Total EEG"] = np.concatenate((self.channel_data["Total EEG"], total_eeg[0]))
        for i, lbl in enumerate(self.electrode_labels):
            self.channel_data[f"{lbl}_clean"] = np.concatenate((self.channel_data[f"{lbl}_clean"], combined[i+1]))
            self.channel_data[f"{lbl}_raw"] = np.concatenate((self.channel_data[f"{lbl}_raw"], raw_eeg[i]))
        if self.time_data.size > MAX_PLOT_SAMPLES:
            diff = self.time_data.size - MAX_PLOT_SAMPLES
            self.time_data = self.time_data[diff:]
            self.channel_data["Total EEG"] = self.channel_data["Total EEG"][diff:]
            for lbl in self.electrode_labels:
                self.channel_data[f"{lbl}_clean"] = self.channel_data[f"{lbl}_clean"][diff:]
                self.channel_data[f"{lbl}_raw"] = self.channel_data[f"{lbl}_raw"][diff:]
        # Smooth signals for smoother UI
        for key in self.channel_data:
            self.channel_data[key] = smooth_signal(self.channel_data[key], window_size=5)

    def redraw_2d(self):
        # Plot Total EEG in subplot 0
        self.axes_2d[0].clear()
        self.axes_2d[0].plot(self.time_data, self.channel_data["Total EEG"], color="black")
        self.axes_2d[0].set_ylabel("Total EEG")
        # For each electrode, overlay raw and cleaned signals
        for i, lbl in enumerate(self.electrode_labels):
            ax = self.axes_2d[i+1]
            ax.clear()
            ax.plot(self.time_data, self.channel_data[f"{lbl}_raw"], color=f"C{i}", alpha=0.8, label="Raw")
            ax.plot(self.time_data, self.channel_data[f"{lbl}_clean"], color=f"C{i}", alpha=0.4, label="Cleaned")
            if self.channel_data[f"{lbl}_clean"].size > 0:
                ax.plot(self.time_data[-1], self.channel_data[f"{lbl}_clean"][-1], 'ro', markersize=12)
            ax.set_ylabel(lbl)
            ax.legend(loc="upper right")
        self.axes_2d[-1].set_xlabel("Time (sec)")
        self.fig_2d.tight_layout()
        self.canvas_2d.draw()

    def draw_3d_surface(self, x, y, z, current_state=None):
        self.ax_3d.clear()
        X = np.array(x)
        Y = np.array(y)
        Z = np.array(z)
        npts = len(X)
        side = int(np.floor(np.sqrt(npts)))
        npts_new = side * side
        X = X[:npts_new].reshape(side, side)
        Y = Y[:npts_new].reshape(side, side)
        Z = Z[:npts_new].reshape(side, side)
        self.ax_3d.plot_surface(X, Y, Z, cmap="viridis")
        if current_state is not None:
            self.ax_3d.scatter([current_state[0]], [current_state[1]], [current_state[2]], color="red", s=300)
        self.ax_3d.set_title("Energy Attractor")
        self.canvas_3d.draw()

    def update_prediction_console(self, text):
        self.prediction_console.config(text=text)
        # Clear the console after 5 seconds.
        self.after(5000, lambda: self.prediction_console.config(text=""))

#############################
# Smoothing helper: moving average
#############################
def smooth_signal(signal, window_size=5):
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode="same")

# Global GUI instance
main_app = EEGMainWindow()

#############################
# Processing pipeline: Filter first, then compute reference, then subtract.
#############################
def compute_reference_subtract_filter(raw_data: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Given raw_data (shape (N, n_samples) in volts) from BrainFlow,
    apply filtering on each channel:
         detrend, bandpass (1–50 Hz) using a Butterworth filter of order 2,
         then smooth using a moving average.
    Then compute the reference used for cleaning:
         - If N >= 5, use channel 0 (hardware reference).
         - If N == 4, compute synthetic reference as the median.
    Then compute cleaned EEG = (filtered EEG) - (filtered reference).
    Returns:
         cleaned: (4, n_samples) cleaned EEG.
         ref: (1, n_samples) filtered reference.
         unref: (4, n_samples) filtered unreferenced EEG.
    """
    data = np.copy(raw_data)
    for ch in range(data.shape[0]):
        DataFilter.detrend(data[ch], DetrendOperations.LINEAR.value)
        DataFilter.perform_bandpass(data[ch], SAMPLE_RATE, 1, 50, 2, FilterTypes.BUTTERWORTH.value, 0)
        data[ch] = smooth_signal(data[ch], window_size=5)
    if data.shape[0] >= 5:
        ref = data[0:1, :]
        unref = data[1:5, :]
    elif data.shape[0] == 4:
        unref = data
        ref = np.median(unref, axis=0, keepdims=True)
    else:
        raise ValueError("Not enough channels in raw_data.")
    cleaned = unref - ref
    return cleaned, ref, unref

#############################
# Async: receive messages from server
#############################
async def receive_server_messages(ws: websockets.WebSocketClientProtocol):
    try:
        async for msg in ws:
            data = json.loads(msg)
            if "energy_landscape" in data:
                en = data["energy_landscape"]
                main_app.after(0, lambda: main_app.draw_3d_surface(en["x"], en["y"], en["z"]))
            elif "prediction_mlp" in data:
                mlp_pred = data["prediction_mlp"]
                mlp_conf = data["confidence_mlp"]
                qsup_pred = data["prediction_qsup"]
                qsup_conf = data["confidence_qsup"]
                pc1 = data["pc1"]
                pc2 = data["pc2"]
                energy = data["energy"]
                features_text = ""
                if "features" in data and isinstance(data["features"], dict):
                    sorted_features = sorted(data["features"].items(), key=lambda kv: abs(kv[1]), reverse=True)
                    features_text = "\n".join(f"{k}: {v:.4f}" for k, v in sorted_features)
                pred_text = (f"MLP: {'Alzheimer' if mlp_pred==1 else 'Control'} ({mlp_conf*100:.1f}%)\n"
                             f"QSUP: {'Alzheimer' if qsup_pred==1 else 'Control'} ({qsup_conf*100:.1f}%)\n"
                             f"PC1 = {pc1:.2f}, PC2 = {pc2:.2f}\nEnergy = {energy:.2f}\n"
                             f"Important Features:\n{features_text}")
                main_app.after(0, lambda: main_app.update_prediction_console(pred_text))
                global energy_landscape_data
                main_app.after(0, lambda: main_app.draw_3d_surface(
                    energy_landscape_data["energy_landscape"]["x"],
                    energy_landscape_data["energy_landscape"]["y"],
                    energy_landscape_data["energy_landscape"]["z"],
                    current_state=(pc1, pc2, energy)
                ))
            elif "error" in data:
                print("[SERVER ERROR]:", data["error"])
            else:
                pass
    except websockets.exceptions.ConnectionClosed:
        print("[INFO] Server closed connection.")
    except Exception as e:
        print("[ERROR] Receiving loop crashed:", e)

#############################
# Async: read BrainFlow, process, and send to server
#############################
async def read_and_send(uri=SERVER_URI):
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes, NoiseTypes

    params = BrainFlowInputParams()
    BOARD_ID = BoardIds.GANGLION_NATIVE_BOARD.value  # Using Ganglion Native with hardware wiring returning 5 channels
    board = BoardShim(BOARD_ID, params)
    try:
        board.prepare_session()
        board.start_stream()
        time.sleep(1)
        print("[INFO] BrainFlow session started.")
    except Exception as e:
        print("[ERROR] BrainFlow init failed:", e)
        return

    try:
        async with websockets.connect(uri) as ws:
            print("[INFO] Connected to server.")
            asyncio.create_task(receive_server_messages(ws))
            while True:
                raw_data = board.get_current_board_data(CHUNK_SIZE) * 1e-3  # in volts
                try:
                    cleaned, ref_signal, unref = compute_reference_subtract_filter(raw_data)
                except Exception as e:
                    print("[ERROR] Montage computation:", e)
                    continue
                total_eeg = np.mean(unref, axis=0, keepdims=True)
                combined = np.vstack((total_eeg, cleaned))
                main_app.after(0, lambda: (main_app.update_signals(combined, unref), main_app.redraw_2d()))
                payload_samples = []
                for row in cleaned.T:
                    payload_samples.append(EEGSample(channels=row.tolist()))
                payload = EEGData(data=payload_samples)
                await ws.send(payload.json())
                await asyncio.sleep(1.0)
    except websockets.exceptions.ConnectionClosed as e:
        print("[INFO] Server closed connection:", e)
    except Exception as e:
        print("[ERROR] Sending loop crashed:", e)
    finally:
        board.stop_stream()
        board.release_session()
        print("[INFO] BrainFlow session ended.")

def run_client():
    def run_async_loop():
        asyncio.run(read_and_send(SERVER_URI))
    t = threading.Thread(target=run_async_loop, daemon=True)
    t.start()
    main_app.mainloop()

if __name__ == "__main__":
    # Precompute a static energy landscape using an 8x8 grid (64 points)
    grid = np.linspace(-2, 2, 8)
    X, Y = np.meshgrid(grid, grid)
    Z = np.exp(-(X**2 + Y**2))
    energy_landscape_data = {
        "energy_landscape": {
            "x": X.flatten().tolist(),
            "y": Y.flatten().tolist(),
            "z": Z.flatten().tolist()
        }
    }
    run_client()
