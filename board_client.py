import asyncio
import websockets
import json
import time
import numpy as np
import os  # For clearing the console
from tabulate import tabulate
import matplotlib
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any

# --------------------------------------------------------------------------------
# Pydantic Models (Updated to match the new server)
# --------------------------------------------------------------------------------

class EEGSample(BaseModel):
    channels: List[float]  # one sample

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

# --------------------------------------------------------------------------------
# Example: Real-Time Plot Setup (Matplotlib)
# --------------------------------------------------------------------------------

# Positions for a rudimentary "head map" with four electrodes:
# (Assuming top view of the head: y-axis up, x-axis positive to the right)
EEG_POSITIONS = {
    "Fp1": (-0.5, 1.0),
    "Fp2": (0.5, 1.0),
    "C3": (-0.7, 0.0),
    "C4": (0.7, 0.0)
}

# Initialize Matplotlib in interactive mode
plt.ion()
fig, (ax_time, ax_head) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Real-Time EEG Visualization")

# Buffers to hold time-series data for each of the 4 channels
# We'll keep ~5 seconds of data for display (5 * 256 = 1280 samples)
MAX_PLOT_SAMPLES = 1280
channel_data = np.zeros((4, 0))  # shape: (4_channels, time_samples)

# A simple mapping from your BoardShim EEG channel indices to [Fp1, Fp2, C3, C4]
# **Adjust** if needed
CHANNEL_MAP = {
    0: "Fp1",
    1: "Fp2",
    2: "C3",
    3: "C4"
}

# We will store the "alpha_ratio" from the last prediction
last_alpha_ratio = 0.0

# Initialize variables to store pc1, pc2, and energy
last_pc1 = 0.0
last_pc2 = 0.0
last_energy = 0.0

# Create line objects for each channel in ax_time
lines = []
colors = ["blue", "red", "green", "purple"]
for i in range(4):
    line_obj, = ax_time.plot([], [], color=colors[i], label=f"{CHANNEL_MAP[i]}")
    lines.append(line_obj)
ax_time.set_xlim(0, MAX_PLOT_SAMPLES)
ax_time.set_ylim(-100e-6, 100e-6)  # Adjust y-limits based on your amplitude
ax_time.set_xlabel("Samples (rolling window)")
ax_time.set_ylabel("Amplitude (Volts)")
ax_time.legend(loc='upper right')
ax_time.grid(True)

# For the head map, we'll plot 4 scatter points for electrodes
head_scatter = ax_head.scatter(
    [p[0] for p in EEG_POSITIONS.values()],
    [p[1] for p in EEG_POSITIONS.values()],
    c="gray",
    s=200
)
ax_head.set_xlim(-1.2, 1.2)
ax_head.set_ylim(-1.2, 1.5)
ax_head.set_aspect("equal", "box")
ax_head.set_title("Head Map (Alpha Ratio Color)")
ax_head.axis("off")

# A helper list for updating the scatter points in real-time
head_coords = list(EEG_POSITIONS.values())

def update_plots():
    """
    Update the time-series plot (4 channels) and the head map (color).
    """
    global channel_data, last_alpha_ratio, last_pc1, last_pc2, last_energy

    # 1) Update the time-series plot
    # Each channel_data[i,:] is a 1D array of amplitude over time
    num_points = channel_data.shape[1]
    # We want x-values from 0..num_points
    x_vals = np.arange(num_points)
    
    for i in range(4):
        if i < channel_data.shape[0]:
            y_vals = channel_data[i, :]
            lines[i].set_xdata(x_vals)
            lines[i].set_ydata(y_vals)
    ax_time.set_xlim(max(0, num_points - MAX_PLOT_SAMPLES), max(MAX_PLOT_SAMPLES, num_points))

    # 2) Update the head map color based on alpha_ratio
    # We'll color each electrode by the same alpha_ratio (0 = dark, 1+ = bright)
    # You could choose a more advanced approach (e.g., per-channel power).
    color_val = min(max(last_alpha_ratio, 0.0), 1.0)  # clamp 0..1
    # Create RGBA color from alpha_ratio (e.g. a gradient from black->yellow)
    color_rgba = (color_val, color_val, 0.0, 1.0)
    colors = [color_rgba]*4
    head_scatter.set_color(colors)

    # 3) Optionally, display pc1, pc2, and energy as text
    ax_head.text(0.5, -1.0, f"PC1: {last_pc1:.2f}\nPC2: {last_pc2:.2f}\nEnergy: {last_energy:.2f}",
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax_head.transAxes,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

    # Redraw
    fig.canvas.draw()
    plt.pause(0.001)  # brief pause to update the figure

# --------------------------------------------------------------------------------
# Client Configuration
# --------------------------------------------------------------------------------

SERVER_URI = "ws://10.148.82.49:8000/ws"  # Adjust to your server IP/port
BOARD_ID = BoardIds.SYNTHETIC_BOARD.value  # Synthetic board for demonstration
SERIAL_PORT = ""

SAMPLING_RATE = 256
CHUNK_SIZE = 256   # 1 second of data at 256 Hz
DELAY = 0.0333     # ~30 updates per second

BANDPASS_CENTER_FREQ = 25.5
BANDPASS_BANDWIDTH = 49.0
FILTER_ORDER = 4
FILTER_TYPE = FilterTypes.BUTTERWORTH.value
ENV_NOISE_TYPE = NoiseTypes.FIFTY.value

# --------------------------------------------------------------------------------
# Async Functions
# --------------------------------------------------------------------------------

async def receive_responses(websocket: websockets.WebSocketClientProtocol):
    """Listen for messages (predictions) from the server."""
    global last_alpha_ratio, last_pc1, last_pc2, last_energy

    try:
        async for msg in websocket:
            try:
                data = json.loads(msg)

                # Check if it's a PredictionResponse
                if all(key in data for key in ["prediction", "confidence", "features", "stats", "pc1", "pc2", "energy"]):
                    resp = PredictionResponse(**data)
                    # Display in console
                    display_prediction(resp)

                    # Grab the alpha_ratio for our head map color
                    # (If it doesn't exist, default to 0)
                    last_alpha_ratio = resp.features.get("Alpha_Ratio", 0.0)

                    # Update pc1, pc2, and energy
                    last_pc1 = resp.pc1
                    last_pc2 = resp.pc2
                    last_energy = resp.energy

                    # Update plots
                    update_plots()

                elif "error" in data:
                    clear_console()
                    print("[SERVER ERROR]:", data["error"])
                else:
                    # Unknown message type
                    clear_console()
                    print("[WARNING] Unknown message structure:", data)

            except ValidationError as ve:
                clear_console()
                print("[ERROR] ValidationError:", ve)
            except Exception as e:
                clear_console()
                print("[ERROR] Failed parsing message:", e)

    except websockets.exceptions.ConnectionClosed:
        print("[INFO] Connection to server closed.")
    except Exception as e:
        print(f"[ERROR] Unexpected error in receive_responses:", e)

def display_prediction(resp: PredictionResponse):
    """
    Print the prediction response in a tabulated format, including features.
    """
    clear_console()
    pred_label = "Alzheimer" if resp.prediction == 1 else "Control"
    conf_str = f"{resp.confidence * 100:.2f}%"

    # Basic table of classification
    table_data = [
        ["Class", pred_label],
        ["Confidence", conf_str]
    ]
    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="pretty"))

    # Additional table for features
    feat_table = [(k, f"{v:.12f}") for k, v in resp.features.items()]
    print("\nFeatures:")
    print(tabulate(feat_table, headers=["Feature", "Value"], tablefmt="pretty"))

    # Display pc1, pc2, and energy
    pc_table = [
        ["PC1", f"{resp.pc1:.4f}"],
        ["PC2", f"{resp.pc2:.4f}"],
        ["Energy", f"{resp.energy:.4f}"]
    ]
    print("\nPCA and Energy:")
    print(tabulate(pc_table, headers=["Metric", "Value"], tablefmt="pretty"))

def clear_console():
    """Clear the console on any OS."""
    os.system("cls" if os.name == "nt" else "clear")

async def send_eeg_data(uri: str):
    """
    Acquire EEG from BrainFlow, apply typical EEG filtering, remove 50 Hz noise,
    reference channel 1, send data to server. Update the local real-time plot.
    """
    global channel_data
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BOARD_ID, params)

    try:
        board.prepare_session()
        board.start_stream()
        time.sleep(2)
        print("[INFO] BrainFlow session started.")
    except Exception as e:
        print("[ERROR] BrainFlow session init failed:", e)
        return

    try:
        async with websockets.connect(uri) as ws:
            print("[INFO] Connected to server:", uri)
            # Start the background receive loop
            asyncio.create_task(receive_responses(ws))

            while True:
                data_chunk = board.get_current_board_data(CHUNK_SIZE)
                eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
                if not eeg_channels:
                    print("[ERROR] No EEG channels found.")
                    break

                # Filtering
                for ch in eeg_channels:
                    # Detrend
                    DataFilter.detrend(data_chunk[ch], DetrendOperations.LINEAR.value)
                    # Bandpass 1–50 Hz
                    DataFilter.perform_bandpass(
                        data_chunk[ch],
                        SAMPLING_RATE,
                        BANDPASS_CENTER_FREQ,
                        BANDPASS_BANDWIDTH,
                        FILTER_ORDER,
                        FILTER_TYPE,
                        0
                    )
                    # Remove 50 Hz noise
                    DataFilter.remove_environmental_noise(
                        data_chunk[ch],
                        SAMPLING_RATE,
                        ENV_NOISE_TYPE
                    )

                # Reference using channel 1
                ref_ch = eeg_channels[1]  # second channel in zero-based indexing
                for ch in eeg_channels:
                    if ch != ref_ch:
                        data_chunk[ch, :] -= data_chunk[ref_ch, :]

                # Assume the first 4 channels map to Fp1, Fp2, C3, C4
                # Reshape => (CHUNK_SIZE, num_eeg_channels)
                chunk_eeg = data_chunk[eeg_channels, :].T  # shape: (256, n_eeg_channels)
                chunk_eeg_scaled = (chunk_eeg * 1e-3)  # convert to Volts (example)

                # Update our local plot buffer for the first 4 channels
                # Ensure we only keep the shape: (4, X)
                four_chan_data = chunk_eeg_scaled[:, 0:4].T  # shape: (4, 256)
                # Append to channel_data
                channel_data = np.concatenate((channel_data, four_chan_data), axis=1)
                # Truncate if too large
                if channel_data.shape[1] > MAX_PLOT_SAMPLES:
                    channel_data = channel_data[:, -MAX_PLOT_SAMPLES:]

                # Send to server
                # Build the EEGData model
                samples = [
                    EEGSample(channels=row.tolist()) for row in chunk_eeg_scaled
                ]
                message = EEGData(data=samples)
                await ws.send(message.json())
                # print(f"[INFO] Sent {CHUNK_SIZE} filtered samples to server.")

                # Update local plots
                # update_plots()  # Moved to after receiving prediction

                # Sleep ~0.0333 seconds => ~30 times per second
                await asyncio.sleep(DELAY)

    except websockets.exceptions.ConnectionClosed as e:
        print("[INFO] Connection closed:", e)
    except Exception as e:
        print(f"[ERROR] Unexpected error in sending loop: {e}")
    finally:
        board.stop_stream()
        board.release_session()
        print("[INFO] BrainFlow session ended.")

def run_client():
    asyncio.run(send_eeg_data(SERVER_URI))

if __name__ == "__main__":
    try:
        run_client()
    except KeyboardInterrupt:
        print("[INFO] Client stopped manually.")
