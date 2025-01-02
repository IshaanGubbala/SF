import asyncio
import websockets
import json
import time
import numpy as np
import os  # For clearing the console
from tabulate import tabulate

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any

# --------------------------------------------------------------------------------
# Pydantic Models (Same as on the Server)
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

class EnergyLandscape(BaseModel):
    x: List[List[float]]
    y: List[List[float]]
    z: List[List[float]]

# --------------------------------------------------------------------------------
# Client Configuration
# --------------------------------------------------------------------------------

SERVER_URI = "ws://192.168.5.71:8000/ws"  # Adjust to your server IP/port
BOARD_ID = BoardIds.GANGLION_NATIVE_BOARD.value  # Synthetic board for demonstration
SERIAL_PORT = ""

# We'll assume a 256 Hz sampling rate
SAMPLING_RATE = 200

CHUNK_SIZE = 200   # 1 second of data at 256 Hz
DELAY = 0.05     # ~3 updates per second

BANDPASS_CENTER_FREQ = 25.5
BANDPASS_BANDWIDTH = 49.0
FILTER_ORDER = 4
FILTER_TYPE = FilterTypes.BUTTERWORTH.value

ENV_NOISE_TYPE = NoiseTypes.FIFTY.value  # 50 Hz mains noise

# --------------------------------------------------------------------------------
# Async Functions
# --------------------------------------------------------------------------------

async def receive_responses(websocket: websockets.WebSocketClientProtocol):
    """Listen for messages from the server."""
    try:
        async for msg in websocket:
            try:
                data = json.loads(msg)

                if all(k in data for k in ("x", "y", "z")):
                    # EnergyLandscape
                    landscape = EnergyLandscape(**data)
                    clear_console()
                    #print("[INFO] Received Energy Landscape from server.")
                    #print(f"Surface Grid Sizes: x={len(landscape.x)}, y={len(landscape.y)}")
                    continue

                elif "prediction" in data and "confidence" in data:
                    resp = PredictionResponse(**data)
                    display_prediction(resp)
                    continue

                elif "error" in data:
                    clear_console()
                    print("[SERVER ERROR]:", data["error"])
                    continue

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
        print("[ERROR] Unexpected error in receive_responses:", e)

def display_prediction(resp: PredictionResponse):
    """
    Print the prediction response in a tabulated format, including features.
    """
    clear_console()
    pred_label = "Alzheimer" if resp.prediction == 1 else "Control"
    conf_str = f"{resp.confidence * 100:.2f}"

    # Basic table of classification
    table_data = [
        ["Class", pred_label],
        ["Confidence", conf_str]
    ]
    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="pretty"))

    # Additional table for features
    feat_table = [(k, f"{v:.6f}") for k, v in resp.features.items()]
    print("\nFeatures:")
    print(tabulate(feat_table, headers=["Feature", "Value"], tablefmt="pretty"))


def clear_console():
    """Clear the console on any OS."""
    os.system("cls" if os.name == "nt" else "clear")

async def send_eeg_data(uri: str):
    """
    Acquire EEG from BrainFlow, apply typical EEG filtering, remove 50 Hz noise,
    reference channel 1, send data to server.
    """
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
            asyncio.create_task(receive_responses(ws))

            while True:
                data_chunk = board.get_current_board_data(CHUNK_SIZE)
                eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
                if not eeg_channels:
                    print("[ERROR] No EEG channels found.")
                    break

                # For each EEG channel, do:
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

                # Reference using channel 1 (2nd in zero-based indexing)
                ref_ch = eeg_channels[1]
                for ch in eeg_channels:
                    if ch != ref_ch:
                        data_chunk[ch, :] -= data_chunk[ref_ch, :]

                # Reshape => (CHUNK_SIZE, num_eeg_channels)
                chunk_eeg = data_chunk[eeg_channels, :].T
                #print("Raw Data (µV):", data_chunk[ch, :])
                # Scale if needed (assume synthetic board might be in nanoVolts)
                chunk_eeg_scaled = (chunk_eeg * 0.000177827941).tolist()

                # Build the EEGData Pydantic model
                samples = [EEGSample(channels=row) for row in chunk_eeg_scaled]
                message = EEGData(data=samples)

                # Send to server
                await ws.send(message.json())
                #print(f"[INFO] Sent {CHUNK_SIZE} filtered samples to server.")

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
