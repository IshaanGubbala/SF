import asyncio
import websockets
import json
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any

# --------------------------------------------------------------------------------
# Pydantic Models (Same as Server)
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

# For receiving the energy landscape as well
class EnergyLandscape(BaseModel):
    x: List[List[float]]
    y: List[List[float]]
    z: List[List[float]]

# --------------------------------------------------------------------------------
# Client Configuration
# --------------------------------------------------------------------------------

SERVER_URI = "ws://192.168.5.222:8000/ws"  # or wherever the server is
BOARD_ID = BoardIds.SYNTHETIC_BOARD.value
SERIAL_PORT = ""
SAMPLING_RATE = 256
CHUNK_SIZE = 256
DELAY = 1

# --------------------------------------------------------------------------------
# Main Client
# --------------------------------------------------------------------------------

async def receive_responses(websocket: websockets.WebSocketClientProtocol):
    """
    Continuously listens for messages from the server.
    We expect either:
      - EnergyLandscape (x,y,z) on first connect
      - PredictionResponse for each window
      - Potential error messages
    """
    try:
        async for msg in websocket:
            try:
                data = json.loads(msg)

                # Check if it's the energy landscape
                if all(k in data for k in ["x", "y", "z"]):
                    # Parse as EnergyLandscape
                    landscape = EnergyLandscape(**data)
                    print("[INFO] Received Energy Landscape from server.")
                    print(f"Surface Grid Sizes: x={len(landscape.x)}, y={len(landscape.y)}")
                    # Here you might forward to a plotting function or store it
                    continue

                # Check if it's a prediction response
                if "prediction" in data and "confidence" in data:
                    # Parse as PredictionResponse
                    resp = PredictionResponse(**data)
                    display_prediction(resp)
                    continue

                # Otherwise, might be an error
                if "error" in data:
                    print("[SERVER ERROR]:", data["error"])
                    continue

                print("[WARNING] Unknown message structure:", data)

            except ValidationError as ve:
                print("[ERROR] ValidationError:", ve)
            except Exception as e:
                print("[ERROR] Failed parsing message:", e)
    except websockets.exceptions.ConnectionClosed:
        print("[INFO] Connection to server closed.")
    except Exception as e:
        print("[ERROR] Unexpected error in receive_responses:", e)

def display_prediction(resp: PredictionResponse):
    pred_label = "Alzheimer" if resp.prediction == 1 else "Control"
    conf_str = f"{resp.confidence*100:.2f}%"
    print("\n=== Prediction ===")
    print(f"  Class: {pred_label}")
    print(f"  Confidence: {conf_str}")
    print(f"  PC1={resp.pc1:.4f}, PC2={resp.pc2:.4f}, Energy={resp.energy:.4f}")
    print("=================\n")

async def send_eeg_data(uri: str):
    """
    Connects to server, starts BrainFlow session, sends chunks of EEG data,
    and concurrently receives server responses (energy landscape + predictions).
    """
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BOARD_ID, params)

    try:
        board.prepare_session()
        board.start_stream()
        print("[INFO] BrainFlow session started.")
    except Exception as e:
        print("[ERROR] BrainFlow session init failed:", e)
        return

    try:
        async with websockets.connect(uri) as ws:
            print("[INFO] Connected to server:", uri)
            # Start background task to handle responses
            asyncio.create_task(receive_responses(ws))

            while True:
                data_chunk = board.get_current_board_data(CHUNK_SIZE)
                eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
                if not eeg_channels:
                    print("[ERROR] No EEG channels found.")
                    break
                
                chunk_eeg = data_chunk[eeg_channels, :].T  # shape=(CHUNK_SIZE, n_channels)
                chunk_eeg_scaled = (chunk_eeg * 1e-9).tolist()

                samples = [EEGSample(channels=row) for row in chunk_eeg_scaled]
                message = EEGData(data=samples)

                await ws.send(message.json())
                print(f"[INFO] Sent {CHUNK_SIZE} samples to server.")
                await asyncio.sleep(DELAY)

    except websockets.exceptions.ConnectionClosed as e:
        print("[INFO] Connection closed:", e)
    except Exception as e:
        print("[ERROR] Unexpected error in sending loop:", e)
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