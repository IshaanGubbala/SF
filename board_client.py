import asyncio
import websockets
import json
import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any

# --------------------------------------------------------------------------------
# Pydantic Models for Sending EEG Data and Receiving Predictions
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
# Client Configuration
# --------------------------------------------------------------------------------

# WebSocket server URI
SERVER_URI = "ws://192.168.5.71:8000/ws"  # Adjust if the server is hosted elsewhere

# BrainFlow Configuration
BOARD_ID = BoardIds.GANGLION_NATIVE_BOARD.value  # Using Synthetic Board for testing
SERIAL_PORT = ''  # Not required for Synthetic Board

# Data Acquisition Parameters
SAMPLING_RATE = 256  # Hz
CHUNK_SIZE = 256  # Number of samples per chunk (1 second)
DELAY = 1  # Seconds to wait between sending chunks

# --------------------------------------------------------------------------------
# Client Implementation
# --------------------------------------------------------------------------------

async def send_eeg_data(uri: str):
    """
    Connects to the WebSocket server, acquires EEG data using BrainFlow,
    sends the data to the server, and listens for prediction responses.
    """
    # Initialize BrainFlow
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT  # Not required for Synthetic Board
    board = BoardShim(BOARD_ID, params)
    
    try:
        board.prepare_session()
        board.start_stream()
        print("[INFO] BrainFlow session started. Acquiring data...")
    except Exception as e:
        print(f"[ERROR] Failed to start BrainFlow session: {e}")
        return
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"[INFO] Connected to server at {uri}")
            
            # Listen for prediction responses in the background
            asyncio.create_task(receive_predictions(websocket))
            
            # Continuously acquire and send data
            while True:
                # Acquire CHUNK_SIZE samples
                data = board.get_current_board_data(CHUNK_SIZE)
                eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
                
                # Validate that EEG channels are available
                if not eeg_channels:
                    print("[ERROR] No EEG channels found. Check board configuration.")
                    await websocket.send(json.dumps({"error": "No EEG channels found."}))
                    break
                
                # Extract EEG data: shape (samples, channels)
                eeg_data = data[eeg_channels, :].T  # Shape: (samples, channels)
                
                # Scale each channel by 1e-9
                eeg_data_scaled = (eeg_data * 1e-9).tolist()  # Convert to list after scaling
                
                # Convert data to list of EEGSample
                eeg_samples = [EEGSample(channels=sample) for sample in eeg_data_scaled]
                
                # Create EEGData message
                eeg_message = EEGData(data=eeg_samples)
                
                # Send data as JSON
                await websocket.send(eeg_message.json())
                print(f"[INFO] Sent {CHUNK_SIZE} samples to server.")
                
                # Wait before sending the next chunk
                await asyncio.sleep(DELAY)
    
    except websockets.exceptions.ConnectionClosed as e:
        print(f"[INFO] Connection closed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    finally:
        board.stop_stream()
        board.release_session()
        print("[INFO] BrainFlow session stopped.")

async def receive_predictions(websocket: websockets.WebSocketClientProtocol):
    """
    Listens for prediction responses from the server and displays them.
    """
    try:
        async for message in websocket:
            try:
                response_dict = json.loads(message)
                
                if "error" in response_dict:
                    print(f"\n[SERVER ERROR]: {response_dict['error']}\n")
                    continue  # Skip processing for this message
                
                # Attempt to parse the response into PredictionResponse
                prediction_response = PredictionResponse(**response_dict)
                
                # Display the prediction results
                display_prediction(prediction_response)
            except json.JSONDecodeError:
                print(f"[WARNING] Received non-JSON message: {message}")
            except ValidationError as ve:
                print(f"[WARNING] Received message with invalid format: {ve}")
            except Exception as e:
                print(f"[ERROR] Failed to process received message: {e}")
    except websockets.exceptions.ConnectionClosed:
        print("[INFO] Server closed the connection.")
    except Exception as e:
        print(f"[ERROR] Error while receiving predictions: {e}")

def display_prediction(prediction_response: PredictionResponse):
    """
    Formats and displays the prediction response.
    """
    print("\n=== Prediction Received ===")
    print(f"Prediction: {'Alzheimer' if prediction_response.prediction == 1 else 'Control'}")
    print(f"Confidence: {prediction_response.confidence * 100:.2f}%")
    print("\nFeatures:")
    for feature, value in prediction_response.features.items():
        print(f"  {feature}: {value:.9f}")
    print("===========================\n")

# --------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(send_eeg_data(SERVER_URI))
    except KeyboardInterrupt:
        print("\n[INFO] Client stopped manually.")
