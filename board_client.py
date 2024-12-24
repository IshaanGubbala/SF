"""
board_client.py

This script connects to an EEG board (via BrainFlow), collects real-time EEG data, 
and sends the data to a Flask server for prediction. It also performs simple 
real-time Matplotlib visualization of the EEG data.

Usage:
  python board_client.py
"""

import os
import time
import requests
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# ------------------------------
# Configuration
# ------------------------------
CHANNELS = ["Fp1", "Fp2", "C3", "C4"]  # The EEG channel labels you want to use
SAMPLING_RATE = 500                   # Your EEG board's sampling rate (Hz)
BUFFER_LENGTH = SAMPLING_RATE * 10    # Number of samples you want to store (e.g., 10 seconds)
ANALYSIS_INTERVAL = 2 * 60            # Perform long-term analysis every 2 minutes
DATA_INTERVAL = 0.005                 # Sleep time between data grabs (5 ms)
SERVER_URL = "http://192.168.5.71:5000"  # Flask server URL
USE_SYNTHETIC_BOARD = True            # If True, use BrainFlow's synthetic board

# ------------------------------
# Visualization Setup
# ------------------------------
plt.ion()
fig, axes = plt.subplots(len(CHANNELS), 1, sharex=True, figsize=(8, 6))

for i, ch in enumerate(CHANNELS):
    axes[i].set_xlim(0, BUFFER_LENGTH)
    axes[i].set_ylim(-1.5, 1.5)
    axes[i].set_title(ch)
    axes[i].set_xlabel("Samples")
    axes[i].set_ylabel("Amplitude")

# Each channel gets a line object
lines = [axes[i].plot([], [])[0] for i in range(len(CHANNELS))]

# ------------------------------
# Data Buffer
# ------------------------------
buffer = deque(maxlen=BUFFER_LENGTH)

def main():
    """
    Main loop for reading data from the EEG board, buffering it, 
    visualizing it, and sending it to the Flask server for prediction.
    """
    # ------------------------------
    # BrainFlow Board Setup
    # ------------------------------
    params = BrainFlowInputParams()
    # Change this to the actual serial_port or other parameter for your board
    # Example: params.serial_port = "/dev/ttyUSB0" or "COM3"
    # If you are using a wifi/shield board or other settings, 
    # configure BrainFlowInputParams accordingly.

    if USE_SYNTHETIC_BOARD:
        board_id = BoardIds.SYNTHETIC_BOARD.value
    else:
        # Replace with the correct board ID for your hardware
        board_id = BoardIds.CYTON_BOARD.value

    board = BoardShim(board_id, params)

    # Prepare and start data stream
    board.prepare_session()
    board.start_stream()
    print("Starting real-time EEG streaming...")

    last_analysis_time = time.time()

    try:
        while True:
            # Grab up to SAMPLING_RATE samples from the board
            data = board.get_current_board_data(SAMPLING_RATE)

            # If no new data, continue
            if data.shape[1] == 0:
                continue

            # For demonstration, scale raw data by dividing by 100 
            # (depends on your board's output amplitude)
            data[:len(CHANNELS), :] /= 100

            # Select only the relevant EEG channels (in row-major, then transpose)
            selected_data = data[:len(CHANNELS), :].T  # shape = (samples, channels)
            buffer.extend(selected_data)

            # ------------------------------
            # Local Visualization (Optional)
            # ------------------------------
            for i, line in enumerate(lines):
                channel_data = [row[i] for row in buffer]
                line.set_data(range(len(buffer)), channel_data)
                axes[i].relim()
                axes[i].autoscale_view()
            plt.pause(0.001)

            # ------------------------------
            # Short-Term Analysis
            # ------------------------------
            # Once buffer is full (10 seconds), send that chunk to the server
            if len(buffer) >= BUFFER_LENGTH:
                short_term_data = np.array(buffer).T  # shape: (channels, samples)

                # Send data to server via POST request
                try:
                    response = requests.post(
                        f"{SERVER_URL}/predict",
                        json={"data": short_term_data.tolist()},
                        timeout=5
                    )
                    if response.status_code == 200:
                        print("Short-term prediction received from server.")
                        # Optionally parse JSON: response.json() 
                    else:
                        print(f"Server error: {response.status_code}, {response.text}")
                except requests.exceptions.RequestException as e:
                    print(f"Error posting data to server: {e}")

            # ------------------------------
            # Long-Term Analysis
            # ------------------------------
            # Every ANALYSIS_INTERVAL seconds, send the entire buffer
            if time.time() - last_analysis_time >= ANALYSIS_INTERVAL:
                long_term_data = np.array(buffer).T  # shape: (channels, samples)

                try:
                    response = requests.post(
                        f"{SERVER_URL}/predict",
                        json={"data": long_term_data.tolist()},
                        timeout=5
                    )
                    if response.status_code == 200:
                        print("Long-term prediction received from server.")
                        # Optionally parse JSON: response.json()
                    else:
                        print(f"Server error: {response.status_code}, {response.text}")
                except requests.exceptions.RequestException as e:
                    print(f"Error posting data to server: {e}")

                last_analysis_time = time.time()

            time.sleep(DATA_INTERVAL)

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        board.stop_stream()
        board.release_session()
        print("EEG session closed.")

if __name__ == "__main__":
    main()
