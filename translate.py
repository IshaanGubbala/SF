import time
import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Initialize BrainFlow parameters
params = BrainFlowInputParams()
board_id = BoardIds.GANGLION_NATIVE_BOARD.value
board = BoardShim(board_id, params)

# Start the session
board.prepare_session()
board.start_stream()
print("Collecting data... Press Ctrl+C to stop.")
#print(BoardShim.get_board_descr(board_id))

try:
    while True:
        # Collect data from the board
        
        data = board.get_board_data()
        if data.shape[1] > 0:
            # Save the data to a CSV file
            df = pd.DataFrame(data.T)
            output_file = 'synthetic_board_data.csv'
            df.to_csv(output_file, index=False)
            print(f"Saved data to {output_file}")
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopping data collection.")
    board.stop_stream()
    board.release_session()
    print("Session ended.")
