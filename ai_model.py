import os
import numpy as np
import pandas as pd
import mne
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
from scipy.signal import find_peaks
from scipy.stats import entropy
import warnings
from tqdm import tqdm
import nolds  # Ensure you have installed nolds: pip install nolds
import concurrent.futures
import tensorflow as tf

# Suppress warnings and set MNE logging level to ERROR
warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')

# Configure TensorFlow to use GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

# Paths to data and model files
DATA_PATH = "/Users/ishaangubbala/Documents/SF/ds004504/derivatives"  # Update if dataset is located elsewhere
PARTICIPANTS_FILE = "./ds004504/participants.tsv"
SCALER_PATH = "/Users/ishaangubbala/Documents/SF/ai_scaler.pkl"
MODEL_PATH = "/Users/ishaangubbala/Documents/SF/deep_ai_model.keras"

# Channels to use
TARGET_CHANNELS = ['Fp1', 'Fp2', 'C3', 'C4']

# Frequency bands
DELTA_BAND = (1, 4)
THETA_BAND = (4, 8)
ALPHA_BAND = (8, 12)
BETA_BAND = (12, 30)

# ERP parameters
ERP_EVENT_ID = {'Stimulus': 1}  # Update based on your event annotations
ERP_TMIN = -0.2  # Start of each epoch (200 ms before the event)
ERP_TMAX = 0.8   # End of each epoch (800 ms after the event)

def load_eeg_data(subject_dir):
    eeg_file = os.path.join(subject_dir, "eeg", f"{os.path.basename(subject_dir)}_task-eyesclosed_eeg.set")
    if os.path.exists(eeg_file):
        try:
            raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
            raw.filter(1., 50., fir_design='firwin')  # Band-pass filter from 1-50Hz

            # Pick only the desired channels using the recommended method
            available_channels = raw.ch_names
            channels_to_pick = [ch for ch in TARGET_CHANNELS if ch in available_channels]
            if not channels_to_pick:
                return None
            raw.pick_channels(channels_to_pick)
            return raw
        except:
            return None
    return None

def extract_psd_features(raw):
    try:
        data = raw.get_data()  # shape: (n_channels, n_samples)

        # Compute PSD using Welch's method
        psds = []
        for ch_data in data:
            psd, freqs = mne.time_frequency.psd_array_welch(
                ch_data, sfreq=raw.info['sfreq'], fmin=1, fmax=50, n_fft=256, n_overlap=128
            )
            psds.append(psd)

        psds = np.array(psds)  # shape: (n_channels, n_freqs)
        psds /= np.sum(psds, axis=-1, keepdims=True)  # Normalize

        # Calculate power in specific frequency bands
        delta_power = psds[:, (freqs >= DELTA_BAND[0]) & (freqs < DELTA_BAND[1])].mean(axis=-1)
        theta_power = psds[:, (freqs >= THETA_BAND[0]) & (freqs < THETA_BAND[1])].mean(axis=-1)
        alpha_power = psds[:, (freqs >= ALPHA_BAND[0]) & (freqs < ALPHA_BAND[1])].mean(axis=-1)
        beta_power = psds[:, (freqs >= BETA_BAND[0]) & (freqs < BETA_BAND[1])].mean(axis=-1)

        # Spatial factor (variance of the data)
        spatial_factor = np.var(data, axis=1)

        # Complexity factor (Shannon entropy)
        complexity_factor = entropy(psds, base=2, axis=1)

        # Combine PSD features
        psd_features = np.hstack([
            delta_power,
            theta_power,
            alpha_power,
            beta_power,
            spatial_factor,
            complexity_factor
        ])  # shape: (n_channels * 6, )

        return psd_features
    except:
        return np.zeros(len(TARGET_CHANNELS) * 6)

def extract_erp_features(raw):
    try:
        # Check if events are present
        events, event_id = mne.events_from_annotations(raw)
        if not events.size:
            return np.zeros(len(TARGET_CHANNELS) * 2)  # Placeholder zeros

        # Create epochs
        epochs = mne.Epochs(raw, events, event_id=ERP_EVENT_ID, tmin=ERP_TMIN, tmax=ERP_TMAX, baseline=(None, 0), preload=True)

        if len(epochs) == 0:
            return np.zeros(len(TARGET_CHANNELS) * 2)  # No valid epochs

        # Compute ERP (average over epochs)
        evoked = epochs.average()

        # Extract P300 and N200 latencies
        erp_features = []
        for ch in TARGET_CHANNELS:
            if ch not in evoked.ch_names:
                erp_features.extend([0, 0])  # Placeholder for missing channels
                continue
            ch_evoked = evoked.copy().pick_channels([ch]).get_data()[0]

            times = epochs.times

            # Find N200 (negative peak around 200 ms)
            try:
                n200_window = (times >= 0.15) & (times <= 0.25)
                n200_data = ch_evoked[n200_window]
                n200_times = times[n200_window]
                if len(n200_data) == 0:
                    n200_latency = 0
                else:
                    n200_idx = np.argmin(n200_data)
                    n200_latency = n200_times[n200_idx]
            except:
                n200_latency = 0

            # Find P300 (positive peak around 300 ms)
            try:
                p300_window = (times >= 0.25) & (times <= 0.35)
                p300_data = ch_evoked[p300_window]
                p300_times = times[p300_window]
                if len(p300_data) == 0:
                    p300_latency = 0
                else:
                    p300_idx = np.argmax(p300_data)
                    p300_latency = p300_times[p300_idx]
            except:
                p300_latency = 0

            erp_features.extend([n200_latency, p300_latency])

        erp_features = np.array(erp_features)  # shape: (n_channels * 2, )
        return erp_features
    except:
        return np.zeros(len(TARGET_CHANNELS) * 2)

def tsallis_entropy(signal, q=2):
    """Compute Tsallis entropy of a signal."""
    probabilities, _ = np.histogram(signal, bins=30, density=True)
    probabilities = probabilities + 1e-12  # Avoid log(0)
    return (1 - np.sum(probabilities ** q)) / (q - 1)

def extract_complexity_features(raw):
    try:
        data = raw.get_data()  # shape: (n_channels, n_samples)
        tsallis_entropies = []
        hfd_values = []
        lz_complexities = []

        for ch_data in data:
            # Tsallis Entropy
            te = tsallis_entropy(ch_data, q=2)
            tsallis_entropies.append(te)

            # Higuchi Fractal Dimension using nolds
            try:
                hfd = nolds.higuchi_fd(ch_data, k_max=5)
            except:
                hfd = 0
            hfd_values.append(hfd)

            # Lempel-Ziv Complexity
            try:
                # Binarize the signal
                median = np.median(ch_data)
                binary_signal = ''.join(['1' if x > median else '0' for x in ch_data])
                # Calculate LZ complexity
                lz = lempel_ziv_complexity(binary_signal)
            except:
                lz = 0
            lz_complexities.append(lz)

        complexity_features = np.hstack([
            tsallis_entropies,
            hfd_values,
            lz_complexities
        ])  # shape: (n_channels * 3, )
        return complexity_features
    except:
        return np.zeros(len(TARGET_CHANNELS) * 3)

def lempel_ziv_complexity(binary_sequence):
    """Compute the Lempel-Ziv complexity of a binary sequence."""
    i, k, l = 0, 1, 1
    c = 1
    n = len(binary_sequence)
    while True:
        try:
            if binary_sequence[i + k - 1] == binary_sequence[l + k - 1]:
                k += 1
                l += 1
            else:
                if k > 1:
                    c += 1
                i += 1
                if i == n:
                    break
                l = i + 1
                k = 1
            if l + k - 1 >= n:
                if binary_sequence[i] != binary_sequence[l]:
                    c += 1
                break
        except IndexError:
            break
    return c

def extract_energy_landscape_features(raw):
    try:
        data = raw.get_data()  # shape: (n_channels, n_samples)
        energy_landscape_features = []

        for ch_data in data:
            # Simplified energy landscape: compute moving average energy
            energy = ch_data ** 2
            window_size = int(raw.info['sfreq'] * 0.5)  # 500 ms window
            if window_size < 1:
                window_size = 1
            moving_energy = np.convolve(energy, np.ones(window_size)/window_size, mode='valid')

            # Find local minima
            minima, _ = find_peaks(-moving_energy)
            num_minima = len(minima)

            # Find basin sizes (distance between minima)
            basin_sizes = np.diff(minima)
            avg_basin_size = np.mean(basin_sizes) if len(basin_sizes) > 0 else 0

            energy_landscape_features.extend([num_minima, avg_basin_size])

        energy_landscape_features = np.array(energy_landscape_features)  # shape: (n_channels * 2, )
        return energy_landscape_features
    except:
        return np.zeros(len(TARGET_CHANNELS) * 2)

def extract_features(raw):
    try:
        psd_features = extract_psd_features(raw)
        erp_features = extract_erp_features(raw)
        complexity_features = extract_complexity_features(raw)
        energy_landscape_features = extract_energy_landscape_features(raw)

        # Combine all features into a single 1D array
        features = np.hstack([
            psd_features,
            erp_features,
            complexity_features,
            energy_landscape_features
        ])  # shape: (52, )
        return features
    except:
        return None

def load_labels():
    try:
        participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t')
        group_mapping = {'A': 0, 'C': 1, 'F': 2}  # A=Alzheimer's, C=Control, F=FTD
        labels = participants['Group'].map(group_mapping).values
        return participants['participant_id'].values, labels
    except:
        return np.array([]), np.array([])

def train_and_save_scaler(features):
    try:
        scaler = StandardScaler()
        scaler.fit(features)
        joblib.dump(scaler, SCALER_PATH)
        return scaler
    except:
        return None

def train_and_save_ai_model(features_scaled, labels):
    try:
        num_classes = len(np.unique(labels))
        model = Sequential([
            Dense(256, activation='relu', input_shape=(features_scaled.shape[1],)),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')  # Multi-class classification
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Split the data into training and validation sets manually
        validation_fraction = 0.2
        num_samples = features_scaled.shape[0]
        num_val = int(num_samples * validation_fraction)

        X_train, X_val = features_scaled[:-num_val], features_scaled[-num_val:]
        y_train, y_val = labels[:-num_val], labels[-num_val:]

        # Utilize TensorFlow's tf.data for efficient data loading
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

        # Use tqdm to display training progress
        epochs = 100
        for epoch in tqdm(range(epochs), desc="Training Model"):
            model.fit(train_dataset, epochs=1, verbose=0, validation_data=val_dataset)

        # Save model in native Keras format
        model.save(MODEL_PATH)
        return model
    except:
        return None

def process_subject(subject_id, label, data_path):
    subject_dir = os.path.join(data_path, subject_id)
    raw = load_eeg_data(subject_dir)
    if raw is not None:
        features = extract_features(raw)
        if features is not None:
            return features, label
    return None, None

def main():
    participant_ids, labels = load_labels()
    if len(participant_ids) == 0:
        return

    all_features = []
    all_labels = []

    # Utilize ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Prepare arguments for each subject
        futures = [executor.submit(process_subject, sid, lbl, DATA_PATH) for sid, lbl in zip(participant_ids, labels)]
        
        # Use tqdm to display progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Subjects"):
            features, label = future.result()
            if features is not None and label is not None:
                all_features.append(features)
                all_labels.append(label)

    # Combine all features and labels
    if not all_features:
        return

    X = np.vstack(all_features)
    y = np.array(all_labels)

    # Train scaler
    scaler = train_and_save_scaler(X)
    if scaler is not None:
        # Scale features
        X_scaled = scaler.transform(X)

        # Train and save AI model
        model = train_and_save_ai_model(X_scaled, y)
        if model is not None:
            # Optionally, you can verify the existence of the saved model
            if os.path.exists(MODEL_PATH):
                # Use tqdm to indicate completion
                tqdm.write("Model training completed and saved successfully.")
            else:
                tqdm.write("Model training failed. Model file not found.")
    else:
        tqdm.write("Scaler training failed.")

if __name__ == "__main__":
        main()
