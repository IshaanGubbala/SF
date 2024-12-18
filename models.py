import os
import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
from multiprocessing import Pool
from tqdm import tqdm  # For progress bars

# Paths to data
data_path = "./ds004504/derivatives"
participants_file = "./ds004504/participants.tsv"

# Channels to use
TARGET_CHANNELS = ['Fp1', 'Fp2', 'C3', 'C4']

# 1. Load EEG Data with Specific Channels
def load_eeg_data(subject_dir):
    eeg_file = os.path.join(subject_dir, "eeg", f"{os.path.basename(subject_dir)}_task-eyesclosed_eeg.set")
    if os.path.exists(eeg_file):
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
        raw.filter(1., 50., fir_design='firwin')  # Band-pass filter from 1-50Hz

        # Pick only the desired channels
        channels_to_pick = [ch for ch in TARGET_CHANNELS if ch in raw.ch_names]
        if not channels_to_pick:
            return None
        raw.pick_channels(channels_to_pick)
        return raw
    return None

def extract_features(raw):
    print("Extracting features...")
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, overlap=1.0)
    psds, freqs = epochs.compute_psd(method='welch', fmin=1, fmax=50, n_fft=256).get_data(return_freqs=True)

    # Entropy (approximate)
    entropy = -np.sum(psds * np.log(psds + 1e-8), axis=2)

    # Power band calculations
    delta = psds[:, :, (freqs >= 1) & (freqs < 4)].mean(axis=2)
    theta = psds[:, :, (freqs >= 4) & (freqs < 8)].mean(axis=2)
    alpha = psds[:, :, (freqs >= 8) & (freqs < 12)].mean(axis=2)
    beta = psds[:, :, (freqs >= 12) & (freqs < 30)].mean(axis=2)
    gamma = psds[:, :, (freqs >= 30) & (freqs <= 50)].mean(axis=2)

    # Combine features
    features = np.hstack([delta, theta, alpha, beta, gamma, entropy])
    return features

# 3. Load Labels
def load_labels(participants_file):
    participants = pd.read_csv(participants_file, sep='\t')
    group_mapping = {'A': 0, 'C': 1, 'F': 2}
    return participants['participant_id'].values, participants['Group'].map(group_mapping).values

# 4. Process a Single Subject
def process_subject(args):
    subject_id, label, data_path = args
    subject_dir = os.path.join(data_path, subject_id)
    raw_data = load_eeg_data(subject_dir)
    if raw_data is not None:
        features = extract_features(raw_data)
        labels = [label] * features.shape[0]
        return features, labels
    return None, None

# 5. Train and Save Models
def train_and_save_models(X_train, y_train):
    print("\nTraining models...")
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42)
    }

    trained_models = {}
    for name, model in tqdm(models.items(), desc="Model Training"):
        model.fit(X_train, y_train)
        joblib.dump(model, f"{name.lower()}_model.pkl")
        print(f"{name} model saved as {name.lower()}_model.pkl")
        trained_models[name] = model
    return trained_models

# Main Function
def main():
    print("Loading labels from participants file...")
    participant_ids, labels = load_labels(participants_file)

    # Parallel processing for EEG data
    print("\nProcessing all subjects in parallel...")
    results = []
    with tqdm(total=len(participant_ids), desc="Processing EEG Files") as pbar:
        with Pool() as pool:
            for result in pool.imap_unordered(
                process_subject, [(pid, lbl, data_path) for pid, lbl in zip(participant_ids, labels)]
            ):
                results.append(result)
                pbar.update(1)

    # Combine features and labels
    all_features, all_labels = [], []
    for features, labels in results:
        if features is not None and labels is not None:
            all_features.append(features)
            all_labels.extend(labels)

    if not all_features:
        print("No valid data found. Exiting.")
        return

    X = np.vstack(all_features)
    y = np.array(all_labels)
    print(f"\nTotal features shape: {X.shape}, Total labels shape: {y.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Handle class imbalance with SMOTE
    print("\nApplying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Standardize data
    print("Standardizing the data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved as scaler.pkl")

    # Train and evaluate models
    trained_models = train_and_save_models(X_train, y_train)

    print("\nEvaluating models...")
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        print(f"\n{name} Results:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    main()
