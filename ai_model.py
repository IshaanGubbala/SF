import os
import numpy as np
import mne
import pandas as pd
import glob
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    GRU, Dense, Dropout, Input, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add, Activation,
    Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from scipy.stats import ttest_ind

# Suppress warnings for cleaner output
mne.set_log_level('WARNING')
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------

# Paths (Update these paths accordingly)
DS004504_PATH = "/Users/ishaangubbala/Documents/SF/ds004504/derivatives"  # Path to ds004504 EEG .set files
DS003800_PATH = "/Users/ishaangubbala/Documents/SF/ds003800/"             # Path to ds003800 EEG .set files
PARTICIPANTS_FILE_DS004504 = "/Users/ishaangubbala/Documents/SF/ds004504/participants.tsv"   # Path to ds004504 participants.tsv
PARTICIPANTS_FILE_DS003800 = "/Users/ishaangubbala/Documents/SF/ds003800/participants.tsv"   # Path to ds003800 participants.tsv

# Sampling rate (Hz)
SAMPLING_RATE = 256

# Frequency bands (Hz)
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

# Output directories
FEATURE_ANALYSIS_DIR = "feature_analysis"
MODELS_DIR = "trained_models"
PLOTS_DIR = "plots"

# Create output directories if they don't exist
os.makedirs(FEATURE_ANALYSIS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --------------------------------------------------------------------------------
# 1) LOAD PARTICIPANT LABELS
# --------------------------------------------------------------------------------

def load_participant_labels(ds004504_participants_file, ds003800_participants_file):
    """
    Reads participants.tsv files from both datasets.
    Assigns labels:
        - For ds004504: 'A' -> 1 (Alzheimer), 'C' -> 0 (Control).
        - For ds003800: All participants are labeled as 1 (Alzheimer).

    Args:
        ds004504_participants_file: Path to ds004504 participants.tsv.
        ds003800_participants_file: Path to ds003800 participants.tsv.

    Returns:
        label_dict: Dictionary mapping participant_id to label.
    """
    label_dict = {}
    group_map_ds004504 = {"A": 1, "C": 0}

    # Process ds004504 participants.tsv
    df_ds004504 = pd.read_csv(ds004504_participants_file, sep="\t")
    if 'Group' not in df_ds004504.columns or 'participant_id' not in df_ds004504.columns:
        raise ValueError("ds004504 participants.tsv must contain 'Group' and 'participant_id' columns.")
    df_ds004504 = df_ds004504[df_ds004504['Group'] != 'F']  # Remove FTD if present
    df_ds004504 = df_ds004504[df_ds004504['Group'].isin(group_map_ds004504.keys())]  # Ensure only A and C groups are present
    labels_ds004504 = df_ds004504.set_index("participant_id")["Group"].map(group_map_ds004504).to_dict()
    label_dict.update(labels_ds004504)

    # Process ds003800 participants.tsv
    df_ds003800 = pd.read_csv(ds003800_participants_file, sep="\t")
    if 'Group' not in df_ds003800.columns or 'participant_id' not in df_ds003800.columns:
        raise ValueError("ds003800 participants.tsv must contain 'Group' and 'participant_id' columns.")
    # Assign label 1 to all participants in ds003800
    labels_ds003800 = df_ds003800.set_index("participant_id")["Group"].apply(lambda x: 1).to_dict()
    label_dict.update(labels_ds003800)

    return label_dict

# Load participant labels from both datasets
participant_labels = load_participant_labels(
    ds004504_participants_file=PARTICIPANTS_FILE_DS004504,
    ds003800_participants_file=PARTICIPANTS_FILE_DS003800
)

# --------------------------------------------------------------------------------
# 2) FEATURE EXTRACTION FUNCTIONS
# --------------------------------------------------------------------------------

def compute_band_powers(data, sfreq):
    """
    Compute average power in each frequency band.
    Returns a dictionary with band names as keys and mean power as values.
    """
    psd, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=sfreq, verbose=False)
    band_powers = {}
    for band, (fmin, fmax) in FREQUENCY_BANDS.items():
        if band in ["Theta1", "Theta2", "Alpha1", "Alpha2", "Gamma1", "Gamma2"]:
            # Handle sub-bands by combining or keeping separate as needed
            band_powers[band] = np.mean(psd[:, :, (freqs >= fmin) & (freqs <= fmax)], axis=2)
        else:
            band_powers[band] = np.mean(psd[:, :, (freqs >= fmin) & (freqs <= fmax)], axis=2)
    # Aggregate mean power across epochs and channels
    aggregated_powers = {band: np.mean(band_powers[band]) for band in band_powers}
    return aggregated_powers

def compute_shannon_entropy(data):
    """
    Compute Shannon entropy for each epoch.
    Returns the mean entropy across all epochs.
    """
    n_epochs = data.shape[0]
    entropies = np.zeros(n_epochs, dtype=np.float32)
    for i in range(n_epochs):
        flattened = data[i].flatten()
        counts, _ = np.histogram(flattened, bins=256)
        probs = counts / np.sum(counts)
        entropies[i] = -np.sum(probs * np.log2(probs + 1e-12))  # Add epsilon to avoid log(0)
    return np.mean(entropies)

def compute_hjorth_parameters(data):
    """
    Compute Hjorth Activity, Mobility, and Complexity for each epoch.
    Returns the mean of each parameter across all epochs.
    """
    n_epochs, n_channels, n_times = data.shape
    activities = np.empty(n_epochs, dtype=np.float32)
    mobilities = np.empty(n_epochs, dtype=np.float32)
    complexities = np.empty(n_epochs, dtype=np.float32)

    for i in range(n_epochs):
        epoch = data[i]
        var0 = np.var(epoch, axis=1) + 1e-12  # Activity
        mob = np.sqrt(np.var(np.diff(epoch, axis=1), axis=1) / var0)  # Mobility
        comp = np.sqrt(np.var(np.diff(np.diff(epoch, axis=1), axis=1), axis=1) / (mob + 1e-12))  # Complexity
        activities[i] = var0.mean()
        mobilities[i] = mob.mean()
        complexities[i] = comp.mean()

    mean_act = activities.mean()
    mean_mob = mobilities.mean()
    mean_comp = complexities.mean()

    return mean_act, mean_mob, mean_comp

# --------------------------------------------------------------------------------
# 3) PARALLEL FEATURE EXTRACTION
# --------------------------------------------------------------------------------

def extract_features_parallel(data):
    """
    Compute features in parallel, ensuring all features are correctly added to the feature vector.
    """
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit tasks
        future_band_powers = executor.submit(compute_band_powers, data, SAMPLING_RATE)
        future_shannon_entropy = executor.submit(compute_shannon_entropy, data)
        future_hjorth_parameters = executor.submit(compute_hjorth_parameters, data)

        # Retrieve results
        band_powers = future_band_powers.result()
        shannon_entropy = future_shannon_entropy.result()
        hjorth_activity, hjorth_mobility, hjorth_complexity = future_hjorth_parameters.result()

    # Initialize feature vector
    feature_vector = []

    # Add band powers (ensure consistent ordering)
    for band in FREQUENCY_BANDS.keys():
        feature_vector.append(band_powers.get(band, 0.0))

    # Add ratios
    alpha_power = band_powers.get("Alpha1", 0.0) + band_powers.get("Alpha2", 0.0)
    theta_power = band_powers.get("Theta1", 0.0) + band_powers.get("Theta2", 0.0)
    total_power = sum(band_powers.values()) + 1e-12  # Avoid division by zero
    alpha_ratio = alpha_power / total_power
    theta_ratio = theta_power / total_power
    feature_vector.extend([alpha_ratio, theta_ratio])

    # Add Shannon Entropy
    feature_vector.append(shannon_entropy)

    # Add Hjorth Parameters (Spatial_Complexity is removed)
    feature_vector.extend([hjorth_activity, hjorth_mobility, hjorth_complexity])

    return np.array(feature_vector, dtype=np.float32)

def process_subject(args):
    """
    Process a single subject file and extract features and raw timeseries data.
    """
    file, label = args
    try:
        # Read EEG data using MNE
        raw = mne.io.read_raw_eeglab(file, preload=True, verbose=False)
        raw.filter(1.0, 50.0, fir_design="firwin", verbose=False)
        raw.resample(SAMPLING_RATE, npad="auto")
        epochs = mne.make_fixed_length_epochs(raw, duration=20.0, overlap=0.0, verbose=False)
        data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
        if data.size == 0:
            raise ValueError("No data extracted from epochs.")
        features = extract_features_parallel(data)

        # Extract raw timeseries data for Transformer
        # Here, we average across epochs to obtain a single sample per subject
        raw_timeseries = np.mean(data, axis=0)  # shape: (n_channels, n_times)
        # Expand dimensions to simulate a single sample with multiple channels and time steps
        # For Keras, input shape is (samples, time_steps, channels)
        raw_timeseries = raw_timeseries[np.newaxis, :, :]  # shape: (1, n_times, n_channels)

        return features, raw_timeseries, label
    except Exception as e:
        print(f"[ERROR] Failed to process {file}: {e}")
        return None, None, None

def load_dataset_parallel():
    """
    Load datasets in parallel using ProcessPoolExecutor from multiple dataset paths.

    Returns:
        X_features: Feature matrix (NumPy array).
        X_timeseries: Timeseries data (NumPy array).
        y: Labels (NumPy array).
    """
    dataset_paths = [DS004504_PATH, DS003800_PATH]
    all_files = []
    dataset_origin = {}  # Dictionary to track which dataset each file belongs to

    for dataset_path in dataset_paths:
        # Define glob pattern based on dataset
        if 'ds003800' in dataset_path:
            # Only include 'Rest' task files
            pattern = os.path.join(dataset_path, "**", "*_task-Rest_eeg.set")
        else:
            # Include all '.set' files from ds004504
            pattern = os.path.join(dataset_path, "**", "*.set")

        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
        for f in files:
            dataset_origin[f] = dataset_path  # Map file to its dataset

    tasks = []
    for f in all_files:
        # Extract participant ID; adjust based on your filename convention
        participant_id = os.path.basename(f).split('_')[0]  # Example: 'sub-001'
        label = participant_labels.get(participant_id, None)  # Get label; set to None if not found
        if label is not None:
            tasks.append((f, label))

    features_list = []
    timeseries_list = []
    labels_list = []

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_subject, tasks), total=len(tasks)))

    for feat, ts, lab in results:
        if feat is not None and ts is not None and lab is not None:
            features_list.append(feat)
            timeseries_list.append(ts)
            labels_list.append(lab)

    X_features = np.array(features_list, dtype=np.float32)
    X_timeseries = np.vstack(timeseries_list)  # shape: (n_samples, n_times, n_channels)
    y = np.array(labels_list, dtype=np.int32)
    print(f"[DEBUG] Combined Dataset loaded with {X_features.shape[0]} samples and {X_features.shape[1]} features.")
    return X_features, X_timeseries, y

# --------------------------------------------------------------------------------
# 4) TRANSFORMER MODEL DEFINITION
# --------------------------------------------------------------------------------
#@register_keras_serializable
class PositionalEncoding(Layer):
    def __init__(self, maxlen, d_model):
        """
        Positional Encoding Layer
        Args:
            maxlen: Maximum sequence length.
            d_model: Dimensionality of embeddings.
        """
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(maxlen, d_model)

    def get_angles(self, position, i, d_model):
        """Generate angles for the positional encoding formula."""
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, maxlen, d_model):
        """Create positional encoding for sequences."""
        position = np.arange(maxlen)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rads = self.get_angles(position, i, d_model)

        # Apply sin to even indices, cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        """Add positional encoding to inputs."""
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def positional_encoding(inputs, maxlen=5000, embed_dim=64):
    """
    Adds positional encoding to the input tensor.
    """
    pos = np.arange(maxlen)[:, np.newaxis]
    i = np.arange(embed_dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(embed_dim))
    angle_rads = pos * angle_rates
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def build_transformer_model(input_shape, d_model=32, num_heads=2, ff_dim=64, num_layers=2, dropout=0.45):
    """
    Builds a sophisticated Transformer-based model for binary classification.

    Args:
        input_shape: Tuple, e.g., (n_timesteps, n_channels).
        d_model: Dimensionality of the model/embedding.
        num_heads: Number of attention heads.
        ff_dim: Hidden dimension in the feed-forward network.
        num_layers: Number of Transformer encoder layers.
        dropout: Dropout rate.

    Returns:
        model: Compiled Keras Model outputting a single probability (sigmoid).
    """
    inputs = Input(shape=input_shape, name="transformer_input")
    
    # 1) Linear projection to d_model
    x = Dense(d_model)(inputs)  # shape: (batch, n_timesteps, d_model)
    
    # 2) Positional Encoding using the custom layer
    x = PositionalEncoding(maxlen=input_shape[0], d_model=d_model)(x)
    
    # 3) Transformer Encoder Blocks
    for i in range(num_layers):
        # Multi-Head Self-Attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn_output = Dropout(dropout)(attn_output)
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)
        
        # Feed-Forward Network
        ff_output = Dense(ff_dim, activation='relu')(x)
        ff_output = Dropout(dropout)(ff_output)
        ff_output = Dense(d_model)(ff_output)
        ff_output = Dropout(dropout)(ff_output)
        x = Add()([x, ff_output])
        x = LayerNormalization()(x)
    
    # 4) Global Average Pooling
    x = GlobalAveragePooling1D()(x)
    
    # 5) Output Layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=1e-5),
        metrics=['AUC']
    )
    
    return model

def print_transformer_training_stats(history, plots_dir):
    """
    Prints and visualizes the training statistics of the Transformer model.

    Args:
        history (tf.keras.callbacks.History): Training history from model.fit().
        plots_dir (str): Directory path to save the plots.
    """
    train_loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    train_auc = history.history.get('AUC', [])
    val_auc = history.history.get('val_AUC', [])

    print("\n--- Transformer Training Summary ---")
    if train_loss:
        print(f"Final Training Loss: {train_loss[-1]:.4f}")
    if val_loss:
        print(f"Final Validation Loss: {val_loss[-1]:.4f}")
    if train_auc and val_auc:
        print(f"Final Training AUC: {train_auc[-1]:.4f}")
        print(f"Final Validation AUC: {val_auc[-1]:.4f}")
    print("-" * 40)

    # Plot Loss
    if train_loss and val_loss:
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Transformer Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        loss_plot_path = os.path.join(plots_dir, "transformer_loss_plot.png")
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"[INFO] Transformer Loss plot saved to {loss_plot_path}")

    # Plot AUC
    if train_auc and val_auc:
        plt.figure(figsize=(10, 6))
        plt.plot(train_auc, label='Training AUC')
        plt.plot(val_auc, label='Validation AUC')
        plt.title('Transformer AUC Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        auc_plot_path = os.path.join(plots_dir, "transformer_auc_plot.png")
        plt.savefig(auc_plot_path)
        plt.close()
        print(f"[INFO] Transformer AUC plot saved to {auc_plot_path}")

# --------------------------------------------------------------------------------
# 5) BASELINE MODELS USING KERAS AND SKLEARN
# --------------------------------------------------------------------------------


def build_mlp_model(input_dim, hidden_layer_sizes=(20, 12), activation='relu', alpha=0.001, dropout=0.3):
    """
    Builds an MLP model using Keras.

    Args:
        input_dim: Number of input features.
        hidden_layer_sizes: Tuple representing the number of neurons in each hidden layer.
        activation: Activation function for hidden layers.
        alpha: L2 regularization parameter.
        dropout: Dropout rate.

    Returns:
        model: Compiled Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for units in hidden_layer_sizes:
        model.add(Dense(units, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(alpha)))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=1e-4),
        metrics=['AUC']
    )
    return model

def baseline_logistic_regression(X, y, feature_names, models_dir=MODELS_DIR, plots_dir=PLOTS_DIR):
    """
    Logistic Regression baseline with StratifiedKFold cross-validation and SMOTE.
    Saves trained models and ROC curves.
    Returns a list of trained models and their metrics.
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5)
    fold_num = 1
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []}
    models = []
    smote = SMOTE(random_state=5)

    for train_idx, test_idx in skf.split(X, y):
        print(f"\n[LogReg] Fold {fold_num} - Training")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply SMOTE to the training data
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        print(f"[LogReg] Fold {fold_num} - After SMOTE: {np.bincount(y_train_resampled)}")

        # Train Logistic Regression
        clf = LogisticRegression(max_iter=5000, random_state=5)
        clf.fit(X_train_resampled, y_train_resampled)
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]

        # Compute Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)

        # Print Classification Report
        print(f"\n[LogReg] Fold {fold_num} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Control', 'Alzheimer']))

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'LogReg Fold {fold_num} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Logistic Regression ROC Curve - Fold {fold_num}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"logreg_fold_{fold_num}_roc_curve.png"))
        plt.close()

        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Control', 'Alzheimer'])
        disp.plot(cmap='Blues')
        plt.title(f"Logistic Regression Confusion Matrix - Fold {fold_num}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"logreg_fold_{fold_num}_confusion_matrix.png"))
        plt.close()

        # Save Model and Scaler
        model_filename = os.path.join(models_dir, f"logreg_fold_{fold_num}.joblib")
        scaler_filename = os.path.join(models_dir, f"logreg_scaler_fold_{fold_num}.joblib")
        joblib.dump(clf, model_filename)
        joblib.dump(scaler, scaler_filename)
        print(f"[LogReg] Fold {fold_num} model and scaler saved.")

        # Store model and its ROC AUC for later selection
        models.append({
            'model': clf,
            'scaler': scaler,
            'roc_auc': roc_auc,
            'fold': fold_num
        })

        fold_num += 1

    # Aggregate and Print Average Metrics
    print("\n[LogReg] Cross-Validation Metrics:")
    for metric in metrics:
        avg = np.mean(metrics[metric])
        std = np.std(metrics[metric])
        print(f"{metric.capitalize()}: {avg:.4f} ± {std:.4f}")

    return models, metrics

def baseline_mlp(X, y, feature_names, models_dir=MODELS_DIR, plots_dir=PLOTS_DIR):
    """
    MLP Classifier baseline with StratifiedKFold cross-validation and SMOTE.
    Saves trained models and ROC curves.
    Returns a list of trained models and their metrics.
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5)
    fold_num = 1
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []}
    models = []
    smote = SMOTE(random_state=5)
    
    for train_idx, test_idx in skf.split(X, y):
        print(f"\n[MLP] Fold {fold_num} - Training")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply SMOTE to the training data
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        print(f"[MLP] Fold {fold_num} - After SMOTE: {np.bincount(y_train_resampled)}")

        # Define MLP Classifier using SciKeras
        mlp_model = KerasClassifier(
            model=build_mlp_model,
            model__input_dim=X_train_resampled.shape[1],
            model__hidden_layer_sizes=(20, 12),
            model__activation='relu',
            model__alpha=0.005,
            model__dropout=0.3,
            compile__loss='binary_crossentropy',
            compile__optimizer=Adam(learning_rate=1e-4),
            compile__metrics=['AUC'],
            epochs=2000,
            batch_size=32,
            validation_split=0.15,
            callbacks=[EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)],
            verbose=0,
            random_state=5
        )
        
        # Train MLP
        mlp_model.fit(X_train_resampled, y_train_resampled)
        
        # Predict probabilities and select the positive class probability (Alzheimer)
        y_proba = mlp_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        # Compute Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)

        # Print Classification Report
        print(f"\n[MLP] Fold {fold_num} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Control', 'Alzheimer']))

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'MLP Fold {fold_num} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'MLP ROC Curve - Fold {fold_num}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"mlp_fold_{fold_num}_roc_curve.png"))
        plt.close()

        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Control', 'Alzheimer'])
        disp.plot(cmap='Greens')
        plt.title(f"MLP Confusion Matrix - Fold {fold_num}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"mlp_fold_{fold_num}_confusion_matrix.png"))
        plt.close()

        # Save MLP Model and Scaler
        model_filename = os.path.join(models_dir, f"mlp_fold_{fold_num}.keras")
        scaler_filename = os.path.join(models_dir, f"mlp_scaler_fold_{fold_num}.joblib")
        mlp_model.model_.save(model_filename)  # Access the underlying Keras model
        joblib.dump(scaler, scaler_filename)
        print(f"[MLP] Fold {fold_num} augmented model and scaler saved.")

        # Store model and its ROC AUC for later selection
        models.append({
            'model': mlp_model,
            'scaler': scaler,
            'roc_auc': roc_auc,
            'fold': fold_num
        })

        fold_num += 1

    # Aggregate and Print Average Metrics
    print("\n[MLP] Cross-Validation Metrics:")
    for metric in metrics:
        avg = np.mean(metrics[metric])
        std = np.std(metrics[metric])
        print(f"{metric.capitalize()}: {avg:.4f} ± {std:.4f}")

    return models, metrics

# --------------------------------------------------------------------------------
# 6) UTILITY FUNCTIONS FOR MODEL EVALUATION
# --------------------------------------------------------------------------------

def plot_logreg_coefficients(model, feature_names, plots_dir=PLOTS_DIR):
    """
    Plots the coefficients of a Logistic Regression model.

    Args:
        model: Trained LogisticRegression model.
        feature_names: List of feature names.
    """
    if model is None:
        print("[ERROR] The provided Logistic Regression model is None.")
        return

    try:
        coefficients = model.coef_[0]
    except AttributeError:
        print("[ERROR] The provided model does not have 'coef_' attribute.")
        return

    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm')
    plt.title("Logistic Regression Coefficients")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.axvline(0, color='grey', linestyle='--')
    plt.tight_layout()
    coef_plot_path = os.path.join(plots_dir, "logreg_coefficients.png")
    plt.savefig(coef_plot_path)
    plt.close()
    print(f"[INFO] Logistic Regression coefficients plot saved to {coef_plot_path}.")

def evaluate_model_on_full_data(model, scaler, X, y, method_name='Method', models_dir=MODELS_DIR, plots_dir=PLOTS_DIR):
    """
    Evaluates the given model on the entire dataset without retraining.

    Args:
        model: The trained machine learning model to evaluate.
        scaler: The scaler associated with the model.
        X: Feature matrix.
        y: Labels.
        method_name: Name of the method (e.g., 'LogReg', 'MLP') for logging.
        models_dir: Directory to save the evaluation artifacts.
        plots_dir: Directory to save evaluation plots.

    Returns:
        metrics: Dictionary containing evaluation metrics.
    """
    print(f"\n[{method_name}] Evaluating on the entire dataset.")

    # Feature Scaling
    X_scaled = scaler.transform(X)

    if method_name.startswith('MLP'):
        # For Keras MLP
        y_proba = model.predict(X_scaled).ravel()
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        # For scikit-learn Logistic Regression
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]

    # Compute Metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_proba)

    print(f"\n[{method_name}] Performance on Entire Dataset:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{method_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{method_name} ROC Curve - Entire Dataset')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_curve_path = os.path.join(plots_dir, f"{method_name.lower().replace(' ', '_')}_entire_dataset_roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.close()
    print(f"[{method_name}] ROC curve saved to {roc_curve_path}.")

    # Plot Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Control', 'Alzheimer'])
    disp.plot(cmap='Oranges')
    plt.title(f"{method_name} Confusion Matrix - Entire Dataset")
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, f"{method_name.lower().replace(' ', '_')}_entire_dataset_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[{method_name}] Confusion matrix saved to {cm_path}.")

    # Save Evaluation Metrics
    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'ROC_AUC': roc_auc
    }
    metrics_filename = os.path.join(models_dir, f"{method_name.lower().replace(' ', '_')}_evaluation_metrics.json")
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[{method_name}] Evaluation metrics saved to {metrics_filename}.")

    return metrics

# --------------------------------------------------------------------------------
# 7) MAIN PIPELINE WITH 3-FOLD CROSS-VALIDATION
# --------------------------------------------------------------------------------

def select_best_model(models, method_name='Method'):
    """
    Selects the best model based on ROC AUC from a list of models.

    Args:
        models: List of dictionaries containing 'model', 'scaler', 'roc_auc', and 'fold'.
        method_name: Name of the method (e.g., 'LogReg', 'MLP') for logging.

    Returns:
        best_model: The model object with the highest ROC AUC.
        best_scaler: The scaler object associated with the best model.
    """
    best = max(models, key=lambda x: x['roc_auc'])
    print(f"\n[{method_name}] Best Model: Fold {best['fold']} with ROC AUC = {best['roc_auc']:.4f}")
    return best['model'], best['scaler']

def analyze_feature_statistics(X, y, feature_names, output_dir='feature_analysis'):
    """
    Analyze and visualize the distribution of features by class and compute statistical summaries.

    Args:
        X: Feature matrix (NumPy array).
        y: Labels (NumPy array).
        feature_names: List of feature names.
        output_dir: Directory to save the analysis plots.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Separate classes
    class_0 = X[y == 0]  # Control group
    class_1 = X[y == 1]  # Alzheimer group

    # Store stats summary
    stats_summary = []

    # Analyze each feature
    for i, feature in enumerate(feature_names):
        print(f"[INFO] Analyzing feature: {feature}")

        # Compute statistics
        mean_0, std_0 = np.mean(class_0[:, i]), np.std(class_0[:, i])
        mean_1, std_1 = np.mean(class_1[:, i]), np.std(class_1[:, i])
        t_stat, p_val = ttest_ind(class_0[:, i], class_1[:, i], equal_var=False)

        # Save stats
        stats_summary.append({
            "Feature": feature,
            "Control Mean": mean_0,
            "Control Std": std_0,
            "Alzheimer Mean": mean_1,
            "Alzheimer Std": std_1,
            "T-Statistic": t_stat,
            "P-Value": p_val
        })

        # Plot distributions
        plt.figure(figsize=(8, 6))
        sns.histplot(class_0[:, i], color='blue', label='Control', kde=True, stat="density", alpha=0.6)
        sns.histplot(class_1[:, i], color='orange', label='Alzheimer', kde=True, stat="density", alpha=0.6)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save plot to output directory
        plt.savefig(os.path.join(output_dir, f"{feature}_distribution.png"))
        plt.close()

    # Create DataFrame for summary stats
    stats_df = pd.DataFrame(stats_summary)
    stats_df.to_csv(os.path.join(output_dir, "feature_statistics.csv"), index=False)

    print(f"[INFO] Feature analysis completed. Results saved in '{output_dir}'.")
    return stats_df

def permutation_importance_manual(model, X, y, metric, n_repeats=10):
    """
    Manually computes permutation importance for each feature.
    
    Args:
        model: Trained model with a predict method.
        X: Feature matrix (n_samples, n_features).
        y: True labels.
        metric: Scoring metric function (e.g., accuracy_score).
        n_repeats: Number of times to permute each feature.
    
    Returns:
        importance: Array of importance scores for each feature.
    """
    baseline_score = metric(y, model.predict(X) >= 0.5)
    importance = np.zeros(X.shape[1])
    
    for col in range(X.shape[1]):
        scores = []
        X_permuted = X.copy()
        for _ in range(n_repeats):
            np.random.shuffle(X_permuted[:, col])
            score = metric(y, model.predict(X_permuted) >= 0.5)
            scores.append(baseline_score - score)
        importance[col] = np.mean(scores)
    
    return importance

def main():
    # Set random seeds for reproducibility
    np.random.seed(5)
    tf.random.set_seed(5)

    # Directories for saving models and plots
    MODELS_DIR = 'trained_models'
    PLOTS_DIR = 'plots'
    FEATURE_ANALYSIS_DIR = 'feature_analysis'

    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(FEATURE_ANALYSIS_DIR, exist_ok=True)

    # 1) Load and preprocess data
    print("[DEBUG] Loading dataset...")
    X_features, X_timeseries, y = load_dataset_parallel()
    print(f"[DEBUG] Combined Dataset loaded with {X_features.shape[0]} samples and {X_features.shape[1]} features.")

    # 2) Define feature names for reference
    band_names = list(FREQUENCY_BANDS.keys())  # ['Delta', 'Theta1', 'Theta2', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']
    feature_names = band_names + [
        "Alpha_Ratio", "Theta_Ratio",
        "Shannon_Entropy",
        "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity"
    ]
    # 3) Analyze and visualize features
    stats_df = analyze_feature_statistics(X_features, y, feature_names, output_dir=FEATURE_ANALYSIS_DIR)

    # Optional: Display top 10 statistically significant features
    print("\nTop 10 Statistically Significant Features (by P-Value):")
    print(stats_df.sort_values(by="P-Value").head(10))

    # --------------------------------------------------------------------------------
    # 4) TRAINING TRANSFORMER AND USING ITS OUTPUTS AS FEATURES FOR BLR and MLP
    # --------------------------------------------------------------------------------

    print("\n--- Training Transformer and Integrating with BLR and MLP ---")

    # Initialize Transformer model
    input_shape = X_timeseries.shape[1:]  # (n_times, n_channels)
    transformer_model = build_transformer_model(
        input_shape=input_shape,
        d_model=32,
        num_heads=2,
        ff_dim=64,
        num_layers=1,
        dropout=0.1
    )
    print("[INFO] Transformer model summary:")
    transformer_model.summary()

    # Define Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Train Transformer
    print("[INFO] Starting Transformer training...")
    history = transformer_model.fit(
        X_timeseries, y,
        epochs=1000,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1  # Change to 0 for less verbose output
    )

    # Print Transformer training stats
    print_transformer_training_stats(history, plots_dir=PLOTS_DIR)

    # Save the trained Transformer model
    transformer_model_path = os.path.join(MODELS_DIR, "transformer_model.keras")
    transformer_model.save(transformer_model_path)
    print(f"[INFO] Transformer model saved to {transformer_model_path}")

    # --------------------------------------------------------------------------------
    # **Extracting Transformer Outputs and Augmenting Features**
    # --------------------------------------------------------------------------------

    print("[INFO] Extracting Transformer outputs (probabilities)...")
    transformer_probs = transformer_model.predict(X_timeseries).ravel()  # shape: (n_samples,)

    # 1. Remove 'Spatial_Complexity' from X_features
    # Assuming 'Spatial_Complexity' is at index 12, adjust if different
    spatial_complexity_index = feature_names.index("Spatial_Complexity") if "Spatial_Complexity" in feature_names else None
    if spatial_complexity_index is not None:
        X_features_no_spatial = np.delete(X_features, spatial_complexity_index, axis=1)
        feature_names_no_spatial = [feat for i, feat in enumerate(feature_names) if i != spatial_complexity_index]
    else:
        X_features_no_spatial = X_features.copy()
        feature_names_no_spatial = feature_names.copy()
        print("[WARNING] 'Spatial_Complexity' feature not found. Proceeding without removal.")

    # 2. Augment baseline features with the Transformer output
    X_augmented = np.hstack((X_features_no_spatial, transformer_probs.reshape(-1,1)))
    augmented_feature_names = feature_names_no_spatial + ["Transformer_Output"]
    print(f"[DEBUG] Augmented feature set shape: {X_augmented.shape}")

    # --------------------------------------------------------------------------------
    # 5) TRAINING Baseline Models (BLR and MLP) on Augmented Features
    # --------------------------------------------------------------------------------

    print("\n--- Training Baseline Models (BLR and MLP) with Augmented Features ---")

    # Train Logistic Regression
    print("\n[LogReg] Starting Logistic Regression training with cross-validation...")
    models_blr, metrics_blr = baseline_logistic_regression(
        X=X_augmented,
        y=y,
        feature_names=augmented_feature_names,
        models_dir=MODELS_DIR,
        plots_dir=PLOTS_DIR
    )

    # Train MLP
    print("\n[MLP] Starting MLP training with cross-validation...")
    models_mlp_aug, metrics_mlp_aug = baseline_mlp(
        X=X_augmented,
        y=y,
        feature_names=augmented_feature_names,
        models_dir=MODELS_DIR,
        plots_dir=PLOTS_DIR
    )

    # --------------------------------------------------------------------------------
    # 6) SELECT AND SAVE BEST MODELS
    # --------------------------------------------------------------------------------

    # Select Best BLR Model
    best_blr_model, best_blr_scaler = select_best_model(models_blr, method_name='BLR')

    # Select Best MLP Augmented Model
    best_mlp_aug_model, best_mlp_aug_scaler = select_best_model(models_mlp_aug, method_name='MLP Augmented')

    # --------------------------------------------------------------------------------
    # 7) FINAL EVALUATION AND PLOTTING
    # --------------------------------------------------------------------------------

    # Evaluate Best BLR Model on Entire Dataset
    evaluated_blr = evaluate_model_on_full_data(
        model=best_blr_model,
        scaler=best_blr_scaler,
        X=X_augmented,
        y=y,
        method_name='BLR',
        models_dir=MODELS_DIR,
        plots_dir=PLOTS_DIR
    )

    # Evaluate Best MLP Augmented Model on Entire Dataset
    evaluated_mlp_aug = evaluate_model_on_full_data(
        model=best_mlp_aug_model,
        scaler=best_mlp_aug_scaler,
        X=X_augmented,
        y=y,
        method_name='MLP Augmented',
        models_dir=MODELS_DIR,
        plots_dir=PLOTS_DIR
    )

    # Plot Logistic Regression Coefficients
    plot_logreg_coefficients(best_blr_model, augmented_feature_names, plots_dir=PLOTS_DIR)

    # Plot MLP Feature Importance using Permutation Importance
    print("[INFO] Calculating permutation importance for MLP Augmented...")
    # Scale the entire augmented dataset
    scaler_mlp_full = best_mlp_aug_scaler
    X_scaled_full = scaler_mlp_full.transform(X_augmented)
    permutation_importance_scores = permutation_importance_manual(
        model=best_mlp_aug_model, 
        X=X_scaled_full, 
        y=y, 
        metric=accuracy_score,
        n_repeats=10
    )

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': augmented_feature_names,
        'Importance': permutation_importance_scores
    }).sort_values(by='Importance', ascending=False)

    # Plot Feature Importance
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.xlabel("Permutation Importance (Accuracy Drop)")
    plt.ylabel("Feature")
    plt.title("MLP Augmented Feature Importance")
    plt.tight_layout()
    feature_importance_path = os.path.join(PLOTS_DIR, "mlp_augmented_feature_importance.png")
    plt.savefig(feature_importance_path)
    plt.close()
    print(f"[INFO] MLP Augmented feature importance plot saved to {feature_importance_path}.")

    print("\n[INFO] Best models have been evaluated on the entire dataset and their performances have been reported.")

# --------------------------------------------------------------------------------
# FINAL SCRIPT EXECUTION
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    main()