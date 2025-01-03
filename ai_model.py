import os
import numpy as np
import mne
import pandas as pd
import glob
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import json
# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Parallel processing imports
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Statistical testing
from scipy.stats import ttest_ind
from imblearn.over_sampling import SMOTE
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings if any

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------

# Paths (Update these paths accordingly)
DS004504_PATH = "/Users/ishaangubbala/Documents/SF/ds004504/derivatives"           # Path to ds004504 EEG .set files
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
    #print(f"[DEBUG] Loading participant labels from {ds004504_participants_file}.")
    df_ds004504 = pd.read_csv(ds004504_participants_file, sep="\t")
    if 'Group' not in df_ds004504.columns or 'participant_id' not in df_ds004504.columns:
        raise ValueError("ds004504 participants.tsv must contain 'Group' and 'participant_id' columns.")
    df_ds004504 = df_ds004504[df_ds004504['Group'] != 'F']  # Remove FTD if present
    df_ds004504 = df_ds004504[df_ds004504['Group'].isin(group_map_ds004504.keys())]  # Ensure only A and C groups are present
    labels_ds004504 = df_ds004504.set_index("participant_id")["Group"].map(group_map_ds004504).to_dict()
    label_dict.update(labels_ds004504)
    #print(f"[DEBUG] ds004504: Loaded {len(labels_ds004504)} participant labels.")
    
    # Process ds003800 participants.tsv
    #print(f"[DEBUG] Loading participant labels from {ds003800_participants_file}.")
    df_ds003800 = pd.read_csv(ds003800_participants_file, sep="\t")
    if 'Group' not in df_ds003800.columns or 'participant_id' not in df_ds003800.columns:
        raise ValueError("ds003800 participants.tsv must contain 'Group' and 'participant_id' columns.")
    # Assign label 1 to all participants in ds003800
    labels_ds003800 = df_ds003800.set_index("participant_id")["Group"].apply(lambda x: 1).to_dict()
    label_dict.update(labels_ds003800)
    #print(f"[DEBUG] Total participants after combining datasets: {len(label_dict)}")
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
        idx_band = (freqs >= fmin) & (freqs <= fmax)
        band_powers[band] = np.mean(psd[:, :, idx_band], axis=2)  # mean over frequency band
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
    
    # Add Spatial Complexity
    # Assuming Spatial_Complexity is part of band_powers; if not, set to 0.0 or compute accordingly
    spatial_complexity = band_powers.get("Spatial_Complexity", 0.0)
    feature_vector.append(spatial_complexity)
    
    # Add Hjorth Parameters
    feature_vector.extend([hjorth_activity, hjorth_mobility, hjorth_complexity])
    
    return np.array(feature_vector, dtype=np.float32)

def process_subject(args):
    """
    Process a single subject file and extract features.
    """
    file, label = args
    try:
        #print(f"[DEBUG] Processing file: {file}")
        raw = mne.io.read_raw_eeglab(file, preload=True, verbose=False)
        raw.filter(1.0, 50.0, fir_design="firwin", verbose=False)
        raw.resample(SAMPLING_RATE, npad="auto")
        epochs = mne.make_fixed_length_epochs(raw, duration=20.0, overlap=0.0, verbose=False)
        data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
        if data.size == 0:
            raise ValueError("No data extracted from epochs.")
        features = extract_features_parallel(data)
        return features, label
    except Exception as e:
        print(f"[ERROR] Failed to process {file}: {e}")
        return None, None

def load_dataset_parallel():
    """
    Load datasets in parallel using ProcessPoolExecutor from multiple dataset paths.
    
    Returns:
        X: Feature matrix (NumPy array).
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
        
    #print(f"[DEBUG] Found {len(tasks)} EEG files to process from all datasets.")
    features_list = []
    labels_list = []
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_subject, tasks), total=len(tasks)))
    
    for feat, lab in results:
        if feat is not None and lab is not None:
            features_list.append(feat)
            labels_list.append(lab)
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    print(f"[DEBUG] Combined Dataset loaded with {X.shape[0]} samples and {X.shape[1]} features.")
    return X, y

# --------------------------------------------------------------------------------
# 4) FEATURE STATISTICS AND VISUALIZATION
# --------------------------------------------------------------------------------
def visualize_mlp_architecture(model, figsize=(12, 8)):
    """
    Visualizes the architecture of an MLPClassifier using networkx.

    Args:
        model: Trained MLPClassifier model.
        figsize: Size of the figure.
    """
    hidden_layers = model.hidden_layer_sizes
    layers = [model.n_features_in_] + list(hidden_layers) + [model.n_outputs_]
    num_layers = len(layers)

    G = nx.DiGraph()

    pos = {}
    node_size = 300
    layer_distance = 2
    neuron_distance = 1

    # Assign positions to nodes
    for layer_idx, layer_size in enumerate(layers):
        for neuron_idx in range(layer_size):
            node = f"L{layer_idx}N{neuron_idx}"
            G.add_node(node, layer=layer_idx)
            pos[node] = (layer_idx * layer_distance, neuron_idx * neuron_distance - (layer_size / 2))
    
    # Add edges between layers
    for layer_idx in range(num_layers - 1):
        for src_neuron in range(layers[layer_idx]):
            for dst_neuron in range(layers[layer_idx + 1]):
                src_node = f"L{layer_idx}N{src_neuron}"
                dst_node = f"L{layer_idx + 1}N{dst_neuron}"
                G.add_edge(src_node, dst_node, weight=0.05)

    plt.figure(figsize=figsize)
    layers_dict = nx.get_node_attributes(G, 'layer')
    colors = [layers_dict[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors, cmap=plt.cm.viridis)
    nx.draw_networkx_edges(G, pos, arrows=False, alpha=0.3)
    plt.title("MLP Architecture Visualization")
    for layer_idx, layer_size in enumerate(layers):
        if layer_idx == 0:
            layer_label = f"Input Layer\n({layer_size} Neurons)"
        elif layer_idx == num_layers - 1:
            layer_label = f"Output Layer\n({layer_size} Neurons)"
        else:
            layer_label = f"Hidden Layer {layer_idx}\n({layer_size} Neurons)"
        plt.text(
            layer_idx * layer_distance,
            (layer_size * neuron_distance) / 2 + 35,  # Position above the top neuron
            layer_label,
            horizontalalignment='center',
            fontsize=8,
            fontweight=None,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def analyze_feature_statistics(X, y, feature_names, output_dir=FEATURE_ANALYSIS_DIR):
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

# --------------------------------------------------------------------------------
# 5) BASELINE MODELS
# --------------------------------------------------------------------------------

def baseline_logistic_regression(X, y, feature_names, models_dir=MODELS_DIR, plots_dir=PLOTS_DIR):
    """
    Logistic Regression baseline with StratifiedKFold cross-validation and SMOTE.
    Saves trained models and ROC curves.
    Returns a list of trained models and their metrics.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
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

        # Define Logistic Regression Classifier
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
# Assuming 'retrained_mlp' is your trained MLP model
# and you have 'feature_names' and 'class_names' defined.

def plot_mlp_weights(model, output_dir=PLOTS_DIR):
    """
    Visualizes the distribution of weights for each hidden layer in the MLPClassifier.

    Args:
        model: Trained MLPClassifier model.
        output_dir: Directory to save the weight distribution plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for layer_idx, weight_matrix in enumerate(model.coefs_[:-1]):  # Exclude output layer
        weights = weight_matrix.flatten()
        
        plt.figure(figsize=(14, 6))

        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(weights, bins=50, kde=True, color='skyblue')
        plt.title(f"Layer {layer_idx +1} Weights Distribution")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")

        # Box Plot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=weights, color='lightgreen')
        plt.title(f"Layer {layer_idx +1} Weights Box Plot")
        plt.xlabel("Weight Value")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"mlp_layer_{layer_idx +1}_weights_distribution.png"))
        plt.close()



def baseline_mlp(X, y, feature_names, models_dir=MODELS_DIR, plots_dir=PLOTS_DIR):
    """
    MLP Classifier baseline with StratifiedKFold cross-validation and SMOTE.
    Saves trained models and ROC curves.
    Returns a list of trained models and their metrics.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
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

        # Define MLP Classifier
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 64, 12),
            activation='relu',
            solver='adam',
            max_iter=3000,
            random_state=5,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            alpha=0.1
        )
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

        # Save Model and Scaler
        model_filename = os.path.join(models_dir, f"mlp_fold_{fold_num}.joblib")
        scaler_filename = os.path.join(models_dir, f"mlp_scaler_fold_{fold_num}.joblib")
        joblib.dump(clf, model_filename)
        joblib.dump(scaler, scaler_filename)
        print(f"[MLP] Fold {fold_num} model and scaler saved.")

        # Store model and its ROC AUC for later selection
        models.append({
            'model': clf,
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
# 6) SELECT AND RETRAIN BEST MODELS
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
        model: The evaluated model.
    """
    print(f"\n[{method_name}] Evaluating on the entire dataset.")

    # Feature Scaling
    X_scaled = scaler.transform(X)

    # Predict using the trained model
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
    roc_curve_path = os.path.join(plots_dir, f"{method_name.lower()}_entire_dataset_roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.close()
    print(f"[{method_name}] ROC curve saved to {roc_curve_path}.")

    # Plot Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Control', 'Alzheimer'])
    disp.plot(cmap='Oranges')
    plt.title(f"{method_name} Confusion Matrix - Entire Dataset")
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, f"{method_name.lower()}_entire_dataset_confusion_matrix.png")
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
    metrics_filename = os.path.join(models_dir, f"{method_name.lower()}_evaluation_metrics.json")
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[{method_name}] Evaluation metrics saved to {metrics_filename}.")

    return model


def plot_mlp_feature_importance(model, X, y, feature_names, scaler=None, 
                                scoring='roc_auc', n_repeats=30, random_state=42,
                                output_dir='plots', filename='mlp_feature_importance.png'):
    """
    Calculates and plots feature importance for an MLPClassifier using permutation importance.

    Args:
        model: Trained MLPClassifier model.
        X: Feature matrix (NumPy array or pandas DataFrame).
        y: Labels (NumPy array or pandas Series).
        feature_names: List of feature names.
        scaler: Scaler object used to scale X (e.g., StandardScaler). If provided, X will be transformed before importance calculation.
        scoring: Scoring metric for permutation importance (default: 'roc_auc').
        n_repeats: Number of times to permute a feature (default: 30).
        random_state: Random state for reproducibility (default: 42).
        output_dir: Directory to save the feature importance plot.
        filename: Name of the saved PNG file.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # If a scaler is provided, transform the data
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.copy()
    
    # Calculate permutation importance
    print("[INFO] Calculating permutation importance...")
    result = permutation_importance(model, X_scaled, y, scoring=scoring,
                                    n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    
    # Extract importances
    importances = result.importances_mean
    importances_std = result.importances_std
    
    # Create a DataFrame for visualization
    importance_df = np.vstack((importances, importances_std)).T
    importance_df = importance_df[np.argsort(importances)]
    sorted_indices = np.argsort(importances)
    
    sorted_importances = importances[sorted_indices]
    sorted_importances_std = importances_std[sorted_indices]
    sorted_features = [feature_names[i] for i in sorted_indices]
    
    # Plotting
    plt.figure(figsize=(10, max(6, len(feature_names)*0.3)))  # Adjust height based on number of features
    sns.set_style("whitegrid")
    ax = sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis', 
                     edgecolor='black', linewidth=0.8)
    
    # Add error bars
    ax.errorbar(sorted_importances, range(len(sorted_features)), 
                xerr=sorted_importances_std, fmt='none', c='black', capsize=3)
    
    # Add titles and labels
    plt.title(f'Feature Importance using Permutation Importance ({scoring})', fontsize=16, fontweight='bold')
    plt.xlabel('Decrease in Model Performance')
    plt.ylabel('Features')
    
    # Annotate bars with importance values
    for i, (imp, imp_std) in enumerate(zip(sorted_importances, sorted_importances_std)):
        plt.text(imp + max(sorted_importances)*0.01, i, f"{imp:.4f}", va='center')
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()
    
    print(f"[INFO] Feature importance plot saved as '{save_path}'.")

def plot_logreg_coefficients(model, feature_names):
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

    plt.figure(figsize=(10, 6))
    sns.barplot(x=coefficients, y=feature_names, palette='coolwarm')
    plt.title("Logistic Regression Coefficients")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.axvline(0, color='grey', linestyle='--')
    plt.tight_layout()
    coef_plot_path = os.path.join(PLOTS_DIR, "logreg_coefficients.png")
    plt.savefig(coef_plot_path)
    plt.close()
    print(f"[INFO] Logistic Regression coefficients plot saved to {coef_plot_path}.")

# Example usage after retraining your logistic regression:

# --------------------------------------------------------------------------------
# 7) MAIN PIPELINE
# --------------------------------------------------------------------------------

def main():
    # 1) Load data in parallel with optimized feature extraction
    print("[DEBUG] Loading dataset...")
    X, y = load_dataset_parallel()
    print(f"[DEBUG] Dataset loaded with {X.shape[0]} samples and {X.shape[1]} features.")
    
    # 2) Define feature names for reference
    band_names = list(FREQUENCY_BANDS.keys())  # ['Delta', 'Theta1', 'Theta2', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']
    feature_names = band_names + [
        "Alpha_Ratio", "Theta_Ratio",
        "Shannon_Entropy", "Spatial_Complexity",
        "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity"
    ]
    
    # 3) Analyze and visualize features
    stats_df = analyze_feature_statistics(X, y, feature_names, output_dir=FEATURE_ANALYSIS_DIR)
    
    # Optional: Display top 5 statistically significant features
    print("\nTop 5 Statistically Significant Features (by P-Value):")
    print(stats_df.sort_values(by="P-Value").head(5))
    
    # 4) Baseline Models
    print("\n--- Baseline Logistic Regression ---")
    logreg_models, logreg_metrics = baseline_logistic_regression(X, y, feature_names, models_dir=MODELS_DIR, plots_dir=PLOTS_DIR)
    
    print("\n--- Baseline MLP ---")
    mlp_models, mlp_metrics = baseline_mlp(X, y, feature_names, models_dir=MODELS_DIR, plots_dir=PLOTS_DIR)
    
    # 5) Select Best Models
    best_logreg_model, best_logreg_scaler = select_best_model(logreg_models, method_name='LogReg')
    best_mlp_model, best_mlp_scaler = select_best_model(mlp_models, method_name='MLP')
    
    # 6) Evaluate Best Models on Entire Dataset
    evaluated_logreg = evaluate_model_on_full_data(
        best_logreg_model, 
        best_logreg_scaler, 
        X, 
        y, 
        method_name='LogReg', 
        models_dir=MODELS_DIR, 
        plots_dir=PLOTS_DIR
    )
    evaluated_mlp = evaluate_model_on_full_data(
        best_mlp_model, 
        best_mlp_scaler, 
        X, 
        y, 
        method_name='MLP', 
        models_dir=MODELS_DIR, 
        plots_dir=PLOTS_DIR
    )
    
    # Plot Logistic Regression Coefficients
    plot_logreg_coefficients(evaluated_logreg, feature_names)

    # Optionally, plot MLP weights or other analyses
    # plot_mlp_weights(evaluated_mlp)
    # visualize_mlp_architecture(evaluated_mlp)
    
    print("\n[INFO] Best models have been evaluated on the entire dataset and their accuracies have been reported.")

if __name__ == "__main__":
    main()
    
