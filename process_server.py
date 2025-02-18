#!/usr/bin/env python
"""
EEG Analysis & Modeling Pipeline with PyTorch GCN and Combined Channel Vector (CCV)
Offloading heavy processing with Ray, batching file uploads with ray.put(), and
extracting channel-level features for the GNN branch.
"""

#############################################
# CONFIGURATION & IMPORTS
#############################################
import os, glob, json, joblib, warnings, tempfile, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, log_loss)
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import mne
import networkx as nx

# PyTorch and PyTorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam as TorchAdam
from torch_geometric.data import Data, DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.tensorboard import SummaryWriter

# Numba for speed-up
from numba import njit

# Ray
import ray

# Optional: Graphviz for architecture diagram (ensure you have graphviz installed)
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

# Suppress warnings and logs
mne.set_log_level('WARNING')
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#############################################
# CONFIGURATION (Update paths as needed)
#############################################
DS004504_PATH = "/Users/ishaangubbala/Documents/SF/ds004504/derivatives"
DS003800_PATH = "/Users/ishaangubbala/Documents/SF/ds003800/"
PARTICIPANTS_FILE_DS004504 = "/Users/ishaangubbala/Documents/SF/ds004504/participants.tsv"
PARTICIPANTS_FILE_DS003800 = "/Users/ishaangubbala/Documents/SF/ds003800/participants.tsv"
SAMPLING_RATE = 256  # Hz
FREQUENCY_BANDS = {
    "Delta": (0.5, 4), "Theta1": (4, 6), "Theta2": (6, 8),
    "Alpha1": (8, 10), "Alpha2": (10, 12),
    "Beta1": (12, 20), "Beta2": (20, 30),
    "Gamma1": (30, 40), "Gamma2": (40, 50)
}
FEATURE_ANALYSIS_DIR = "feature_analysis"
MODELS_DIR = "trained_models"
PLOTS_DIR = "plots"
LOG_DIR = "./logs"
for d in [FEATURE_ANALYSIS_DIR, MODELS_DIR, PLOTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

#############################################
# 1) LOAD PARTICIPANT LABELS
#############################################
def load_participant_labels(ds004504_file, ds003800_file):
    label_dict = {}
    group_map_ds004504 = {"A": 1, "C": 0}
    df_ds004504 = pd.read_csv(ds004504_file, sep="\t")
    df_ds004504 = df_ds004504[df_ds004504['Group'] != 'F']
    df_ds004504 = df_ds004504[df_ds004504['Group'].isin(group_map_ds004504.keys())]
    labels_ds004504 = df_ds004504.set_index("participant_id")["Group"].map(group_map_ds004504).to_dict()
    label_dict.update(labels_ds004504)
    df_ds003800 = pd.read_csv(ds003800_file, sep="\t")
    labels_ds003800 = df_ds003800.set_index("participant_id")["Group"].apply(lambda x: 1).to_dict()
    label_dict.update(labels_ds003800)
    return label_dict

participant_labels = load_participant_labels(PARTICIPANTS_FILE_DS004504, PARTICIPANTS_FILE_DS003800)

#############################################
# 2) HANDCRAFTED FEATURE EXTRACTION FUNCTIONS
#############################################
def compute_band_powers(data, sfreq):
    psd, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=sfreq, verbose=False)
    band_powers = {band: np.mean(psd[:, :, (freqs >= fmin) & (freqs <= fmax)], axis=2)
                   for band, (fmin, fmax) in FREQUENCY_BANDS.items()}
    return band_powers

def compute_shannon_entropy(data):
    def entropy_fn(x):
        counts, _ = np.histogram(x, bins=256)
        p = counts / np.sum(counts)
        return -np.sum(p * np.log2(p + 1e-12))
    return np.apply_along_axis(entropy_fn, 1, data)

@njit
def compute_hjorth_parameters_numba(data):
    n_epochs, n_channels, n_times = data.shape
    activities = np.empty(n_epochs, dtype=np.float64)
    mobilities = np.empty(n_epochs, dtype=np.float64)
    complexities = np.empty(n_epochs, dtype=np.float64)
    for i in range(n_epochs):
        var0 = np.empty(n_channels, dtype=np.float64)
        for j in range(n_channels):
            s = 0.0
            for k in range(n_times):
                s += data[i, j, k]
            mean_val = s / n_times
            s2 = 0.0
            for k in range(n_times):
                diff = data[i, j, k] - mean_val
                s2 += diff * diff
            var0[j] = s2 / n_times
            if var0[j] < 1e-12:
                var0[j] = 1e-12
        mobility = np.empty(n_channels, dtype=np.float64)
        for j in range(n_channels):
            s2 = 0.0
            for k in range(1, n_times):
                diff = data[i, j, k] - data[i, j, k-1]
                s2 += diff * diff
            mobility[j] = np.sqrt(s2 / ((n_times - 1) * var0[j]))
        complexity = np.empty(n_channels, dtype=np.float64)
        for j in range(n_channels):
            s2 = 0.0
            for k in range(2, n_times):
                diff1 = data[i, j, k-1] - data[i, j, k-2]
                diff2 = data[i, j, k] - data[i, j, k-1]
                second_diff = diff2 - diff1
                s2 += second_diff * second_diff
            if s2 < 1e-12:
                s2 = 1e-12
            complexity[j] = np.sqrt(s2 / (n_times - 2)) / (mobility[j] + 1e-12)
        activities[i] = np.mean(var0)
        mobilities[i] = np.mean(mobility)
        complexities[i] = np.mean(complexity)
    return activities, mobilities, complexities

def extract_features_epoch(epoch_data):
    """Extract features for the handcrafted branch (per epoch)."""
    data = epoch_data[np.newaxis, :, :]
    band_powers = compute_band_powers(data, SAMPLING_RATE)
    features = [np.mean(band_powers[band]) for band in FREQUENCY_BANDS.keys()]
    alpha_power = np.mean(band_powers["Alpha1"]) + np.mean(band_powers["Alpha2"])
    theta_power = np.mean(band_powers["Theta1"]) + np.mean(band_powers["Theta2"])
    total_power = sum(np.mean(band_powers[band]) for band in FREQUENCY_BANDS.keys()) + 1e-12
    features.extend([alpha_power/total_power, theta_power/total_power])
    sh_entropy = np.mean(compute_shannon_entropy(data))
    features.append(sh_entropy)
    act, mob, comp = compute_hjorth_parameters_numba(data)
    features.extend([np.mean(act), np.mean(mob), np.mean(comp)])
    return np.array(features, dtype=np.float32)

#############################################
# NEW: GNN CHANNEL-FEATURE EXTRACTION FUNCTION
#############################################
def extract_channel_features_GNN_epoch(epoch_data, sfreq):
    """
    Extract features for each channel from an epoch.
    Returns an array of shape [n_channels, num_bands] where num_bands = len(FREQUENCY_BANDS).
    """
    n_channels = epoch_data.shape[0]
    num_bands = len(FREQUENCY_BANDS)
    features = np.zeros((n_channels, num_bands), dtype=np.float32)
    for ch in range(n_channels):
        ts = epoch_data[ch, :]
        psd, freqs = mne.time_frequency.psd_array_multitaper(ts[np.newaxis, :], sfreq=sfreq, verbose=False)
        band_vals = []
        for band, (fmin, fmax) in FREQUENCY_BANDS.items():
            idx = (freqs >= fmin) & (freqs <= fmax)
            band_vals.append(np.mean(psd[0, idx]))
        features[ch, :] = np.array(band_vals, dtype=np.float32)
    return features

#############################################
# 3) COMBINED PROCESSING (HANDCRAFTED + GNN)
#############################################
def process_subject_combined(args):
    file, label = args
    if not isinstance(file, str):
        try:
            file_data = ray.get(file)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".set") as tmp:
                tmp.write(file_data)
                tmp.flush()
                file = tmp.name
        except Exception as e:
            print(f"[ERROR] Failed to write temporary file: {e}")
            return None, None, None, None, None
    try:
        raw = mne.io.read_raw_eeglab(file, preload=True, verbose=False)
        raw.filter(1.0, 50.0, fir_design="firwin", verbose=False)
        raw.resample(SAMPLING_RATE, npad="auto")
        epochs = mne.make_fixed_length_epochs(raw, duration=20.0, overlap=0.0, verbose=False)
        data = epochs.get_data()
        if data.size == 0:
            raise ValueError("No data extracted from epochs.")
        epoch_hand_features = np.array([extract_features_epoch(epoch) for epoch in data])
        mean_features = np.mean(epoch_hand_features, axis=0)
        std_features = np.std(epoch_hand_features, axis=0)
        handcrafted_global = np.hstack((mean_features, std_features))
        # For the GNN branch, extract channel-level features.
        raw_lap = raw.copy()
        raw_lap = mne.preprocessing.compute_current_source_density(raw_lap)
        epochs_lap = mne.make_fixed_length_epochs(raw_lap, duration=20.0, overlap=0.0, verbose=False)
        data_lap = epochs_lap.get_data()  # shape: (num_epochs, n_channels, n_times)
        channel_features_epochs = np.array([extract_channel_features_GNN_epoch(epoch, SAMPLING_RATE) for epoch in data_lap])
        # Average channel features across epochs (subject-level).
        gnn_aggregated_features = np.mean(channel_features_epochs, axis=0)  # shape: [n_channels, num_bands]
        ch_names = raw_lap.ch_names
        return handcrafted_global, epoch_hand_features, gnn_aggregated_features, label, ch_names
    except Exception as e:
        print(f"[ERROR] Combined processing failed for {file}: {e}")
        return None, None, None, None, None

def run_process_subject_combined(args):
    return process_subject_combined(args)

#############################################
# 4) LOAD COMBINED DATASET USING ray.put() FOR FILES
#############################################
def load_combined_dataset():
    dataset_paths = [DS004504_PATH, DS003800_PATH]
    all_files = []
    for path in dataset_paths:
        pattern = os.path.join(path, "**", "*_task-Rest_eeg.set") if 'ds003800' in path else os.path.join(path, "**", "*.set")
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    tasks = []
    for f in all_files:
        subj = os.path.basename(f).split('_')[0]
        label = participant_labels.get(subj, None)
        if label is not None:
            try:
                with open(f, "rb") as file_obj:
                    data = file_obj.read()
                file_ref = ray.put(data)
                tasks.append((file_ref, label))
            except Exception as e:
                print(f"[ERROR] Failed to read and upload {f}: {e}")
    hand_features_list = []
    gnn_features_list = []
    labels_list = []
    ch_names_list = []

    if not ray.is_initialized():
        ray.init(address="ray://3.145.84.12:10001")  # Update with your head node IP/port.

    run_process_subject_combined_remote = ray.remote(run_process_subject_combined)
    futures = [run_process_subject_combined_remote.remote(task) for task in tasks]
    results = ray.get(futures)

    for idx, res in enumerate(results):
        if res[0] is not None and res[2] is not None and res[4] is not None:
            hand_features_list.append(res[0])
            gnn_features_list.append(res[2])
            labels_list.append(res[3])
            ch_names_list.append(res[4])
        else:
            print(f"[WARNING] Subject {idx+1} returned None features and will be skipped.")

    if not hand_features_list:
        raise ValueError("No valid handcrafted/GNN features were extracted.")

    X_handcrafted = np.stack(hand_features_list, axis=0).astype(np.float32)
    y = np.array(labels_list, dtype=np.int32)
    print(f"[DEBUG] Handcrafted global features: {X_handcrafted.shape[0]} subjects, {X_handcrafted.shape[1]} features.")
    print(f"[INFO] GNN aggregated features: {len(gnn_features_list)} subjects, variable nodes per subject.")
    return X_handcrafted, y, gnn_features_list, ch_names_list

#############################################
# 5) GCN MODEL DEFINITION & UTILS
#############################################
class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        logits = self.lin(x)
        return logits
    def embed(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return x

def compute_adjacency_matrix(ch_names):
    montage = mne.channels.make_standard_montage('standard_1020')
    pos_dict = montage.get_positions()['ch_pos']
    nodes = []
    for ch in ch_names:
        if ch in pos_dict:
            nodes.append(pos_dict[ch])
    if len(nodes) == 0:
        return None
    nodes = np.array(nodes)
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(nodes))
    threshold = np.percentile(D, 30)
    A = (D < threshold).astype(np.float32)
    np.fill_diagonal(A, 0)
    return A

def create_pyg_dataset(X_subjects, y, ch_names_list):
    pyg_data_list = []
    for i in range(len(X_subjects)):
        A = compute_adjacency_matrix(ch_names_list[i])
        if A is None:
            print(f"[WARNING] No valid channels for subject {i+1}.")
            continue
        edge_index = np.array(np.nonzero(A))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(X_subjects[i], dtype=torch.float)
        if x.shape[0] != len(ch_names_list[i]):
            print(f"[ERROR] Subject {i+1} has {x.shape[0]} nodes but expected {len(ch_names_list[i])} nodes.")
            continue
        y_val = torch.tensor([y[i]], dtype=torch.long)
        data_obj = Data(x=x, edge_index=edge_index, y=y_val)
        pyg_data_list.append(data_obj)
    return pyg_data_list

#############################################
# NEW: GCN ARCHITECTURE VISUALIZATION USING GRAPHVIZ
#############################################
def visualize_gcn_architecture():
    if not HAS_GRAPHVIZ:
        print("[INFO] Graphviz is not installed. Skipping GCN architecture diagram.")
        return
    dot = Digraph(comment='GCN Architecture')
    dot.node('A', 'Input Features')
    dot.node('B', 'Adjacency Matrix\n(with self-loops)')
    dot.node('C', 'Aggregation\n(Message Passing)')
    dot.node('D', 'Linear Transform\n(Conv1.lin)')
    dot.node('E', 'ReLU Activation')
    dot.node('F', 'GCN Layer 1 Output')
    dot.node('G', 'Linear Transform\n(Conv2.lin)')
    dot.node('H', 'ReLU Activation')
    dot.node('I', 'GCN Layer 2 Output')
    dot.node('J', 'Global Mean Pooling')
    dot.node('K', 'Final Output Features')
    dot.edge('A', 'C')
    dot.edge('B', 'C')
    dot.edge('C', 'D', label='Conv1')
    dot.edge('D', 'E')
    dot.edge('E', 'F')
    dot.edge('F', 'G', label='Conv2')
    dot.edge('G', 'H')
    dot.edge('H', 'I')
    dot.edge('I', 'J')
    dot.edge('J', 'K')
    dot.render(os.path.join(PLOTS_DIR, "gcn_architecture"), view=False, format='png')

#############################################
# 6) DOWNSTREAM CLASSIFICATION & VISUALIZATION UTILS
#############################################
def train_sgd_logistic_tb(X_train, y_train, X_val, y_val, epochs=200, log_dir="./logs/sgd_logreg"):
    from sklearn.linear_model import SGDClassifier
    writer = SummaryWriter(log_dir=log_dir)
    clf = SGDClassifier(loss="modified_huber", penalty="l2", verbose=1, learning_rate="optimal", random_state=5)
    classes = np.unique(y_train)
    train_loss_history = []
    val_loss_history = []
    for epoch in range(1, epochs+1):
        clf.partial_fit(X_train, y_train, classes=classes)
        y_pred_proba_train = clf.predict_proba(X_train)
        train_loss = log_loss(y_train, y_pred_proba_train)
        train_loss_history.append(train_loss)
        writer.add_scalar("SGD_LogReg/Train_Loss", train_loss, epoch)
        y_pred_proba_val = clf.predict_proba(X_val)
        val_loss = log_loss(y_val, y_pred_proba_val)
        val_loss_history.append(val_loss)
        writer.add_scalar("SGD_LogReg/Validation_Loss", val_loss, epoch)
    writer.close()
    plt.figure()
    plt.plot(train_loss_history, label="Train Loss", marker='o')
    plt.plot(val_loss_history, label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("SGD Logistic Regression Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"sgd_logreg_loss_fold.png"))
    plt.close()
    return clf, train_loss_history, val_loss_history

def train_mlp_tb(X_train, y_train, X_val, y_val, epochs=200, log_dir="./logs/mlp"):
    from sklearn.neural_network import MLPClassifier
    writer = SummaryWriter(log_dir=log_dir)
    mlp = MLPClassifier(hidden_layer_sizes=(24,14), activation='logistic', solver='adam', max_iter=2, verbose=1, warm_start=False, random_state=5)
    train_loss_history = []
    val_loss_history = []
    for epoch in range(1, epochs+1):
        mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
        y_pred_proba_train = mlp.predict_proba(X_train)
        train_loss = log_loss(y_train, y_pred_proba_train)
        train_loss_history.append(train_loss)
        writer.add_scalar("MLP/Train_Loss", train_loss, epoch)
        y_pred_proba_val = mlp.predict_proba(X_val)
        val_loss = log_loss(y_val, y_pred_proba_val)
        val_loss_history.append(val_loss)
        writer.add_scalar("MLP/Validation_Loss", val_loss, epoch)
    writer.close()
    plt.figure()
    plt.plot(train_loss_history, label="Train Loss", marker='o')
    plt.plot(val_loss_history, label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("MLP Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"mlp_loss_fold.png"))
    plt.close()
    return mlp, train_loss_history, val_loss_history

def visualize_sgd_architecture(clf):
    coef = clf.coef_.flatten()
    plt.figure(figsize=(10,5))
    plt.bar(np.arange(len(coef)), coef)
    plt.xlabel("Feature Index")
    plt.ylabel("Coefficient Value")
    plt.title("SGD Logistic Regression Coefficients")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sgd_logreg_coeff.png"))
    plt.close()

def visualize_mlp_architecture(mlp):
    for i, W in enumerate(mlp.coefs_):
        plt.figure(figsize=(6,4))
        sns.heatmap(W, annot=False, cmap="viridis")
        plt.title(f"MLP Layer {i+1} Weight Matrix")
        plt.xlabel("Output Nodes")
        plt.ylabel("Input Nodes")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"mlp_layer_{i+1}_weights.png"))
        plt.close()

def downstream_classification(CCV, y):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5)
    sgd_acc_list, sgd_auc_list = [], []
    mlp_acc_list, mlp_auc_list = [], []
    all_fprs_sgd, all_tprs_sgd = [], []
    all_fprs_mlp, all_tprs_mlp = [], []
    sgd_val_losses = []
    mlp_val_losses = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(CCV, y), start=1):
        X_train, X_test = CCV[train_idx], CCV[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        smote = SMOTE(random_state=5)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        
        sgd_model, sgd_train_loss, sgd_val_loss = train_sgd_logistic_tb(X_train_res, y_train_res, X_test_scaled, y_test, epochs=200, log_dir=os.path.join(LOG_DIR, "sgd_logreg"))
        y_pred_sgd = sgd_model.predict(X_test_scaled)
        y_proba_sgd = sgd_model.predict_proba(X_test_scaled)[:, 1]
        sgd_acc = accuracy_score(y_test, y_pred_sgd)
        sgd_auc = roc_auc_score(y_test, y_proba_sgd)
        sgd_acc_list.append(sgd_acc)
        sgd_auc_list.append(sgd_auc)
        fpr_sgd, tpr_sgd, _ = roc_curve(y_test, y_proba_sgd)
        all_fprs_sgd.append(fpr_sgd)
        all_tprs_sgd.append(tpr_sgd)
        overfit_metric_sgd = (sgd_val_loss[-1] - sgd_train_loss[-1]) / sgd_train_loss[-1]
        sgd_val_losses.append(sgd_val_loss[-1])
        print(f"[SGD Fold {fold}] Overfitting metric: {overfit_metric_sgd:.4f}")
        print(f"[SGD Fold {fold}] Classification Report:")
        print(classification_report(y_test, y_pred_sgd, target_names=['Control', 'Alzheimer']))
        
        mlp_model, mlp_train_loss, mlp_val_loss = train_mlp_tb(X_train_res, y_train_res, X_test_scaled, y_test, epochs=200, log_dir=os.path.join(LOG_DIR, "mlp"))
        y_pred_mlp = mlp_model.predict(X_test_scaled)
        y_proba_mlp = mlp_model.predict_proba(X_test_scaled)[:, 1]
        mlp_acc = accuracy_score(y_test, y_pred_mlp)
        mlp_auc = roc_auc_score(y_test, y_proba_mlp)
        mlp_acc_list.append(mlp_acc)
        mlp_auc_list.append(mlp_auc)
        fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_proba_mlp)
        all_fprs_mlp.append(fpr_mlp)
        all_tprs_mlp.append(tpr_mlp)
        overfit_metric_mlp = (mlp_val_loss[-1] - mlp_train_loss[-1]) / mlp_train_loss[-1]
        mlp_val_losses.append(mlp_val_loss[-1])
        print(f"[MLP Fold {fold}] Overfitting metric: {overfit_metric_mlp:.4f}")
        print(f"[MLP Fold {fold}] Classification Report:")
        print(classification_report(y_test, y_pred_mlp, target_names=['Control', 'Alzheimer']))
        
        # Plot per-fold loss curves.
        plt.figure()
        plt.plot(sgd_train_loss, label="SGD Train Loss", marker='o')
        plt.plot(sgd_val_loss, label="SGD Val Loss", marker='x')
        plt.xlabel("Epoch")
        plt.ylabel("Log Loss")
        plt.title(f"SGD Logistic Regression Loss (Fold {fold})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"sgd_logreg_loss_fold_{fold}.png"))
        plt.close()
        
        plt.figure()
        plt.plot(mlp_train_loss, label="MLP Train Loss", marker='o')
        plt.plot(mlp_val_loss, label="MLP Val Loss", marker='x')
        plt.xlabel("Epoch")
        plt.ylabel("Log Loss")
        plt.title(f"MLP Loss (Fold {fold})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"mlp_loss_fold_{fold}.png"))
        plt.close()
    
    print("\n[SGD LogReg] Average Accuracy: {:.4f}, AUC: {:.4f}".format(np.mean(sgd_acc_list), np.mean(sgd_auc_list)))
    print("[MLP] Average Accuracy: {:.4f}, AUC: {:.4f}".format(np.mean(mlp_acc_list), np.mean(mlp_auc_list)))
    
    # Plot ROC curves.
    plt.figure()
    for i in range(len(all_fprs_sgd)):
        plt.plot(all_fprs_sgd[i], all_tprs_sgd[i], label=f'SGD Fold {i+1}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("SGD Logistic Regression ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sgd_logreg_roc.png"))
    plt.close()
    
    plt.figure()
    for i in range(len(all_fprs_mlp)):
        plt.plot(all_fprs_mlp[i], all_tprs_mlp[i], label=f'MLP Fold {i+1}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MLP ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "mlp_roc.png"))
    plt.close()
    
    # Plot average validation loss across folds.
    plt.figure()
    plt.bar(np.arange(1, len(sgd_val_losses)+1), sgd_val_losses)
    plt.xlabel("Fold")
    plt.ylabel("Validation Loss")
    plt.title("SGD Validation Loss Across Folds")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sgd_val_loss.png"))
    plt.close()
    
    plt.figure()
    plt.bar(np.arange(1, len(mlp_val_losses)+1), mlp_val_losses)
    plt.xlabel("Fold")
    plt.ylabel("Validation Loss")
    plt.title("MLP Validation Loss Across Folds")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "mlp_val_loss.png"))
    plt.close()
    
    visualize_sgd_architecture(sgd_model)
    visualize_mlp_architecture(mlp_model)

#############################################
# 7) MAIN PIPELINE
#############################################
def main():
    np.random.seed(5)
    tf.random.set_seed(5)
    start_time = time.time()
    
    # Optionally, visualize the GCN architecture as a flowchart.
    visualize_gcn_architecture()
    
    # Load combined dataset using ray.put() for files.
    X_handcrafted, y, X_gnn_list, ch_names_list = load_combined_dataset()
    print(f"[DEBUG] Handcrafted global features: {X_handcrafted.shape[0]} subjects, {X_handcrafted.shape[1]} features.")
    
    # Create PyG dataset from per-subject GNN features.
    pyg_dataset = create_pyg_dataset(X_gnn_list, y, ch_names_list)
    
    # Visualize the GCN input graph for the first subject.
    if len(ch_names_list) > 0:
        A = compute_adjacency_matrix(ch_names_list[0])
        if A is not None:
            G = nx.from_numpy_array(A)
            plt.figure(figsize=(8,8))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_size=300, font_size=8)
            plt.title("GCN Input Graph Structure (Subject 1)")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "gcn_graph_subject1.png"))
            plt.close()
    
    # Split PyG dataset into training and validation sets.
    indices = np.arange(len(pyg_dataset))
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=5,
                                           stratify=np.array([d.y.item() for d in pyg_dataset]))
    train_dataset = [pyg_dataset[i] for i in train_idx]
    val_dataset = [pyg_dataset[i] for i in val_idx]
    train_loader = PyGDataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Set up TensorBoard writer for GCN training.
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, "gcn"))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = X_gnn_list[0].shape[1]  # number of bands
    hidden_channels = 32
    num_classes = 2
    gcn_model = GCNNet(in_channels, hidden_channels, num_classes).to(device)
    optimizer = TorchAdam(gcn_model.parameters(), lr=1e-3)
    epochs_gcn = 50
    train_loss_history = []
    val_loss_history = []
    for epoch in range(1, epochs_gcn+1):
        gcn_model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            logits = gcn_model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(logits, data.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        avg_train_loss = total_loss / len(train_loader.dataset)
        
        gcn_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                logits = gcn_model(data.x, data.edge_index, data.batch)
                loss = F.cross_entropy(logits, data.y.view(-1))
                total_val_loss += loss.item() * data.num_graphs
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        if epoch % 10 == 0:
            print(f"GCN Epoch {epoch}/{epochs_gcn} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    writer.close()
    
    # Plot GCN loss curves.
    plt.figure()
    plt.plot(train_loss_history, label="Train Loss", marker='o')
    plt.plot(val_loss_history, label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GCN Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gcn_loss.png"))
    plt.close()
    
    # Compute overall GCN overfitting metric.
    overfit_metric_gcn = (val_loss_history[-1] - train_loss_history[-1]) / train_loss_history[-1]
    print(f"[GCN] Overfitting metric (final val loss gap / train loss): {overfit_metric_gcn:.4f}")
    
    # Visualize GCN linear weights (from the internal linear layers inside GCNConv).
    plt.figure(figsize=(8,6))
    plt.imshow(gcn_model.conv1.lin.weight.detach().cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("GCN Conv1 Linear Weights")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gcn_conv1_weights.png"))
    plt.close()

    plt.figure(figsize=(8,6))
    plt.imshow(gcn_model.conv2.lin.weight.detach().cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("GCN Conv2 Linear Weights")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gcn_conv2_weights.png"))
    plt.close()
    
    full_loader = PyGDataLoader(pyg_dataset, batch_size=4, shuffle=False)
    gcn_embeddings = []
    gcn_model.eval()
    with torch.no_grad():
        for data in full_loader:
            data = data.to(device)
            emb = gcn_model.embed(data.x, data.edge_index, data.batch)
            gcn_embeddings.append(emb.cpu().numpy())
    gcn_embeddings = np.vstack(gcn_embeddings)
    print(f"[INFO] GCN latent embeddings shape: {gcn_embeddings.shape}")
    
    if X_handcrafted.shape[0] != gcn_embeddings.shape[0]:
        print("[ERROR] Mismatch between handcrafted features and GCN embeddings.")
        return
    CCV = np.hstack((X_handcrafted, gcn_embeddings))
    print(f"[INFO] CCV shape: {CCV.shape}")
    
    downstream_classification(CCV, y)
    
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    
if __name__ == "__main__":
    import multiprocessing
    import time
    multiprocessing.set_start_method('spawn', force=True)
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")