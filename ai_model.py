#!/usr/bin/env python
"""
EEG Analysis & Modeling Pipeline with PyTorch GCN and Combined Channel Vector (CCV)
+ TensorBoard logging and visual representations for downstream models (SGD logistic regression and MLP)

Pipeline summary:
  1. Each EEG file is loaded once. For each subject:
       - Handcrafted branch: per-epoch features are computed (with Numba-accelerated Hjorth parameters)
         and then summarized (mean and std) into a global feature vector.
       - GNN branch: after applying a Laplacian montage, channel-level band-power features are computed per epoch.
  2. The GNN branch features are averaged over epochs to form a per-subject graph input.
  3. A PyTorch Geometric GCN (with a classification head) is trained using the actual subject labels.
     Training and validation losses are logged to TensorBoard and saved as a loss plot.
  4. The latent embedding from the GCN is concatenated with the handcrafted features to form the Combined Channel Vector (CCV).
  5. Downstream, we train two models using scikit-learn via partial_fit:
       - An SGD-based Logistic Regression model.
       - An MLP classifier.
     For each model, training loss is logged (and saved as a plot) and final model “architectures” are visualized
     (via a bar chart for SGD coefficients and weight heatmaps for the MLP).
  6. ROC curves and classification reports for each downstream model are generated and saved.
"""

#############################################
# CONFIGURATION & IMPORTS
#############################################
import os, glob, json, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay, log_loss)
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
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

# Suppress warnings and logs
mne.set_log_level('WARNING')
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# 1) LOAD PARTICIPANT LABELS
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# 2) HANDCRAFTED FEATURE EXTRACTION (Per Epoch) and Aggregation
# --------------------------------------------------------------------------------
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

def process_subject_handcrafted(file, label):
    try:
        raw = mne.io.read_raw_eeglab(file, preload=True, verbose=False)
        raw.filter(1.0, 50.0, fir_design="firwin", verbose=False)
        raw.resample(SAMPLING_RATE, npad="auto")
        epochs = mne.make_fixed_length_epochs(raw, duration=20.0, overlap=0.0, verbose=False)
        data = epochs.get_data()
        if data.size == 0:
            raise ValueError("No data extracted from epochs.")
        epoch_features = np.array([extract_features_epoch(epoch) for epoch in data])
        mean_features = np.mean(epoch_features, axis=0)
        std_features = np.std(epoch_features, axis=0)
        global_features = np.hstack((mean_features, std_features))
        return global_features, epoch_features, label
    except Exception as e:
        print(f"[ERROR] Handcrafted branch failed for {file}: {e}")
        return None, None, None

def run_process_subject_handcrafted(args):
    return process_subject_handcrafted(*args)

def load_dataset_handcrafted():
    dataset_paths = [DS004504_PATH, DS003800_PATH]
    all_files = []
    for path in dataset_paths:
        pattern = os.path.join(path, "**", "*_task-Rest_eeg.set") if 'ds003800' in path else os.path.join(path, "**", "*.set")
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    tasks = [(f, participant_labels.get(os.path.basename(f).split('_')[0], None))
             for f in all_files if participant_labels.get(os.path.basename(f).split('_')[0], None) is not None]
    features_list = []
    labels_list = []
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(run_process_subject_handcrafted, tasks), total=len(tasks)))
    for feat, _, lab in results:
        if feat is not None:
            features_list.append(feat)
            labels_list.append(lab)
    X_handcrafted = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    print(f"[DEBUG] Handcrafted dataset: {X_handcrafted.shape[0]} subjects, {X_handcrafted.shape[1]} features.")
    return X_handcrafted, y

# --------------------------------------------------------------------------------
# 3) GNN BRANCH: Feature Extraction Per Epoch
# --------------------------------------------------------------------------------
def extract_channel_features_GNN_epoch(epoch_data, sfreq):
    n_channels = epoch_data.shape[0]
    features = np.zeros((n_channels, len(FREQUENCY_BANDS)), dtype=np.float32)
    for ch in range(n_channels):
        ts = epoch_data[ch, :]
        psd, freqs = mne.time_frequency.psd_array_multitaper(ts[np.newaxis, :], sfreq=sfreq, verbose=False)
        band_vals = []
        for band, (fmin, fmax) in FREQUENCY_BANDS.items():
            idx = (freqs >= fmin) & (freqs <= fmax)
            band_vals.append(np.mean(psd[0, idx]))
        features[ch, :] = np.array(band_vals, dtype=np.float32)
    return features

def process_subject_gnn(file, label):
    try:
        raw = mne.io.read_raw_eeglab(file, preload=True, verbose=False)
        raw.filter(1.0, 50.0, fir_design="firwin", verbose=False)
        raw.resample(SAMPLING_RATE, npad="auto")
        raw = mne.preprocessing.compute_current_source_density(raw)
        epochs = mne.make_fixed_length_epochs(raw, duration=20.0, overlap=0.0, verbose=False)
        data = epochs.get_data()
        if data.size == 0:
            raise ValueError("No data extracted from epochs.")
        epoch_channel_features = np.array([extract_channel_features_GNN_epoch(epoch, SAMPLING_RATE) for epoch in data])
        ch_names = raw.ch_names
        return epoch_channel_features, label, ch_names
    except Exception as e:
        print(f"[ERROR] GNN branch failed for {file}: {e}")
        return None, None, None

def run_process_subject_gnn(args):
    return process_subject_gnn(*args)

def load_dataset_gnn():
    dataset_paths = [DS004504_PATH, DS003800_PATH]
    all_files = []
    for path in dataset_paths:
        pattern = os.path.join(path, "**", "*_task-Rest_eeg.set") if 'ds003800' in path else os.path.join(path, "**", "*.set")
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    tasks = [(f, participant_labels.get(os.path.basename(f).split('_')[0], None))
             for f in all_files if participant_labels.get(os.path.basename(f).split('_')[0], None) is not None]
    features_list = []
    labels_list = []
    ch_names = None
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(run_process_subject_gnn, tasks), total=len(tasks)))
    for feat, lab, chn in results:
        if feat is not None:
            features_list.append(feat)
            labels_list.append(lab)
            if ch_names is None:
                ch_names = chn
    X_channels = np.array(features_list, dtype=np.float32)  # (n_subjects, n_epochs, n_channels, n_band_features)
    y_gnn = np.array(labels_list, dtype=np.int32)
    print(f"[INFO] GNN raw dataset: {X_channels.shape[0]} subjects, {X_channels.shape[1]} epochs, {X_channels.shape[2]} channels, {X_channels.shape[3]} features.")
    return X_channels, y_gnn, ch_names

# --------------------------------------------------------------------------------
# 4) AGGREGATE GNN FEATURES PER SUBJECT VIA A GCN
# --------------------------------------------------------------------------------
def aggregate_gnn_input(X_channels):
    return np.mean(X_channels, axis=1)  # (n_subjects, n_channels, n_band_features)

# --------------------------------------------------------------------------------
# 5) PyTorch Geometric GCN MODEL DEFINITION (with classification head)
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# 6) CREATE PYG DATASET FROM AGGREGATED GNN FEATURES
# --------------------------------------------------------------------------------
def compute_adjacency_matrix(ch_names):
    montage = mne.channels.make_standard_montage('standard_1020')
    pos_dict = montage.get_positions()['ch_pos']
    nodes = []
    for ch in ch_names:
        if ch in pos_dict:
            nodes.append(pos_dict[ch])
    nodes = np.array(nodes)
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(nodes))
    threshold = np.percentile(D, 30)
    A = (D < threshold).astype(np.float32)
    np.fill_diagonal(A, 0)
    return A

def create_pyg_dataset(X_subjects, y, ch_names):
    A = compute_adjacency_matrix(ch_names)
    edge_index = np.array(np.nonzero(A))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    pyg_data_list = []
    for i in range(X_subjects.shape[0]):
        x = torch.tensor(X_subjects[i], dtype=torch.float)
        y_val = torch.tensor([y[i]], dtype=torch.long)
        data_obj = Data(x=x, edge_index=edge_index, y=y_val)
        pyg_data_list.append(data_obj)
    return pyg_data_list

# --------------------------------------------------------------------------------
# 7) DOWNSTREAM CLASSIFICATION WITH TENSORBOARD LOGGING and ARCHITECTURE VISUALIZATION
# --------------------------------------------------------------------------------
from sklearn.metrics import log_loss

def train_sgd_logistic_tb(X_train, y_train, epochs=200, log_dir="./logs/sgd_logreg"):
    from sklearn.linear_model import SGDClassifier
    writer = SummaryWriter(log_dir=log_dir)
    clf = SGDClassifier(loss="modified_huber", penalty="l2", verbose=1,learning_rate="optimal", random_state=5)
    classes = np.unique(y_train)
    loss_history = []
    for epoch in range(1, epochs+1):
        clf.partial_fit(X_train, y_train, classes=classes)
        y_pred_proba = clf.predict_proba(X_train)
        loss = log_loss(y_train, y_pred_proba)
        loss_history.append(loss)
        writer.add_scalar("SGD_LogReg/Train_Loss-", loss, epoch)
    writer.close()
    # Plot loss history
    plt.figure()
    plt.plot(loss_history, label="SGD LogReg Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("SGD Logistic Regression Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sgd_logreg_loss.png"))
    plt.close()
    return clf, loss_history

def train_mlp_tb(X_train, y_train, epochs=200, log_dir="./logs/mlp"):
    from sklearn.neural_network import MLPClassifier
    writer = SummaryWriter(log_dir=log_dir)
    mlp = MLPClassifier(hidden_layer_sizes=(26,14), activation='logistic', solver='adam', max_iter=2, verbose=1,warm_start=False, random_state=5)
    loss_history = []
    for epoch in range(1, epochs+1):
        mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
        # scikit-learn's MLPClassifier doesn't provide loss_ after partial_fit in every version,
        # so we compute log_loss manually:
        y_pred_proba = mlp.predict_proba(X_train)
        loss = log_loss(y_train, y_pred_proba)
        loss_history.append(loss)
        writer.add_scalar("MLP/Train_Loss", loss, epoch)
    writer.close()
    plt.figure()
    plt.plot(loss_history, label="MLP Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("MLP Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "mlp_loss.png"))
    plt.close()
    return mlp, loss_history

def visualize_sgd_architecture(clf):
    # For a linear model, we can plot the coefficients as a bar chart.
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
    # Visualize each weight matrix from the MLP as a heatmap.
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
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(CCV, y), start=1):
        X_train, X_test = CCV[train_idx], CCV[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        smote = SMOTE(random_state=5)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        
        # Train SGD Logistic Regression with TensorBoard logging
        sgd_model, sgd_loss = train_sgd_logistic_tb(X_train_res, y_train_res, epochs=200, log_dir=os.path.join(LOG_DIR, "sgd_logreg"))
        y_pred_sgd = sgd_model.predict(X_test_scaled)
        y_proba_sgd = sgd_model.predict_proba(X_test_scaled)[:, 1]
        sgd_acc = accuracy_score(y_test, y_pred_sgd)
        sgd_auc = roc_auc_score(y_test, y_proba_sgd)
        sgd_acc_list.append(sgd_acc)
        sgd_auc_list.append(sgd_auc)
        fpr_sgd, tpr_sgd, _ = roc_curve(y_test, y_proba_sgd)
        all_fprs_sgd.append(fpr_sgd)
        all_tprs_sgd.append(tpr_sgd)
        print(f"[SGD LogReg Fold {fold}] Classification Report:")
        print(classification_report(y_test, y_pred_sgd, target_names=['Control', 'Alzheimer']))
        
        # Train MLP with partial_fit and TensorBoard logging
        mlp_model, mlp_loss = train_mlp_tb(X_train_res, y_train_res, epochs=200, log_dir=os.path.join(LOG_DIR, "mlp"))
        y_pred_mlp = mlp_model.predict(X_test_scaled)
        y_proba_mlp = mlp_model.predict_proba(X_test_scaled)[:, 1]
        mlp_acc = accuracy_score(y_test, y_pred_mlp)
        mlp_auc = roc_auc_score(y_test, y_proba_mlp)
        mlp_acc_list.append(mlp_acc)
        mlp_auc_list.append(mlp_auc)
        fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_proba_mlp)
        all_fprs_mlp.append(fpr_mlp)
        all_tprs_mlp.append(tpr_mlp)
        print(f"[MLP Fold {fold}] Classification Report:")
        print(classification_report(y_test, y_pred_mlp, target_names=['Control', 'Alzheimer']))
    
    print("\n[SGD LogReg] Average Accuracy: {:.4f}, AUC: {:.4f}".format(np.mean(sgd_acc_list), np.mean(sgd_auc_list)))
    print("[MLP] Average Accuracy: {:.4f}, AUC: {:.4f}".format(np.mean(mlp_acc_list), np.mean(mlp_auc_list)))
    
    # Plot ROC curves for SGD Logistic Regression
    plt.figure()
    for i in range(len(all_fprs_sgd)):
        plt.plot(all_fprs_sgd[i], all_tprs_sgd[i], label=f'SGD LogReg Fold {i+1}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("SGD Logistic Regression ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sgd_logreg_roc.png"))
    plt.close()
    
    # Plot ROC curves for MLP
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
    
    # Visualize final architectures:
    visualize_sgd_architecture(sgd_model)
    visualize_mlp_architecture(mlp_model)

# --------------------------------------------------------------------------------
# 8) COMPLEX/COMBINED DATASET LOADING: Process Each File Once for Both Branches
# --------------------------------------------------------------------------------
def process_subject_combined(args):
    file, label = args
    try:
        # Handcrafted branch (without Laplacian)
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
        
        # GNN branch: apply Laplacian montage then extract channel-level features per epoch
        raw_lap = raw.copy()
        raw_lap = mne.preprocessing.compute_current_source_density(raw_lap)
        epochs_lap = mne.make_fixed_length_epochs(raw_lap, duration=20.0, overlap=0.0, verbose=False)
        data_lap = epochs_lap.get_data()
        gnn_epoch_features = np.array([extract_channel_features_GNN_epoch(epoch, SAMPLING_RATE) for epoch in data_lap])
        ch_names = raw_lap.ch_names
        return handcrafted_global, epoch_hand_features, gnn_epoch_features, label, ch_names
    except Exception as e:
        print(f"[ERROR] Combined processing failed for {file}: {e}")
        return None, None, None, None, None

def run_process_subject_combined(args):
    return process_subject_combined(args)

def load_combined_dataset():
    dataset_paths = [DS004504_PATH, DS003800_PATH]
    all_files = []
    for path in dataset_paths:
        pattern = os.path.join(path, "**", "*_task-Rest_eeg.set") if 'ds003800' in path else os.path.join(path, "**", "*.set")
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    tasks = [(f, participant_labels.get(os.path.basename(f).split('_')[0], None))
             for f in all_files if participant_labels.get(os.path.basename(f).split('_')[0], None) is not None]
    hand_features_list = []
    gnn_features_list = []
    labels_list = []
    ch_names = None
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(run_process_subject_combined, tasks), total=len(tasks)))
    for res in results:
        if res[0] is not None and res[2] is not None:
            hand_features_list.append(res[0])
            gnn_features_list.append(np.mean(res[2], axis=0))  # aggregate over epochs per subject
            labels_list.append(res[3])
            if ch_names is None:
                ch_names = res[4]
    X_handcrafted = np.array(hand_features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    X_gnn = np.array(gnn_features_list, dtype=np.float32)
    print(f"[DEBUG] Handcrafted global features: {X_handcrafted.shape[0]} subjects, {X_handcrafted.shape[1]} features.")
    print(f"[INFO] GNN aggregated features: {X_gnn.shape[0]} subjects, {X_gnn.shape[1]} channels, {X_gnn.shape[2]} features per channel.")
    return X_handcrafted, y, X_gnn, ch_names

# --------------------------------------------------------------------------------
# 9) MAIN PIPELINE
# --------------------------------------------------------------------------------
def main():
    np.random.seed(5)
    tf.random.set_seed(5)
    
    # Load combined dataset
    X_handcrafted, y, X_gnn_raw, ch_names = load_combined_dataset()
    print(f"[DEBUG] Handcrafted global features: {X_handcrafted.shape[0]} subjects, {X_handcrafted.shape[1]} features.")
    
    # X_gnn_raw is aggregated per subject; shape: (n_subjects, n_channels, n_band_features)
    X_gnn = X_gnn_raw
    
    # Create PyG dataset from aggregated GNN features
    pyg_dataset = create_pyg_dataset(X_gnn, y, ch_names)
    
    # Visualize the GCN graph structure using networkx
    def visualize_gcn_graph(ch_names):
        A = compute_adjacency_matrix(ch_names)
        G = nx.from_numpy_array(A)
        plt.figure(figsize=(8,8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=300, font_size=8)
        plt.title("GCN Input Graph Structure")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "gcn_graph.png"))
        plt.close()
    visualize_gcn_graph(ch_names)
    
    # Split PyG dataset into training and validation sets for GCN training.
    indices = np.arange(len(pyg_dataset))
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=5,
                                           stratify=np.array([d.y.item() for d in pyg_dataset]))
    train_dataset = [pyg_dataset[i] for i in train_idx]
    val_dataset = [pyg_dataset[i] for i in val_idx]
    train_loader = PyGDataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Set up TensorBoard writer for GCN training
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, "gcn"))
    
    # Train the GCN using actual subject labels.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = X_gnn.shape[2]
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
    
    # Plot GCN loss curves
    plt.figure()
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GCN Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gcn_loss.png"))
    plt.close()
    
    # Extract latent embeddings from GCN (using embed() method)
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
    
    # Form the Combined Channel Vector (CCV)
    if X_handcrafted.shape[0] != gcn_embeddings.shape[0]:
        print("[ERROR] Mismatch between handcrafted features and GCN embeddings.")
        return
    CCV = np.hstack((X_handcrafted, gcn_embeddings))
    print(f"[INFO] CCV shape: {CCV.shape}")
    
    # Downstream Classification with TensorBoard logging for SGD LogReg and MLP
    downstream_classification(CCV, y)
    
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
