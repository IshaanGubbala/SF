#!/usr/bin/env python
"""
Downstream Training Script Using Saved Features (EEG):
Now includes both:
  1) MLP (existing partial-fit approach)
  2) ExtendedQSUP (an extended quantum-inspired superposition model implemented with torch.nn)

We:
  • Load previously saved EEG features.
  • Train a GCN to get embeddings.
  • Combine GCN embeddings + handcrafted features => CCV.
  • Cross-validate with both MLP and ExtendedQSUP, logging losses & generating plots.

Important:
  • The old QSUP model is replaced by ExtendedQSUP with the same hyperparameters as before.
"""

#############################################
# CONFIGURATION & IMPORTS
#############################################
import os
import glob
import json
import time
import numpy as np
import pandas as pd
import mne
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report, log_loss
)
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam as TorchAdam
from torch_geometric.data import Data, DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.tensorboard import SummaryWriter

# Directories
HANDCRAFTED_DIR = "processed_features/handcrafted"
GNN_DIR         = "processed_features/gnn"
CHANNELS_DIR    = "processed_features/channels"
PLOTS_DIR       = "plots"
LOG_DIR         = "logs"
MODELS_DIR      = "trained_models"
for d in [PLOTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

PARTICIPANTS_FILE_DS004504 = "/Users/ishaangubbala/Documents/SF/ds004504/participants.tsv"
PARTICIPANTS_FILE_DS003800 = "/Users/ishaangubbala/Documents/SF/ds003800/participants.tsv"

#############################################
# 1) ENABLE ANOMALY DETECTION (for PyTorch)
#############################################
torch.autograd.set_detect_anomaly(True)

#############################################
# 2) LOAD PARTICIPANT LABELS
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

participant_labels = load_participant_labels(
    PARTICIPANTS_FILE_DS004504,
    PARTICIPANTS_FILE_DS003800
)

#############################################
# 3) LOAD SAVED FEATURE FILES
#############################################
def load_saved_features():
    handcrafted_files = glob.glob(os.path.join(HANDCRAFTED_DIR, "*_handcrafted.npy"))
    X_handcrafted = []
    X_gnn = []
    labels = []
    ch_names_list = []
    subj_ids = []

    for f in handcrafted_files:
        subj_id = os.path.basename(f).split('_')[0]
        gnn_file = os.path.join(GNN_DIR, f"{subj_id}_gnn.npy")
        channels_file = os.path.join(CHANNELS_DIR, f"{subj_id}_channels.json")

        if not os.path.exists(gnn_file) or not os.path.exists(channels_file):
            continue
        if subj_id not in participant_labels:
            continue

        handcrafted = np.load(f)
        gnn_feat = np.load(gnn_file)
        with open(channels_file, "r") as fp:
            ch_names = json.load(fp)

        X_handcrafted.append(handcrafted)
        X_gnn.append(gnn_feat)
        labels.append(participant_labels[subj_id])
        subj_ids.append(subj_id)
        ch_names_list.append(ch_names)

    if len(X_handcrafted) == 0:
        raise ValueError("No saved feature files found!")

    X_handcrafted = np.stack(X_handcrafted, axis=0).astype(np.float32)
    labels = np.array(labels, dtype=np.int32)
    return X_handcrafted, X_gnn, labels, ch_names_list, subj_ids

#############################################
# 4) GCN MODEL & DATASET (PyTorch)
#############################################
class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, bias=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, bias=False)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, bias=False)
        self.lin = nn.Linear(hidden_channels, num_classes, bias=False)

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
    used = [ch for ch in ch_names if ch in pos_dict]
    if not used:
        return None
    coords = np.array([pos_dict[ch] for ch in used])
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(coords))
    threshold = np.percentile(D, 30)
    A = (D < threshold).astype(np.float32)
    np.fill_diagonal(A, 0)
    return A

def create_pyg_dataset(gnn_list, y, ch_names_list):
    pyg_data_list = []
    for i in range(len(gnn_list)):
        A = compute_adjacency_matrix(ch_names_list[i])
        if A is None:
            print(f"[WARNING] No valid adjacency for subject {i+1}.")
            continue
        edge_index = np.array(np.nonzero(A))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(gnn_list[i], dtype=torch.float)
        if x.shape[0] != A.shape[0]:
            print(f"[ERROR] Subject {i+1}: mismatch in node count.")
            continue
        y_val = torch.tensor([y[i]], dtype=torch.long)
        data_obj = Data(x=x, edge_index=edge_index, y=y_val)
        pyg_data_list.append(data_obj)
    return pyg_data_list

#############################################
# 5) MLP Classification (Partial-Fit) - PyTorch
#############################################
def train_mlp_tb(X_train, y_train, X_val, y_val, epochs=200, log_dir=None):
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import log_loss

    if log_dir is None:
        log_dir = "logs/mlp_fold"
    writer = SummaryWriter(log_dir=log_dir)

    mlp = MLPClassifier(
        hidden_layer_sizes=(26,14),
        activation='logistic',
        solver='adam',
        max_iter=1,
        alpha=0.275,
        warm_start=True,
        random_state=5
    )
    classes = np.unique(y_train)
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, epochs+1):
        mlp.partial_fit(X_train, y_train, classes=classes)

        y_train_proba = mlp.predict_proba(X_train)
        train_loss = log_loss(y_train, y_train_proba)

        y_val_proba = mlp.predict_proba(X_val)
        val_loss = log_loss(y_val, y_val_proba)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        writer.add_scalar("MLP/Train_Loss", train_loss, epoch)
        writer.add_scalar("MLP/Val_Loss", val_loss, epoch)

    writer.close()
    model_path = os.path.join(MODELS_DIR, "mlp_model.pkl")
    import joblib
    joblib.dump(mlp, model_path)
    print(f"[INFO] Saved MLP model checkpoint to {model_path}")
    return mlp, train_loss_history, val_loss_history

#############################################
# 6) EXTENDED QSUP MODEL (PyTorch-based)
#############################################
class ExtendedQSUP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_wavefunctions=3,
                 partial_norm=1.5, phase_per_dim=False, self_modulation_steps=2, topk=8):
        """
        Extended QSUP Model:
          - Uses multiple wave guesses.
          - Applies ArcBell activation, partial normalization, and dynamic phase rotation.
          - Computes interference weights via cosine similarity and aggregates the waves.
          - Optionally applies gating/self-modulation.
          - "Measures" the superposition by computing squared magnitude and normalizes it.
          - Final classification is performed on the normalized measurement.
        
        Hyperparameters are set to the same as your previous QSUP:
          num_wavefunctions=3, partial_norm=1.5, phase_per_dim=False,
          self_modulation_steps=2, topk=8.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_wavefunctions = num_wavefunctions
        self.partial_norm = partial_norm
        self.phase_per_dim = phase_per_dim
        self.self_modulation_steps = self_modulation_steps
        self.topk = topk

        # Sub-wavefunction networks
        self.wavefunction_nets = nn.ModuleList([
            nn.Linear(input_dim, 2 * hidden_dim) for _ in range(num_wavefunctions)
        ])
        # Phase parameters
        if phase_per_dim:
            self.phases = nn.Parameter(torch.zeros(num_wavefunctions, hidden_dim))
        else:
            self.phases = nn.Parameter(torch.zeros(num_wavefunctions, 1))
        
        # Optional gating network
        if self_modulation_steps > 0:
            self.gating_net = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.gating_net = None
        
        # Final classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.constant_(self.classifier.bias, 0.0)  # Initialize all biases to zero
        self.classifier.bias.data[1] = 0.75            # Set bias for Alzheimer’s to 0.5
    
    def forward(self, x):
        batch_size = x.size(0)
        eps = 1e-8

        # Process each wave guess and stack results
        wave_r_list = []
        wave_i_list = []
        for s in range(self.num_wavefunctions):
            out = self.wavefunction_nets[s](x)  # shape: (batch, 2*hidden_dim)
            out = torch.exp(-out * out)  # ArcBell activation
            alpha = out[:, :self.hidden_dim]
            beta  = out[:, self.hidden_dim:]
            norm_sq = torch.sum(alpha**2 + beta**2, dim=1, keepdim=True) + eps
            factor = torch.sqrt((self.partial_norm**2) / norm_sq)
            alpha = alpha * factor
            beta  = beta * factor

            if self.phase_per_dim:
                phase = self.phases[s].unsqueeze(0)  # shape: (1, hidden_dim)
            else:
                phase = self.phases[s]
            wave_r = alpha * torch.cos(phase) - beta * torch.sin(phase)
            wave_i = alpha * torch.sin(phase) + beta * torch.cos(phase)
            wave_r_list.append(wave_r)
            wave_i_list.append(wave_i)
        
        # Stack along new dimension: (batch, num_wavefunctions, hidden_dim)
        real_stack = torch.stack(wave_r_list, dim=1)
        imag_stack = torch.stack(wave_i_list, dim=1)
        
        # Compute mean real vector
        mean_real = torch.mean(real_stack, dim=1)  # (batch, hidden_dim)
        mean_norm = torch.sqrt(torch.sum(mean_real**2, dim=1, keepdim=True)) + eps  # (batch, 1)
        mean_norm = mean_norm.unsqueeze(1)  # (batch, 1, 1)
        wave_norms = torch.sqrt(torch.sum(real_stack**2, dim=2, keepdim=True)) + eps  # (batch, num_wavefunctions, 1)
        dot_prod = torch.sum(real_stack * mean_real.unsqueeze(1), dim=2, keepdim=True)  # (batch, num_wavefunctions, 1)
        cosine_sim = dot_prod / (wave_norms * mean_norm)  # (batch, num_wavefunctions, 1)
        cosine_sim = cosine_sim.squeeze(2)  # (batch, num_wavefunctions)
        interference_weights = F.softmax(cosine_sim, dim=1).unsqueeze(2)  # (batch, num_wavefunctions, 1)
        
        # Form superposition via weighted sum
        sup_real = torch.sum(real_stack * interference_weights, dim=1)  # (batch, hidden_dim)
        sup_imag = torch.sum(imag_stack * interference_weights, dim=1)  # (batch, hidden_dim)
        
        # Optional gating/self-modulation
        if self.self_modulation_steps > 0 and self.gating_net is not None:
            for _ in range(self.self_modulation_steps):
                mag = torch.sqrt(sup_real**2 + sup_imag**2 + eps)
                gate = torch.sigmoid(self.gating_net(mag))
                sup_real = sup_real * gate
                sup_imag = sup_imag * gate
        
        # Measurement: squared magnitude
        mag_sq = sup_real**2 + sup_imag**2  # (batch, hidden_dim)
        if self.topk > 0 and self.topk < self.hidden_dim:
            values, indices = torch.topk(mag_sq, k=self.topk, dim=1)
            mask = torch.zeros_like(mag_sq).scatter_(1, indices, 1.0)
            masked = mag_sq * mask
            sums = torch.sum(masked, dim=1, keepdim=True) + eps
            probs = masked / sums
        else:
            sums = torch.sum(mag_sq, dim=1, keepdim=True) + eps
            probs = mag_sq / sums
        
        # Final classification
        logits = self.classifier(probs)
        return logits

#############################################
# 7) TRAINING FUNCTION FOR EXTENDED QSUP (PyTorch)
#############################################
def train_extended_qsup_tb(X_train, y_train, X_val, y_val,
                           input_dim, hidden_dim, num_classes,
                           num_wavefunctions=3, partial_norm=1.5,
                           phase_per_dim=False, self_modulation_steps=2, topk=8,
                           epochs=150, log_dir="logs/qsup_fold", device="cpu"):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    
    # Convert data to tensors on device
    X_train_t = torch.tensor(X_train, dtype=torch.float, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    
    model = ExtendedQSUP(input_dim, hidden_dim, num_classes,
                         num_wavefunctions=num_wavefunctions,
                         partial_norm=partial_norm,
                         phase_per_dim=phase_per_dim,
                         self_modulation_steps=self_modulation_steps,
                         topk=topk).to(device)
    
    optimizer = TorchAdam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            train_proba = F.softmax(model(X_train_t), dim=1).cpu().numpy()
            train_loss = log_loss(y_train, train_proba)
            val_logits = model(X_val_t)
            val_proba = F.softmax(val_logits, dim=1).cpu().numpy()
            val_loss = log_loss(y_val, val_proba)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        writer.add_scalar("ExtendedQSUP/Train_Loss", train_loss, epoch)
        writer.add_scalar("ExtendedQSUP/Val_Loss", val_loss, epoch)
        
    writer.close()
    return model, train_loss_history, val_loss_history

#############################################
# 8) CROSS-VALIDATION FOR MLP (PyTorch)
#############################################
def cv_classification_MLP(CCV, y):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5)
    all_y_true = []
    all_y_pred = []
    fold_idx = 1

    for train_idx, test_idx in skf.split(CCV, y):
        X_train, X_test = CCV[train_idx], CCV[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        smote = SMOTE(random_state=5)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train_res = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)

        log_dir = os.path.join(LOG_DIR, f"mlp_fold_{fold_idx}")
        mlp, train_loss_hist, val_loss_hist = train_mlp_tb(
            X_train_res, y_train_res, X_test_scaled, y_test,
            epochs=200, log_dir=log_dir
        )

        y_pred = mlp.predict(X_test_scaled)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        y_proba = mlp.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, marker='o', label="MLP")
        plt.plot([0,1],[0,1],'k--')
        plt.title(f"MLP ROC Fold {fold_idx}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"mlp_roc_fold_{fold_idx}.png"))
        plt.close()

        plt.figure()
        plt.plot(train_loss_hist, marker='o', label="Train Loss")
        plt.plot(val_loss_hist,   marker='x', label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Log Loss")
        plt.title(f"MLP Train/Val Loss - Fold {fold_idx}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"mlp_loss_fold_{fold_idx}.png"))
        plt.close()

        print(f"[Fold {fold_idx}] MLP Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        fold_idx += 1

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    print("\n--- MLP Overall Classification ---")
    print(confusion_matrix(all_y_true, all_y_pred))
    print(classification_report(all_y_true, all_y_pred, target_names=['Control','Alzheimer']))
    print(f"Overall MLP Accuracy: {overall_acc:.4f}")

#############################################
# 9) CROSS-VALIDATION FOR EXTENDED QSUP (PyTorch)
#############################################
def cv_classification_ExtendedQSUP(CCV, y):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5)
    all_y_true = []
    all_y_pred = []
    fold_idx = 1
    input_dim = CCV.shape[1]
    n_classes = len(np.unique(y))
    for train_idx, test_idx in skf.split(CCV, y):
        X_train, X_test = CCV[train_idx], CCV[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        smote = SMOTE(random_state=5)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        scaler = StandardScaler()
        X_train_res = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)
        
        log_dir = os.path.join(LOG_DIR, f"qsup_fold_{fold_idx}")
        qsup_model, train_loss_hist, val_loss_hist = train_extended_qsup_tb(
            X_train_res, y_train_res, X_test_scaled, y_test,
            input_dim=input_dim,
            hidden_dim=32,
            num_classes=n_classes,
            num_wavefunctions=6,
            partial_norm=1.5,
            phase_per_dim=True,
            self_modulation_steps=2,
            topk=8,
            epochs=200,
            log_dir=log_dir,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        qsup_model.eval()
        with torch.no_grad():
            test_logits = qsup_model(X_test_tensor)
        test_preds = test_logits.argmax(dim=1).cpu().numpy()
        all_y_true.extend(y_test)
        all_y_pred.extend(test_preds)
        print(f"[Fold {fold_idx}] Extended QSUP Accuracy: {accuracy_score(y_test, test_preds):.4f}")
        fold_idx += 1
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    print("\n--- Extended QSUP Overall Classification ---")
    print(confusion_matrix(all_y_true, all_y_pred))
    print(classification_report(all_y_true, all_y_pred, target_names=['Control','Alzheimer']))
    print(f"Overall Extended QSUP Accuracy: {overall_acc:.4f}")

#############################################
# 10) MAIN (Combines GCN + MLP & Extended QSUP)
#############################################
def main():
    start_time = time.time()
    # 1) Load features
    X_handcrafted, X_gnn, y, ch_names_list, subj_ids = load_saved_features()
    print(f"[INFO] Handcrafted shape: {X_handcrafted.shape}")
    print(f"[INFO] {len(X_gnn)} GNN feature matrices loaded")

    # 2) Build PyG dataset for GCN
    pyg_dataset = create_pyg_dataset(X_gnn, y, ch_names_list)
    if len(pyg_dataset) == 0:
        raise ValueError("No valid PyG dataset could be constructed!")
    indices = np.arange(len(pyg_dataset))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.15, random_state=5,
        stratify=np.array([d.y.item() for d in pyg_dataset])
    )
    train_dataset = [pyg_dataset[i] for i in train_idx]
    val_dataset   = [pyg_dataset[i] for i in val_idx]
    train_loader = PyGDataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader   = PyGDataLoader(val_dataset,   batch_size=4, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = X_gnn[0].shape[1]
    hidden_channels = 32
    gcn_num_classes = 2
    gcn_model = GCNNet(in_channels, hidden_channels, gcn_num_classes).to(device)
    optimizer = TorchAdam(gcn_model.parameters(), lr=10**-2.25)
    epochs_gcn = 100

    # 3) Train GCN
    print("[GCN] Training embeddings ...")
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
        avg_loss = total_loss / len(train_loader.dataset)
        if epoch % 5 == 0:
            print(f"GCN Epoch {epoch}/{epochs_gcn} - Train Loss: {avg_loss:.4f}")

    # 4) Extract embeddings
    print("[GCN] Extracting embeddings for entire dataset ...")
    full_loader = PyGDataLoader(pyg_dataset, batch_size=4, shuffle=False)
    gcn_model.eval()
    embeddings = []
    with torch.no_grad():
        for data in full_loader:
            data = data.to(device)
            emb = gcn_model.embed(data.x, data.edge_index, data.batch)
            embeddings.append(emb.cpu().numpy())
    embeddings = np.vstack(embeddings)
    print(f"[INFO] GCN embeddings shape: {embeddings.shape}")

    # 5) Combine => CCV
    if X_handcrafted.shape[0] != embeddings.shape[0]:
        raise ValueError("Mismatch between handcrafted features and GCN embeddings!")
    CCV = np.hstack((X_handcrafted, embeddings))
    print(f"[INFO] Final CCV shape: {CCV.shape}")

    # 6) 3-fold classification with MLP
    print("\n=== 3-Fold Classification with MLP ===")
    cv_classification_MLP(CCV, y)

    # 7) 3-fold classification with Extended QSUP (PyTorch)
    print("\n=== 3-Fold Classification with Extended QSUP (PyTorch) ===")
    cv_classification_ExtendedQSUP(CCV, y)

    end_time = time.time()
    print(f"[DONE] total runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
