#!/usr/bin/env python
"""
Downstream Training Script Using Saved Features (EEG):
Now includes both:
  1) TorchMLP (PyTorch MLP with modern training)
  2) ExtendedQSUP v2 (enhanced quantum-inspired model with SWA)

Key Changes (v2):
  - ExtendedQSUP: LayerNorm, learnable temperature, leaky residual gating,
    2-layer classifier head, spectral norm on wavefunction nets.
  - TorchMLP: full PyTorch replacement with BatchNorm, GELU, CosineAnnealingWarmRestarts.
  - Mixup augmentation for both models.
  - Stochastic Weight Averaging (SWA) for QSUP.
  - Rich visualization: confusion matrices, training comparison, QSUP internals,
    gradient feature importance, t-SNE, ROC comparison.

Steps:
  1) Load previously saved EEG features.
  2) Train a GCN to get embeddings.
  3) Combine GCN embeddings + handcrafted features => CCV.
  4) Cross-validate with both TorchMLP and ExtendedQSUP, logging losses & generating plots.
  5) Save best MLP as mlp_model.pth (PyTorch) + mlp_model.pkl (sklearn fallback),
     best QSUP as qsup_model.pth (state_dict), GCN as gcn_model.pth, and scalers.
"""

#############################################
# CONFIGURATION & IMPORTS
#############################################
import os
import glob
import json
import copy
import time
import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report, log_loss
)
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE, ADASYN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations
from torch.optim import Adam as TorchAdam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR
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
for d in [PLOTS_DIR, LOG_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARTICIPANTS_FILE_DS004504 = os.path.join(BASE_DIR, "ds004504", "participants.tsv")
PARTICIPANTS_FILE_DS003800 = os.path.join(BASE_DIR, "ds003800", "participants.tsv")
PARTICIPANTS_FILE_DS006036 = os.path.join(BASE_DIR, "ds006036", "participants.tsv")

#############################################
# 1) LOAD PARTICIPANT LABELS
#############################################
def load_participant_labels(ds004504_file, ds003800_file):
    """
    Binary labelling (FTD counted as Alzheimer):
      0 = Healthy Control (C)
      1 = Alzheimer's Disease or FTD (A / F / all ds003800)
    Datasets:
      ds004504:  88 subjects (sub-001..sub-088) — A/F/C groups
      ds003800:  13 subjects (sub-01..sub-13)   — all AD
      ds006036:  88 subjects (zsub001..zsub088) — same cohort as ds004504, photic EEG
      Zenodo:     2 samples  (S055=HC, i108=AD)
    """
    label_dict = {}
    group_map = {"A": 1, "C": 0, "F": 1}   # FTD counted as AD

    # ds004504
    df = pd.read_csv(ds004504_file, sep="\t")
    df = df[df['Group'].isin(group_map.keys())]
    label_dict.update(df.set_index("participant_id")["Group"].map(group_map).to_dict())

    # ds003800 — all AD, processed with "bsub" prefix (bsub001..bsub013)
    df = pd.read_csv(ds003800_file, sep="\t")
    for pid in df["participant_id"]:
        num = int(pid.split('-')[1])
        label_dict[f"bsub{num:03d}"] = 1

    # ds006036 — same cohort, photic EEG; subject IDs prefixed with "z" (zsub001 etc.)
    if os.path.exists(PARTICIPANTS_FILE_DS006036):
        df = pd.read_csv(PARTICIPANTS_FILE_DS006036, sep="\t")
        df = df[df['Group'].isin(group_map.keys())]
        for pid, grp in zip(df["participant_id"], df["Group"]):
            # sub-001 → zsub001
            zsub_id = "z" + pid.replace("-", "")
            label_dict[zsub_id] = group_map[grp]

    # Zenodo samples (HC=0, AD=1)
    label_dict["S055"] = 0   # HC
    label_dict["i108"] = 1   # AD

    return label_dict

participant_labels = load_participant_labels(
    PARTICIPANTS_FILE_DS004504,
    PARTICIPANTS_FILE_DS003800
)

#############################################
# 1b) DOMAIN / SOURCE MAPPING
#############################################
# Source IDs:  0=ds004504  1=ds003800  2=ds006036  3=zenodo/other
SOURCE_NAMES = ['ds004504', 'ds003800', 'ds006036', 'zenodo']
N_SOURCES = len(SOURCE_NAMES)

def get_source_id(subj_id: str) -> int:
    if subj_id.startswith('bsub'):
        return 1
    if subj_id.startswith('zsub'):
        return 2
    if subj_id.startswith('sub'):
        return 0
    return 3   # zenodo / other uppercase IDs


class SourceAwareScaler:
    """
    Agentic Context Scaler — fits one StandardScaler per source domain
    on the training fold, then applies the domain-matched scaler at
    inference.  Falls back to a global scaler for unseen source IDs.

    This corrects for systematic EEG amplitude / power offsets that
    arise from different recording hardware and paradigms across datasets,
    without ever leaking test-set statistics.
    """
    def __init__(self):
        self.scalers: dict = {}
        self.global_scaler = StandardScaler()

    def fit(self, X: np.ndarray, source_ids: np.ndarray) -> 'SourceAwareScaler':
        self.global_scaler.fit(X)
        for sid in np.unique(source_ids):
            mask = source_ids == sid
            if mask.sum() >= 2:
                self.scalers[int(sid)] = StandardScaler().fit(X[mask])
        return self

    def transform(self, X: np.ndarray, source_ids: np.ndarray) -> np.ndarray:
        out = np.empty_like(X, dtype=np.float32)
        for sid in np.unique(source_ids):
            mask = source_ids == sid
            scaler = self.scalers.get(int(sid), self.global_scaler)
            out[mask] = scaler.transform(X[mask]).astype(np.float32)
        return out

    def fit_transform(self, X: np.ndarray, source_ids: np.ndarray) -> np.ndarray:
        return self.fit(X, source_ids).transform(X, source_ids)


#############################################
# 2) LOAD SAVED FEATURE FILES
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
    source_ids = np.array([get_source_id(s) for s in subj_ids], dtype=np.int32)
    return X_handcrafted, X_gnn, labels, ch_names_list, subj_ids, source_ids

#############################################
# 2b) EXTRA FEATURES FROM GNN BAND-POWER MATRIX
#############################################
def extract_extra_features_from_gnn(gnn_feat: np.ndarray) -> np.ndarray:
    """
    Extract 42 engineered features from an (N_ch, 9) band-power matrix.
    Channel-count-agnostic: all features are aggregated across channels,
    so the output is always 42-dim whether N_ch=4 (live inference) or 19 (training).

    Band column order (matches process_server.py FREQUENCY_BANDS):
      0:Delta(0.5-4), 1:Theta1(4-6), 2:Theta2(6-8), 3:Alpha1(8-10), 4:Alpha2(10-12),
      5:Beta1(12-20), 6:Beta2(20-30), 7:Gamma1(30-40), 8:Gamma2(40-50)

    Returns 42-dim float32 vector:
      - per-band mean across channels          (9)
      - per-band std across channels           (9)
      - log1p of per-band mean                 (9)
      - normalized band powers (band/total)    (9)
      - 6 clinically-motivated band ratios     (6)
    """
    eps = 1e-12
    band_means = np.mean(gnn_feat, axis=0).astype(np.float64)   # (9,)
    band_stds  = np.std(gnn_feat,  axis=0).astype(np.float64)   # (9,)
    log_means  = np.log1p(np.clip(band_means, 0, None))          # (9,)

    total_mean = float(np.sum(band_means)) + eps
    norm_bands = band_means / total_mean                          # (9,)

    delta = band_means[0]
    theta = band_means[1] + band_means[2]
    alpha = band_means[3] + band_means[4]
    beta  = band_means[5] + band_means[6]
    gamma = band_means[7] + band_means[8]

    ratios = np.array([
        theta / (alpha + eps),                 # theta/alpha — elevated in AD
        alpha / (beta  + eps),                 # alpha/beta
        delta / (alpha + eps),                 # delta/alpha — elevated in AD
        (alpha + theta) / (total_mean + eps),  # slow-band fraction
        delta / (total_mean + eps),            # delta fraction
        gamma / (beta  + eps),                 # high-freq ratio
    ], dtype=np.float64)

    return np.concatenate([
        band_means, band_stds, log_means, norm_bands, ratios
    ]).astype(np.float32)   # 9+9+9+9+6 = 42

#############################################
# 3) GCN MODEL & DATASET (PyTorch)
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
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        logits = self.lin(x)
        return logits

    def embed(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
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
    """Returns (pyg_data_list, valid_indices) — valid_indices are the original positions kept."""
    pyg_data_list = []
    valid_indices = []
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
        valid_indices.append(i)
    return pyg_data_list, valid_indices

#############################################
# HELPER: Compute class weights for CE loss
#############################################
def compute_class_weights(y, device):
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    classes, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts.astype(np.float32)
    weights = weights / weights.sum() * len(classes)  # normalize so mean weight = 1
    return torch.tensor(weights, dtype=torch.float, device=device)

# Per-source training loss weights — higher = more trusted signal
# ds004504 (resting EEG, validated AD cohort) is the gold standard
SOURCE_LOSS_WEIGHTS = {
    0: 1.5,   # ds004504  — resting EEG, gold standard
    1: 1.0,   # ds003800  — resting EEG, same paradigm
    2: 0.6,   # ds006036  — photostimulation EEG, different paradigm
    3: 0.4,   # zenodo    — HD-EEG, different hardware+paradigm
}

def get_sample_weights(source_ids: np.ndarray) -> np.ndarray:
    """Return per-sample training loss weights based on source domain."""
    w = np.array([SOURCE_LOSS_WEIGHTS[int(s)] for s in source_ids], dtype=np.float32)
    return w / w.mean()   # normalise so mean weight = 1 (keeps learning-rate scale stable)

#############################################
# MIXUP AUGMENTATION
#############################################
# DOMAIN-STRATIFIED SMOTE
#############################################
def domain_stratified_smote(X: np.ndarray, y: np.ndarray,
                             src: np.ndarray, random_state: int = 5):
    """
    Apply SMOTE within each source domain separately, then concatenate.
    Avoids creating synthetic samples that interpolate across EEG paradigms
    (e.g. resting-state ↔ photostimulation).

    Domains with < 6 samples of either class fall back to random oversampling
    (SMOTE needs at least k+1=6 neighbours by default).
    Returns X_res, y_res, src_res.
    """
    from imblearn.over_sampling import RandomOverSampler
    X_parts, y_parts, src_parts = [], [], []
    unique_srcs = np.unique(src)
    max_class_count = 0
    # First pass: find the max minority count after per-domain resampling
    for sid in unique_srcs:
        mask = src == sid
        counts = np.bincount(y[mask])
        if len(counts) == 2:
            max_class_count = max(max_class_count, max(counts))

    for sid in unique_srcs:
        mask = src == sid
        Xs, ys = X[mask], y[mask]
        src_vec = src[mask]
        counts = np.bincount(ys)
        if len(counts) < 2 or min(counts) < 2:
            # Only one class or too few — keep as-is
            X_parts.append(Xs); y_parts.append(ys); src_parts.append(src_vec)
            continue
        min_count = min(counts)
        if min_count >= 6:
            resampler = SMOTE(random_state=random_state)
        else:
            resampler = RandomOverSampler(random_state=random_state)
        try:
            Xr, yr = resampler.fit_resample(Xs, ys)
            # Assign source ID to new samples (inherited from same domain)
            src_r = np.full(len(yr), sid, dtype=src.dtype)
            X_parts.append(Xr); y_parts.append(yr); src_parts.append(src_r)
        except Exception:
            X_parts.append(Xs); y_parts.append(ys); src_parts.append(src_vec)

    return (np.vstack(X_parts),
            np.concatenate(y_parts),
            np.concatenate(src_parts))


#############################################
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to a batch."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

#############################################
# 4) TORCH MLP MODEL (PyTorch replacement)
#############################################
class TorchMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_torch(X_train, y_train, X_val, y_val, input_dim, num_classes,
                    epochs=200, patience=30, log_dir=None, device="cpu",
                    sample_weights=None):
    """Train TorchMLP with AdamW, CosineAnnealingWarmRestarts, label smoothing, mixup.
    sample_weights: per-sample float array (same len as X_train) for source-weighted loss."""
    if log_dir is None:
        log_dir = "logs/mlp_fold"
    writer = SummaryWriter(log_dir=log_dir)

    X_train_t = torch.tensor(X_train, dtype=torch.float, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float, device=device)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long, device=device)
    sw_t = (torch.tensor(sample_weights, dtype=torch.float, device=device)
            if sample_weights is not None else None)

    model = TorchMLP(input_dim, num_classes).to(device)

    class_weights = compute_class_weights(y_train, device)
    # Use reduction='none' when we have per-sample source weights
    loss_fn_none = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05, reduction='none')
    loss_fn      = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    train_loss_history = []
    val_loss_history = []

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Mixup augmentation
        mixed_x, y_a, y_b, lam = mixup_data(X_train_t, y_train_t, alpha=0.2)
        logits = model(mixed_x)
        if sw_t is not None:
            # Source-weighted loss: weight per-sample CE before averaging
            loss_a = loss_fn_none(logits, y_a) * sw_t
            loss_b = loss_fn_none(logits, y_b) * sw_t
            loss = (lam * loss_a + (1 - lam) * loss_b).mean()
        else:
            loss = mixup_criterion(loss_fn, logits, y_a, y_b, lam)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            train_proba = F.softmax(model(X_train_t), dim=1).cpu().numpy()
            train_loss = log_loss(y_train, train_proba)
            val_proba = F.softmax(model(X_val_t), dim=1).cpu().numpy()
            val_loss = log_loss(y_val, val_proba)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        writer.add_scalar("MLP/Train_Loss", train_loss, epoch)
        writer.add_scalar("MLP/Val_Loss", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            print(f"  [MLP] Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    writer.close()
    return model, train_loss_history, val_loss_history


def train_sklearn_mlp_fallback(X_train, y_train, X_val, y_val, epochs=200, patience=30):
    """Train a fallback sklearn MLP for backward compatibility."""
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='logistic',
        solver='adam',
        max_iter=1,
        alpha=0.275,
        warm_start=True,
        random_state=5
    )
    classes = np.unique(y_train)
    best_val_loss = float('inf')
    best_epoch = 0
    best_coefs = None
    best_intercepts = None

    for epoch in range(1, epochs + 1):
        mlp.partial_fit(X_train, y_train, classes=classes)
        y_val_proba = mlp.predict_proba(X_val)
        val_loss = log_loss(y_val, y_val_proba)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_coefs = [c.copy() for c in mlp.coefs_]
            best_intercepts = [b.copy() for b in mlp.intercepts_]
        elif epoch - best_epoch >= patience:
            break

    if best_coefs is not None:
        mlp.coefs_ = best_coefs
        mlp.intercepts_ = best_intercepts

    return mlp

#############################################
# 5) EXTENDED QSUP v2 MODEL (PyTorch)
#############################################
class ExtendedQSUP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes,
                 num_wavefunctions=3, partial_norm=1.5,
                 phase_per_dim=False, self_modulation_steps=2,
                 topk=8):
        """
        Extended QSUP v2 Model:
          - Uses multiple wave guesses with spectral-normed Linear layers.
          - LayerNorm on wavefunction outputs before ArcBell activation.
          - Learnable temperature in interference softmax.
          - Leaky residual in self-modulation gating.
          - 2-layer classifier head with GELU.
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

        # Auto-determined input projection:
        #   proj_dim = nearest multiple of 8 ≥ max(hidden_dim, input_dim//2)
        #   This compresses high-dim CCVs to a sensible scale for the wavefunction
        #   nets without requiring a hard-coded target dimension.
        proj_dim = ((max(hidden_dim, input_dim // 2) + 7) // 8) * 8
        self.use_proj = proj_dim < input_dim
        if self.use_proj:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        wf_input_dim = proj_dim if self.use_proj else input_dim

        # Wavefunction nets with spectral norm
        self.wavefunction_nets = nn.ModuleList()
        for _ in range(num_wavefunctions):
            layer = nn.Linear(wf_input_dim, 2 * hidden_dim)
            layer = parametrizations.spectral_norm(layer)
            self.wavefunction_nets.append(layer)

        # LayerNorm applied to each wavefunction net output before ArcBell
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(2 * hidden_dim) for _ in range(num_wavefunctions)
        ])

        # Learnable temperature for interference softmax
        self.temperature = nn.Parameter(torch.ones(1))

        if phase_per_dim:
            self.phases = nn.Parameter(torch.zeros(num_wavefunctions, hidden_dim))
        else:
            self.phases = nn.Parameter(torch.zeros(num_wavefunctions, 1))

        if self_modulation_steps > 0:
            self.gating_net = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.gating_net = None

        self.dropout = nn.Dropout(0.1)

        # 2-layer classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )
        # Initialize final layer bias for AD
        nn.init.constant_(self.classifier[-1].bias, 0.0)
        self.classifier[-1].bias.data[1] = 0.75

    def forward(self, x):
        batch_size = x.size(0)
        eps = 1e-8

        if self.use_proj:
            x = self.input_proj(x)   # (batch, proj_dim)

        wave_r_list = []
        wave_i_list = []
        for s in range(self.num_wavefunctions):
            out = self.wavefunction_nets[s](x)    # shape: (batch, 2*hidden_dim)
            out = self.layer_norms[s](out)         # LayerNorm before ArcBell
            out = torch.exp(-out * out)            # ArcBell
            alpha = out[:, :self.hidden_dim]
            beta  = out[:, self.hidden_dim:]
            norm_sq = torch.sum(alpha**2 + beta**2, dim=1, keepdim=True) + eps
            factor = torch.sqrt((self.partial_norm**2) / norm_sq)
            alpha = alpha * factor
            beta  = beta * factor

            if self.phase_per_dim:
                phase = self.phases[s].unsqueeze(0)
            else:
                phase = self.phases[s]
            wave_r = alpha * torch.cos(phase) - beta * torch.sin(phase)
            wave_i = alpha * torch.sin(phase) + beta * torch.cos(phase)
            wave_r_list.append(wave_r)
            wave_i_list.append(wave_i)

        real_stack = torch.stack(wave_r_list, dim=1)   # (batch, num_wavefunctions, hidden_dim)
        imag_stack = torch.stack(wave_i_list, dim=1)   # (batch, num_wavefunctions, hidden_dim)

        mean_real = torch.mean(real_stack, dim=1)      # (batch, hidden_dim)
        mean_norm = torch.sqrt(torch.sum(mean_real**2, dim=1, keepdim=True)) + eps
        mean_norm = mean_norm.unsqueeze(1)
        wave_norms = torch.sqrt(torch.sum(real_stack**2, dim=2, keepdim=True)) + eps
        dot_prod = torch.sum(real_stack * mean_real.unsqueeze(1), dim=2, keepdim=True)
        cosine_sim = dot_prod / (wave_norms * mean_norm)
        cosine_sim = cosine_sim.squeeze(2)
        # Learnable temperature in interference softmax
        interference_weights = F.softmax(cosine_sim / self.temperature, dim=1).unsqueeze(2)

        sup_real = torch.sum(real_stack * interference_weights, dim=1)
        sup_imag = torch.sum(imag_stack * interference_weights, dim=1)

        if self.self_modulation_steps > 0 and self.gating_net is not None:
            for _ in range(self.self_modulation_steps):
                mag = torch.sqrt(sup_real**2 + sup_imag**2 + eps)
                gate = torch.sigmoid(self.gating_net(mag))
                # Leaky residual connection in gating
                sup_real = sup_real * gate + sup_real * 0.1
                sup_imag = sup_imag * gate + sup_imag * 0.1

        mag_sq = sup_real**2 + sup_imag**2
        if self.topk > 0 and self.topk < self.hidden_dim:
            vals, inds = torch.topk(mag_sq, self.topk, dim=1)
            mask = torch.zeros_like(mag_sq).scatter_(1, inds, 1.0)
            masked = mag_sq * mask
            sums = torch.sum(masked, dim=1, keepdim=True) + eps
            probs = masked / sums
        else:
            sums = torch.sum(mag_sq, dim=1, keepdim=True) + eps
            probs = mag_sq / sums

        probs = self.dropout(probs)
        logits = self.classifier(probs)
        return logits

    def forward_with_internals(self, x):
        """Forward pass that also returns internal states for visualization."""
        batch_size = x.size(0)
        eps = 1e-8

        if self.use_proj:
            x = self.input_proj(x)   # (batch, proj_dim)

        wave_r_list = []
        wave_i_list = []
        wave_magnitudes = []
        for s in range(self.num_wavefunctions):
            out = self.wavefunction_nets[s](x)
            out = self.layer_norms[s](out)
            out = torch.exp(-out * out)
            alpha = out[:, :self.hidden_dim]
            beta  = out[:, self.hidden_dim:]
            norm_sq = torch.sum(alpha**2 + beta**2, dim=1, keepdim=True) + eps
            factor = torch.sqrt((self.partial_norm**2) / norm_sq)
            alpha = alpha * factor
            beta  = beta * factor

            mag = torch.sqrt(alpha**2 + beta**2)
            wave_magnitudes.append(mag)

            if self.phase_per_dim:
                phase = self.phases[s].unsqueeze(0)
            else:
                phase = self.phases[s]
            wave_r = alpha * torch.cos(phase) - beta * torch.sin(phase)
            wave_i = alpha * torch.sin(phase) + beta * torch.cos(phase)
            wave_r_list.append(wave_r)
            wave_i_list.append(wave_i)

        real_stack = torch.stack(wave_r_list, dim=1)
        imag_stack = torch.stack(wave_i_list, dim=1)

        mean_real = torch.mean(real_stack, dim=1)
        mean_norm = torch.sqrt(torch.sum(mean_real**2, dim=1, keepdim=True)) + eps
        mean_norm = mean_norm.unsqueeze(1)
        wave_norms = torch.sqrt(torch.sum(real_stack**2, dim=2, keepdim=True)) + eps
        dot_prod = torch.sum(real_stack * mean_real.unsqueeze(1), dim=2, keepdim=True)
        cosine_sim = dot_prod / (wave_norms * mean_norm)
        cosine_sim = cosine_sim.squeeze(2)
        interference_weights = F.softmax(cosine_sim / self.temperature, dim=1).unsqueeze(2)

        sup_real = torch.sum(real_stack * interference_weights, dim=1)
        sup_imag = torch.sum(imag_stack * interference_weights, dim=1)

        if self.self_modulation_steps > 0 and self.gating_net is not None:
            for _ in range(self.self_modulation_steps):
                mag = torch.sqrt(sup_real**2 + sup_imag**2 + eps)
                gate = torch.sigmoid(self.gating_net(mag))
                sup_real = sup_real * gate + sup_real * 0.1
                sup_imag = sup_imag * gate + sup_imag * 0.1

        mag_sq = sup_real**2 + sup_imag**2
        if self.topk > 0 and self.topk < self.hidden_dim:
            vals, inds = torch.topk(mag_sq, self.topk, dim=1)
            mask = torch.zeros_like(mag_sq).scatter_(1, inds, 1.0)
            masked = mag_sq * mask
            sums = torch.sum(masked, dim=1, keepdim=True) + eps
            probs = masked / sums
        else:
            sums = torch.sum(mag_sq, dim=1, keepdim=True) + eps
            probs = mag_sq / sums

        logits = self.classifier(probs)

        internals = {
            'wave_magnitudes': torch.stack(wave_magnitudes, dim=1),  # (batch, num_wf, hidden_dim)
            'interference_weights': interference_weights.squeeze(2),  # (batch, num_wf)
            'phases': self.phases.detach(),
            'probs': probs,
        }
        return logits, internals


def train_extended_qsup_tb(
    X_train, y_train, X_val, y_val,
    input_dim, hidden_dim, num_classes,
    num_wavefunctions=3, partial_norm=1.5,
    phase_per_dim=False, self_modulation_steps=2, topk=8,
    epochs=150, patience=25, log_dir="logs/qsup_fold", device="cpu",
    sample_weights=None):

    writer = SummaryWriter(log_dir=log_dir)

    X_train_t = torch.tensor(X_train, dtype=torch.float, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float, device=device)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long, device=device)
    sw_t = (torch.tensor(sample_weights, dtype=torch.float, device=device)
            if sample_weights is not None else None)

    model = ExtendedQSUP(
        input_dim, hidden_dim, num_classes,
        num_wavefunctions=num_wavefunctions,
        partial_norm=partial_norm,
        phase_per_dim=phase_per_dim,
        self_modulation_steps=self_modulation_steps,
        topk=topk
    ).to(device)

    # Class-weighted loss with mild label smoothing
    class_weights = compute_class_weights(y_train, device)
    loss_fn_none = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02, reduction='none')
    loss_fn      = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)

    optimizer = TorchAdam(model.parameters(), lr=5e-4, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_loss_history = []
    val_loss_history = []

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Mixup augmentation
        mixed_x, y_a, y_b, lam = mixup_data(X_train_t, y_train_t, alpha=0.2)
        logits = model(mixed_x)
        if sw_t is not None:
            loss_a = loss_fn_none(logits, y_a) * sw_t
            loss_b = loss_fn_none(logits, y_b) * sw_t
            loss = (lam * loss_a + (1 - lam) * loss_b).mean()
        else:
            loss = mixup_criterion(loss_fn, logits, y_a, y_b, lam)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
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

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            print(f"  [QSUP] Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Stochastic Weight Averaging (SWA) for 30 more epochs ---
    print("  [QSUP] Starting SWA phase (30 epochs)...")
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-5)

    for swa_epoch in range(1, 31):
        model.train()
        optimizer.zero_grad()

        mixed_x, y_a, y_b, lam = mixup_data(X_train_t, y_train_t, alpha=0.2)
        logits = model(mixed_x)
        loss = mixup_criterion(loss_fn, logits, y_a, y_b, lam)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        swa_scheduler.step()
        swa_model.update_parameters(model)

    # Copy SWA averaged params back to model
    # Since we do full-batch (no BN update needed for QSUP which has no BN),
    # we just copy the averaged parameters.
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in swa_model.state_dict().items()
         if k.startswith('module.')}
    )

    writer.close()
    return model, train_loss_history, val_loss_history

#############################################
# 6) CROSS-VALIDATION FOR MLP (PyTorch)
#############################################
def cv_classification_MLP(CCV, y, source_ids=None):
    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=5)
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fold_idx = 1

    best_val_auc = -1.0
    best_mlp_state = None
    best_mlp_scaler = None
    best_sklearn_mlp = None

    input_dim = CCV.shape[1]
    num_classes = len(np.unique(y))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mlp_histories = []
    fold_accuracies = []
    fold_aucs = []

    for train_idx, test_idx in skf.split(CCV, y):
        X_train, X_test = CCV[train_idx], CCV[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Global scaler — stable with small per-domain sample counts
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        smote = SMOTE(random_state=5)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

        log_dir = os.path.join(LOG_DIR, f"mlp_fold_{fold_idx}")

        # Train PyTorch MLP
        mlp_model, train_loss_hist, val_loss_hist = train_mlp_torch(
            X_train_res, y_train_res, X_test_scaled, y_test,
            input_dim=input_dim, num_classes=num_classes,
            epochs=200, patience=30, log_dir=log_dir, device=device
        )

        mlp_histories.append((train_loss_hist, val_loss_hist))

        # Evaluate on test set
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float, device=device)
        mlp_model.eval()
        with torch.no_grad():
            test_logits = mlp_model(X_test_tensor)
            test_proba = F.softmax(test_logits, dim=1).cpu().numpy()
        test_preds = test_logits.argmax(dim=1).cpu().numpy()

        all_y_true.extend(y_test)
        all_y_pred.extend(test_preds)
        all_y_proba.extend(test_proba.tolist())   # store full (n, num_classes) rows

        fold_acc = accuracy_score(y_test, test_preds)
        n_cls = len(np.unique(y_test))
        if n_cls > 1:
            fold_auc = roc_auc_score(y_test, test_proba[:, 1])
        else:
            fold_auc = 0.5
        fold_accuracies.append(fold_acc)
        fold_aucs.append(fold_auc)

        # Track best model by val AUC
        if fold_auc > best_val_auc:
            best_val_auc = fold_auc
            best_mlp_state = copy.deepcopy(mlp_model.state_dict())
            best_mlp_scaler = scaler

        # Also train sklearn fallback on same fold if this is best
        if fold_auc >= best_val_auc:
            best_sklearn_mlp = train_sklearn_mlp_fallback(
                X_train_res, y_train_res, X_test_scaled, y_test,
                epochs=200, patience=30
            )

        # ROC plot: one curve per class vs rest
        fig, ax = plt.subplots()
        for cls_i in range(test_proba.shape[1]):
            y_bin = (y_test == cls_i).astype(int)
            if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                fpr_i, tpr_i, _ = roc_curve(y_bin, test_proba[:, cls_i])
                auc_i = roc_auc_score(y_bin, test_proba[:, cls_i])
                lbl = ['Control', 'Alzheimer'][cls_i]
                ax.plot(fpr_i, tpr_i, label=f"{lbl} (AUC={auc_i:.2f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title(f"MLP ROC Fold {fold_idx} (macro AUC={fold_auc:.3f})")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=8)
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

        print(f"[Fold {fold_idx}] MLP Accuracy: {fold_acc:.4f}, AUC: {fold_auc:.4f}")
        fold_idx += 1

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    print("\n--- MLP Overall Classification ---")
    print(confusion_matrix(all_y_true, all_y_pred))
    print(classification_report(all_y_true, all_y_pred, target_names=['Control', 'Alzheimer']))
    print(f"Overall MLP Accuracy: {overall_acc:.4f}")

    # Save best PyTorch MLP model
    if best_mlp_state is not None:
        mlp_pth_path = os.path.join(MODELS_DIR, "mlp_model.pth")
        torch.save(best_mlp_state, mlp_pth_path)
        print(f"[INFO] Saved best MLP state_dict (AUC={best_val_auc:.4f}) to {mlp_pth_path}")

    # Save fallback sklearn MLP
    if best_sklearn_mlp is not None:
        mlp_pkl_path = os.path.join(MODELS_DIR, "mlp_model.pkl")
        joblib.dump(best_sklearn_mlp, mlp_pkl_path)
        print(f"[INFO] Saved fallback sklearn MLP to {mlp_pkl_path}")

    if best_mlp_scaler is not None:
        scaler_path = os.path.join(MODELS_DIR, "mlp_scaler.joblib")
        joblib.dump(best_mlp_scaler, scaler_path)
        print(f"[INFO] Saved MLP scaler to {scaler_path}")

    return (np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba),
            mlp_histories, fold_accuracies, fold_aucs)

#############################################
# 7) CROSS-VALIDATION FOR Extended QSUP
#############################################
def cv_classification_ExtendedQSUP(CCV, y, source_ids=None):
    """
    Trains ExtendedQSUP v2 with 5-fold cross validation (5-model ensemble per fold).
    Saves the best fold's model (highest val AUC) as "qsup_model.pth".
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fold_idx = 1

    input_dim = CCV.shape[1]
    n_classes = len(np.unique(y))

    best_val_auc = -1.0
    best_qsup_state = None
    best_qsup_scaler = None
    best_qsup_model = None
    # Store model config for saving
    qsup_config = dict(
        input_dim=input_dim, hidden_dim=48, num_classes=n_classes,
        num_wavefunctions=8, partial_norm=1.5,
        phase_per_dim=True, self_modulation_steps=3, topk=12
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    qsup_histories = []
    fold_accuracies = []
    fold_aucs = []

    for train_idx, test_idx in skf.split(CCV, y):
        X_train, X_test = CCV[train_idx], CCV[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Global scaler — stable with small per-domain sample counts
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        smote = SMOTE(random_state=5)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

        # Ensemble: train 5 models with different seeds, average predictions
        ENSEMBLE_SIZE = 5
        ensemble_models = []
        best_hist = None

        for ens_idx in range(ENSEMBLE_SIZE):
            torch.manual_seed(5 + ens_idx * 17 + fold_idx * 31)
            np.random.seed(5 + ens_idx * 17 + fold_idx * 31)
            log_dir = os.path.join(LOG_DIR, f"qsup_fold_{fold_idx}_ens{ens_idx}")

            model_i, train_loss_hist, val_loss_hist = train_extended_qsup_tb(
                X_train_res, y_train_res, X_test_scaled, y_test,
                input_dim=qsup_config['input_dim'],
                hidden_dim=qsup_config['hidden_dim'],
                num_classes=qsup_config['num_classes'],
                num_wavefunctions=qsup_config['num_wavefunctions'],
                partial_norm=qsup_config['partial_norm'],
                phase_per_dim=qsup_config['phase_per_dim'],
                self_modulation_steps=qsup_config['self_modulation_steps'],
                topk=qsup_config['topk'],
                epochs=300,
                patience=40,
                log_dir=log_dir,
                device=device
            )
            ensemble_models.append(model_i)
            if ens_idx == 0:
                best_hist = (train_loss_hist, val_loss_hist)

        qsup_histories.append(best_hist)

        # Ensemble prediction: average softmax outputs
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float, device=device)
        avg_proba = np.zeros((len(y_test), n_classes), dtype=np.float32)
        for model_i in ensemble_models:
            model_i.eval()
            with torch.no_grad():
                logits_i = model_i(X_test_tensor)
                proba_i = F.softmax(logits_i, dim=1).cpu().numpy()
            avg_proba += proba_i
        avg_proba /= ENSEMBLE_SIZE
        test_proba = avg_proba
        test_preds = np.argmax(test_proba, axis=1)
        # Keep first ensemble member as "the model" for saving
        qsup_model = ensemble_models[0]

        all_y_true.extend(y_test)
        all_y_pred.extend(test_preds)
        all_y_proba.extend(test_proba.tolist())   # full (n, num_classes) rows

        fold_acc = accuracy_score(y_test, test_preds)
        n_cls = len(np.unique(y_test))
        if n_cls > 1:
            fold_auc = roc_auc_score(y_test, test_proba[:, 1])
        else:
            fold_auc = 0.5
        fold_accuracies.append(fold_acc)
        fold_aucs.append(fold_auc)

        # Track best model by val AUC
        if fold_auc > best_val_auc:
            best_val_auc = fold_auc
            best_qsup_state = copy.deepcopy(qsup_model.state_dict())
            best_qsup_scaler = scaler
            best_qsup_model = qsup_model

        # ROC plot: one curve per class vs rest
        fig, ax = plt.subplots()
        for cls_i in range(test_proba.shape[1]):
            y_bin = (y_test == cls_i).astype(int)
            if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                fpr_i, tpr_i, _ = roc_curve(y_bin, test_proba[:, cls_i])
                auc_i = roc_auc_score(y_bin, test_proba[:, cls_i])
                lbl = ['Control', 'Alzheimer'][cls_i]
                ax.plot(fpr_i, tpr_i, label=f"{lbl} (AUC={auc_i:.2f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title(f"QSUP ROC Fold {fold_idx} (macro AUC={fold_auc:.3f})")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"qsup_roc_fold_{fold_idx}.png"))
        plt.close()

        # Loss plot for QSUP
        plt.figure()
        plt.plot(train_loss_hist, marker='o', label="Train Loss")
        plt.plot(val_loss_hist,   marker='x', label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Log Loss")
        plt.title(f"QSUP Train/Val Loss - Fold {fold_idx}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"qsup_loss_fold_{fold_idx}.png"))
        plt.close()

        print(f"[Fold {fold_idx}] Extended QSUP Accuracy: {fold_acc:.4f}, AUC: {fold_auc:.4f}")
        fold_idx += 1

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    print("\n--- Extended QSUP Overall Classification ---")
    print(confusion_matrix(all_y_true, all_y_pred))
    print(classification_report(all_y_true, all_y_pred, target_names=['Control', 'Alzheimer']))
    print(f"Overall Extended QSUP Accuracy: {overall_acc:.4f}")

    # Save best QSUP model state_dict and scaler
    if best_qsup_state is not None:
        qsup_state_path = os.path.join(MODELS_DIR, "qsup_model.pth")
        torch.save(best_qsup_state, qsup_state_path)
        print(f"[INFO] Saved best QSUP state_dict (AUC={best_val_auc:.4f}) to {qsup_state_path}")
    if best_qsup_scaler is not None:
        scaler_path = os.path.join(MODELS_DIR, "qsup_scaler.joblib")
        joblib.dump(best_qsup_scaler, scaler_path)
        print(f"[INFO] Saved QSUP scaler to {scaler_path}")

    return (np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba),
            qsup_histories, fold_accuracies, fold_aucs, best_qsup_model, best_qsup_scaler)

#############################################
# 8) END-TO-END MLP + GCN TRAINING
#    MLP loss backpropagates through the GCN,
#    updating both sets of weights jointly.
#############################################
def cv_mlp_e2e(X_handcrafted, X_extra, pyg_dataset, y, gcn_init,
               input_dim, device, epochs=100, patience=20, source_ids=None):
    """
    Joint GCN + MLP training: MLP loss gradients flow all the way back through the GCN.
    No SMOTE (synthetic samples lack graph structure).
    Uses class-weighted CE + mixup for regularisation.

    Algorithm per fold:
      1. Clone pre-trained GCN as starting point.
      2. Fit a SourceAwareScaler on the full CCV (HC + extra + frozen GCN embs).
      3. Each epoch: forward GCN differentiably → scale embs → concat with
         pre-scaled HC+extra → MLP → loss → backward → both GCN and MLP update.
      4. Save best MLP + GCN by val loss; evaluate by AUC.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    all_y_true, all_y_pred, all_y_proba = [], [], []
    fold_accuracies, fold_aucs = [], []

    best_val_auc = -1.0
    best_mlp_state = None
    best_gcn_state = None
    best_scaler = None
    num_classes = len(np.unique(y))

    fold_idx = 1
    for train_idx, test_idx in skf.split(X_handcrafted, y):
        HC_tr  = X_handcrafted[train_idx]
        ex_tr  = X_extra[train_idx]
        y_tr   = y[train_idx]
        HC_val = X_handcrafted[test_idx]
        ex_val = X_extra[test_idx]
        y_val  = y[test_idx]

        pyg_train_list = [pyg_dataset[i] for i in train_idx]
        pyg_val_list   = [pyg_dataset[i] for i in test_idx]

        # --- Fit scaler using initial frozen embeddings ---
        gcn = copy.deepcopy(gcn_init).to(device)
        gcn.eval()
        with torch.no_grad():
            init_loader = PyGDataLoader(pyg_dataset, batch_size=len(pyg_dataset), shuffle=False)
            for d in init_loader:
                d = d.to(device)
                all_init_emb = gcn.embed(d.x, d.edge_index, d.batch).cpu().numpy()

        init_CCV_tr = np.hstack([HC_tr, ex_tr, all_init_emb[train_idx]])
        if source_ids is not None:
            scaler = SourceAwareScaler()
            scaler.fit(init_CCV_tr, source_ids[train_idx])
        else:
            scaler = StandardScaler()
            scaler.fit(init_CCV_tr)

        # GCN emb scaler — global (used for differentiable normalisation in the training loop)
        base_dim = HC_tr.shape[1] + ex_tr.shape[1]   # 72
        emb_scaler   = StandardScaler().fit(all_init_emb[train_idx])
        emb_mean_np  = emb_scaler.mean_
        emb_scale_np = emb_scaler.scale_

        # Pre-scale static HC+extra with global scaler
        HC_ex_tr_raw  = np.hstack([HC_tr, ex_tr])
        HC_ex_val_raw = np.hstack([HC_val, ex_val])
        base_scaler = StandardScaler()
        HC_ex_tr_s  = base_scaler.fit_transform(HC_ex_tr_raw).astype(np.float32)
        HC_ex_val_s = base_scaler.transform(HC_ex_val_raw).astype(np.float32)

        X_base_t   = torch.tensor(HC_ex_tr_s, dtype=torch.float, device=device)
        y_tr_t     = torch.tensor(y_tr,       dtype=torch.long,  device=device)
        emb_mean_t = torch.tensor(emb_mean_np,  dtype=torch.float, device=device)
        emb_std_t  = torch.tensor(emb_scale_np, dtype=torch.float, device=device)

        # Source one-hot tensors for train and val folds
        if source_ids is not None:
            src_tr_oh  = np.zeros((len(train_idx), N_SOURCES), dtype=np.float32)
            src_tr_oh[np.arange(len(train_idx)), source_ids[train_idx]] = 1.0
            src_val_oh = np.zeros((len(test_idx),  N_SOURCES), dtype=np.float32)
            src_val_oh[np.arange(len(test_idx)),  source_ids[test_idx]]  = 1.0
            src_tr_t  = torch.tensor(src_tr_oh,  dtype=torch.float, device=device)
            src_val_t = torch.tensor(src_val_oh, dtype=torch.float, device=device)
        else:
            src_tr_t  = None
            src_val_t = None

        # --- Build fresh MLP and joint optimizer ---
        mlp = TorchMLP(input_dim, num_classes).to(device)
        class_weights = compute_class_weights(y_tr, device)
        loss_fn   = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        optimizer = AdamW(
            list(mlp.parameters()) + list(gcn.parameters()),
            lr=3e-4, weight_decay=5e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss  = float('inf')
        best_epoch     = 0
        best_state_pair = None

        for epoch in range(1, epochs + 1):
            gcn.train(); mlp.train()
            optimizer.zero_grad()

            # Forward GCN on training graphs — DIFFERENTIABLE
            tr_loader = PyGDataLoader(pyg_train_list, batch_size=len(pyg_train_list), shuffle=False)
            for batch_data in tr_loader:
                batch_data = batch_data.to(device)
                emb = gcn.embed(batch_data.x, batch_data.edge_index, batch_data.batch)

            # Differentiable normalisation of GCN embs (gradients flow through here)
            emb_scaled = (emb - emb_mean_t) / (emb_std_t + 1e-8)
            parts = [X_base_t, emb_scaled]
            if src_tr_t is not None:
                parts.append(src_tr_t)
            X_full = torch.cat(parts, dim=1)  # (N_train, input_dim)

            mixed_x, y_a, y_b, lam = mixup_data(X_full, y_tr_t, alpha=0.2)
            logits = mlp(mixed_x)
            loss   = mixup_criterion(loss_fn, logits, y_a, y_b, lam)
            loss.backward()   # <-- gradients flow: MLP → emb_scaled → gcn.embed()
            torch.nn.utils.clip_grad_norm_(
                list(mlp.parameters()) + list(gcn.parameters()), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            # --- Validation ---
            gcn.eval(); mlp.eval()
            with torch.no_grad():
                val_loader = PyGDataLoader(pyg_val_list, batch_size=len(pyg_val_list), shuffle=False)
                for vd in val_loader:
                    vd = vd.to(device)
                    val_emb = gcn.embed(vd.x, vd.edge_index, vd.batch)
                val_emb_s  = (val_emb - emb_mean_t) / (emb_std_t + 1e-8)
                X_val_base = torch.tensor(HC_ex_val_s, dtype=torch.float, device=device)
                val_parts = [X_val_base, val_emb_s]
                if src_val_t is not None:
                    val_parts.append(src_val_t)
                X_val_full = torch.cat(val_parts, dim=1)
                val_logits = mlp(X_val_full)
                val_proba  = F.softmax(val_logits, dim=1).cpu().numpy()
                val_loss   = log_loss(y_val, val_proba)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch    = epoch
                best_state_pair = (copy.deepcopy(mlp.state_dict()), copy.deepcopy(gcn.state_dict()))
            elif epoch - best_epoch >= patience:
                print(f"  [E2E Fold {fold_idx}] Early stop at epoch {epoch} (best: {best_epoch})")
                break

        # Restore best states
        if best_state_pair:
            mlp.load_state_dict(best_state_pair[0])
            gcn.load_state_dict(best_state_pair[1])

        # --- Final fold evaluation ---
        gcn.eval(); mlp.eval()
        with torch.no_grad():
            val_loader = PyGDataLoader(pyg_val_list, batch_size=len(pyg_val_list), shuffle=False)
            for vd in val_loader:
                vd = vd.to(device)
                val_emb = gcn.embed(vd.x, vd.edge_index, vd.batch)
            val_emb_s  = (val_emb - emb_mean_t) / (emb_std_t + 1e-8)
            X_val_base = torch.tensor(HC_ex_val_s, dtype=torch.float, device=device)
            final_val_parts = [X_val_base, val_emb_s]
            if src_val_t is not None:
                final_val_parts.append(src_val_t)
            X_val_full = torch.cat(final_val_parts, dim=1)
            val_logits = mlp(X_val_full)
            val_proba  = F.softmax(val_logits, dim=1).cpu().numpy()
        val_preds = val_proba.argmax(axis=1)

        fold_acc = accuracy_score(y_val, val_preds)
        n_cls_e2e = len(np.unique(y_val))
        fold_auc = (roc_auc_score(y_val, val_proba[:, 1])
                    if n_cls_e2e > 1 else 0.5)
        fold_accuracies.append(fold_acc)
        fold_aucs.append(fold_auc)
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(val_preds.tolist())
        all_y_proba.extend(val_proba.tolist())    # full (n, num_classes) rows
        print(f"  [E2E Fold {fold_idx}] Acc={fold_acc:.4f}  AUC={fold_auc:.4f}")

        if fold_auc > best_val_auc:
            best_val_auc   = fold_auc
            best_mlp_state = copy.deepcopy(mlp.state_dict())
            best_gcn_state = copy.deepcopy(gcn.state_dict())
            best_scaler    = scaler
        fold_idx += 1

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_auc = roc_auc_score(all_y_true, np.array(all_y_proba)[:, 1])
    print(f"\n--- E2E MLP+GCN: Accuracy={overall_acc:.4f}  AUC={overall_auc:.4f} ---")

    if best_mlp_state:
        torch.save(best_mlp_state, os.path.join(MODELS_DIR, "mlp_e2e.pth"))
        print(f"[E2E] Saved mlp_e2e.pth (best fold AUC={best_val_auc:.4f})")
    if best_gcn_state:
        torch.save(best_gcn_state, os.path.join(MODELS_DIR, "gcn_e2e.pth"))
        print("[E2E] Saved gcn_e2e.pth")
    if best_scaler:
        joblib.dump(best_scaler, os.path.join(MODELS_DIR, "e2e_scaler.joblib"))

    return (np.array(all_y_true), np.array(all_y_pred),
            np.array(all_y_proba), fold_accuracies, fold_aucs)

#############################################
# 9) VISUALIZATION FUNCTIONS
#############################################

def plot_confusion_matrices(all_y_true_mlp, all_y_pred_mlp, all_y_true_qsup, all_y_pred_qsup):
    """Side-by-side confusion matrix heatmaps for MLP and QSUP."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_mlp = confusion_matrix(all_y_true_mlp, all_y_pred_mlp)
    sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Control', 'Alzheimer'], yticklabels=['Control', 'Alzheimer'])
    axes[0].set_title('MLP Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    cm_qsup = confusion_matrix(all_y_true_qsup, all_y_pred_qsup)
    sns.heatmap(cm_qsup, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=['Control', 'Alzheimer'], yticklabels=['Control', 'Alzheimer'])
    axes[1].set_title('QSUP Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()
    print("[VIZ] Saved confusion_matrices.png")


def plot_training_comparison(mlp_histories, qsup_histories):
    """Overlay all fold loss curves for both models in a 2x1 subplot."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    # MLP folds
    for i, (train_hist, val_hist) in enumerate(mlp_histories):
        c = colors[i % len(colors)]
        axes[0].plot(train_hist, color=c, alpha=0.8, label=f'Fold {i+1} Train')
        axes[0].plot(val_hist, color=c, linestyle='--', alpha=0.8, label=f'Fold {i+1} Val')
    axes[0].set_title('MLP Training Curves (All Folds)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Log Loss')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # QSUP folds
    for i, (train_hist, val_hist) in enumerate(qsup_histories):
        c = colors[i % len(colors)]
        axes[1].plot(train_hist, color=c, alpha=0.8, label=f'Fold {i+1} Train')
        axes[1].plot(val_hist, color=c, linestyle='--', alpha=0.8, label=f'Fold {i+1} Val')
    axes[1].set_title('QSUP Training Curves (All Folds)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Log Loss')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_comparison.png"), dpi=150)
    plt.close()
    print("[VIZ] Saved training_comparison.png")


def plot_qsup_internals(model, X_sample, feature_names=None):
    """Visualize QSUP internal states: wavefunction magnitudes, interference weights, phases, probs."""
    device = next(model.parameters()).device
    X_t = torch.tensor(X_sample, dtype=torch.float, device=device)

    model.eval()
    with torch.no_grad():
        _, internals = model.forward_with_internals(X_t)

    wave_mags = internals['wave_magnitudes'].cpu().numpy()   # (batch, num_wf, hidden_dim)
    interf_w = internals['interference_weights'].cpu().numpy()  # (batch, num_wf)
    phases = internals['phases'].cpu().numpy()                  # (num_wf, dim)
    probs = internals['probs'].cpu().numpy()                    # (batch, hidden_dim)

    # Average across batch
    avg_mags = wave_mags.mean(axis=0)        # (num_wf, hidden_dim)
    avg_interf = interf_w.mean(axis=0)       # (num_wf,)
    avg_probs = probs.mean(axis=0)           # (hidden_dim,)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: Wavefunction magnitudes heatmap
    sns.heatmap(avg_mags, ax=axes[0, 0], cmap='viridis', cbar=True)
    axes[0, 0].set_title('Wavefunction Magnitudes (avg over samples)')
    axes[0, 0].set_xlabel('Hidden Dim')
    axes[0, 0].set_ylabel('Wavefunction Index')

    # Subplot 2: Interference weights as stacked bar
    num_wf = avg_interf.shape[0]
    axes[0, 1].bar(range(num_wf), avg_interf, color=[f'C{i}' for i in range(num_wf)])
    axes[0, 1].set_title('Interference Weights (avg over samples)')
    axes[0, 1].set_xlabel('Wavefunction Index')
    axes[0, 1].set_ylabel('Weight')
    axes[0, 1].set_xticks(range(num_wf))

    # Subplot 3: Learned phases as polar plot
    ax_polar = fig.add_subplot(2, 2, 3, projection='polar')
    axes[1, 0].set_visible(False)
    for wf_idx in range(phases.shape[0]):
        phase_vals = phases[wf_idx].flatten()
        # Plot first few phase values as points on polar plot
        r = np.ones_like(phase_vals) * (wf_idx + 1)
        ax_polar.scatter(phase_vals, r, alpha=0.6, s=15, label=f'WF {wf_idx}')
    ax_polar.set_title('Learned Phase Values', pad=20)
    ax_polar.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # Subplot 4: Measurement probabilities bar chart
    axes[1, 1].bar(range(len(avg_probs)), avg_probs, color='teal', alpha=0.8)
    axes[1, 1].set_title('Measurement Probabilities (avg over samples)')
    axes[1, 1].set_xlabel('Hidden Dim Index')
    axes[1, 1].set_ylabel('Probability')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "qsup_internals.png"), dpi=150)
    plt.close()
    print("[VIZ] Saved qsup_internals.png")


def plot_feature_importance(model, X, y, feature_names):
    """Gradient-based feature importance: average |grad(loss) w.r.t. input| across samples."""
    device = next(model.parameters()).device
    X_t = torch.tensor(X, dtype=torch.float, device=device, requires_grad=True)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    model.eval()
    logits = model(X_t)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, y_t)
    loss.backward()

    grad_importance = X_t.grad.abs().mean(dim=0).cpu().numpy()

    # Sort by importance
    sorted_idx = np.argsort(grad_importance)
    sorted_importance = grad_importance[sorted_idx]

    if feature_names is not None and len(feature_names) == len(grad_importance):
        sorted_names = [feature_names[i] for i in sorted_idx]
    else:
        sorted_names = [f"F{i}" for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, max(6, len(sorted_names) * 0.2)))
    ax.barh(range(len(sorted_importance)), sorted_importance, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=6)
    ax.set_xlabel('Mean |Gradient|')
    ax.set_title('Gradient-Based Feature Importance (QSUP)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("[VIZ] Saved feature_importance.png")


def plot_embedding_tsne(CCV, y, title="CCV t-SNE"):
    """t-SNE visualization of CCV colored by class."""
    perplexity = min(30, len(y) - 1)
    tsne = TSNE(n_components=2, random_state=5, perplexity=perplexity)
    X_2d = tsne.fit_transform(CCV)

    fig, ax = plt.subplots(figsize=(8, 6))
    mask_ctrl = y == 0
    mask_ad = y == 1
    ax.scatter(X_2d[mask_ctrl, 0], X_2d[mask_ctrl, 1], c='blue', alpha=0.7,
               label='Control', edgecolors='k', linewidth=0.3, s=50)
    ax.scatter(X_2d[mask_ad, 0], X_2d[mask_ad, 1], c='red', alpha=0.7,
               label='Alzheimer', edgecolors='k', linewidth=0.3, s=50)
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "tsne_embeddings.png"), dpi=150)
    plt.close()
    print("[VIZ] Saved tsne_embeddings.png")


def plot_roc_comparison(all_y_true, all_y_proba_mlp, all_y_proba_qsup):
    """
    Multi-class ROC comparison (one-vs-rest per class) for MLP and QSUP.
    all_y_proba_* are (N, num_classes) arrays.
    """
    y_true = np.array(all_y_true)
    mlp_p  = np.array(all_y_proba_mlp)
    qsup_p = np.array(all_y_proba_qsup)
    num_classes = mlp_p.shape[1]
    class_names = ['Control', 'FTD', 'Alzheimer'][:num_classes]
    colors_mlp  = ['tab:blue',   'tab:cyan',   'steelblue']
    colors_qsup = ['tab:orange', 'tab:red',    'saddlebrown']

    fig, ax = plt.subplots(figsize=(9, 6))
    for cls_i in range(num_classes):
        y_bin = (y_true == cls_i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr_m, tpr_m, _ = roc_curve(y_bin, mlp_p[:, cls_i])
        auc_m = roc_auc_score(y_bin, mlp_p[:, cls_i])
        ax.plot(fpr_m, tpr_m, color=colors_mlp[cls_i], linewidth=2,
                label=f'MLP {class_names[cls_i]} (AUC={auc_m:.2f})')
        fpr_q, tpr_q, _ = roc_curve(y_bin, qsup_p[:, cls_i])
        auc_q = roc_auc_score(y_bin, qsup_p[:, cls_i])
        ax.plot(fpr_q, tpr_q, color=colors_qsup[cls_i], linewidth=2, linestyle='--',
                label=f'QSUP {class_names[cls_i]} (AUC={auc_q:.2f})')

    macro_mlp  = roc_auc_score(y_true, mlp_p[:, 1])
    macro_qsup = roc_auc_score(y_true, qsup_p[:, 1])
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_title(f'ROC: MLP macro={macro_mlp:.3f} vs QSUP macro={macro_qsup:.3f}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_comparison.png"), dpi=150)
    plt.close()
    print("[VIZ] Saved roc_comparison.png")

#############################################
# 9) MAIN
#############################################
def main():
    start_time = time.time()
    # 1) Load features
    X_handcrafted, X_gnn, y, ch_names_list, subj_ids, source_ids = load_saved_features()
    print(f"[INFO] Handcrafted shape: {X_handcrafted.shape}")
    print(f"[INFO] {len(X_gnn)} GNN feature matrices loaded.")
    for sid, name in enumerate(SOURCE_NAMES):
        print(f"[INFO]   Source {sid} ({name}): {(source_ids == sid).sum()} subjects")

    # 1b) Extract engineered features from GNN band-power matrices (42 dims each)
    X_extra = np.stack([extract_extra_features_from_gnn(g) for g in X_gnn], axis=0)
    print(f"[INFO] Extra GNN features shape: {X_extra.shape}")   # (N, 42)

    # 2) Build PyG dataset for GCN
    pyg_dataset, valid_idx = create_pyg_dataset(X_gnn, y, ch_names_list)
    if len(pyg_dataset) == 0:
        raise ValueError("No valid PyG dataset could be constructed!")
    # Filter handcrafted/extra/y to only valid subjects (those with adjacency matrices)
    valid_idx = np.array(valid_idx)
    X_handcrafted = X_handcrafted[valid_idx]
    X_extra       = X_extra[valid_idx]
    y             = y[valid_idx]
    source_ids    = source_ids[valid_idx]
    print(f"[INFO] Valid subjects after adjacency filter: {len(valid_idx)}/{len(X_gnn)}")
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
    hidden_channels = 64          # expanded from 32 for richer embeddings
    gcn_num_classes = len(np.unique(y))   # 2-class: Control / Alzheimer (FTD counted as AD)
    gcn_model = GCNNet(in_channels, hidden_channels, gcn_num_classes).to(device)

    # Class-weighted loss for GCN
    gcn_labels = np.array([d.y.item() for d in pyg_dataset])
    gcn_train_labels = gcn_labels[train_idx]
    gcn_class_weights = compute_class_weights(gcn_train_labels, device)
    gcn_loss_fn = nn.CrossEntropyLoss(weight=gcn_class_weights)

    optimizer = TorchAdam(gcn_model.parameters(), lr=10**-2.25)
    epochs_gcn = 150

    # GCN early stopping
    gcn_patience = 15
    gcn_best_val_loss = float('inf')
    gcn_best_epoch = 0
    gcn_best_state = None

    print("[GCN] Training embeddings ...")
    for epoch in range(1, epochs_gcn + 1):
        gcn_model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            logits = gcn_model(data.x, data.edge_index, data.batch)
            loss = gcn_loss_fn(logits, data.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        avg_loss = total_loss / len(train_loader.dataset)

        # Validation loop
        gcn_model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                logits = gcn_model(data.x, data.edge_index, data.batch)
                loss = gcn_loss_fn(logits, data.y.view(-1))
                val_total_loss += loss.item() * data.num_graphs
        avg_val_loss = val_total_loss / len(val_loader.dataset)

        if epoch % 5 == 0:
            print(f"GCN Epoch {epoch}/{epochs_gcn} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < gcn_best_val_loss:
            gcn_best_val_loss = avg_val_loss
            gcn_best_epoch = epoch
            gcn_best_state = copy.deepcopy(gcn_model.state_dict())
        elif epoch - gcn_best_epoch >= gcn_patience:
            print(f"[GCN] Early stopping at epoch {epoch} (best epoch: {gcn_best_epoch})")
            break

    # Restore best GCN weights
    if gcn_best_state is not None:
        gcn_model.load_state_dict(gcn_best_state)

    # Save GCN model
    gcn_state_path = os.path.join(MODELS_DIR, "gcn_model.pth")
    torch.save(gcn_model.state_dict(), gcn_state_path)
    print(f"[INFO] Saved GCN state_dict to {gcn_state_path}")

    # Extract embeddings for entire dataset
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

    # Combine => CCV (30 handcrafted + 42 extra GNN + 64 GCN embeddings + 4 source = 140 dims)
    if X_handcrafted.shape[0] != embeddings.shape[0]:
        raise ValueError("Mismatch between handcrafted features and GCN embeddings!")
    source_onehot = np.zeros((len(source_ids), N_SOURCES), dtype=np.float32)
    source_onehot[np.arange(len(source_ids)), source_ids] = 1.0
    CCV = np.hstack((X_handcrafted, X_extra, embeddings, source_onehot))
    print(f"[INFO] Final CCV shape: {CCV.shape}")   # (N, 140)

    # Build feature names for visualization
    handcrafted_names = [f"HC_{i}" for i in range(X_handcrafted.shape[1])]
    extra_names = (
        [f"bp_mean_{i}" for i in range(9)] +
        [f"bp_std_{i}"  for i in range(9)] +
        [f"bp_log_{i}"  for i in range(9)] +
        [f"bp_norm_{i}" for i in range(9)] +
        ["theta_alpha", "alpha_beta", "delta_alpha", "slow_total", "delta_total", "gamma_beta"]
    )
    gcn_emb_names = [f"GCN_{i}" for i in range(embeddings.shape[1])]
    feature_names = handcrafted_names + extra_names + gcn_emb_names

    # 5-fold classification with MLP
    print("\n=== 5-Fold Classification with MLP (PyTorch) ===")
    (mlp_y_true, mlp_y_pred, mlp_y_proba,
     mlp_histories, mlp_fold_accs, mlp_fold_aucs) = cv_classification_MLP(CCV, y, source_ids)

    # 5-fold classification with Extended QSUP
    print("\n=== 5-Fold Classification with Extended QSUP v2 ===")
    (qsup_y_true, qsup_y_pred, qsup_y_proba,
     qsup_histories, qsup_fold_accs, qsup_fold_aucs,
     best_qsup_model, best_qsup_scaler) = cv_classification_ExtendedQSUP(CCV, y, source_ids)

    # ===== E2E JOINT GCN+MLP TRAINING =====
    print("\n=== End-to-End Joint GCN+MLP Training (backprop through GCN) ===")
    (e2e_y_true, e2e_y_pred, e2e_y_proba,
     e2e_fold_accs, e2e_fold_aucs) = cv_mlp_e2e(
        X_handcrafted, X_extra, pyg_dataset, y,
        gcn_init=gcn_model, input_dim=CCV.shape[1], device=device,
        epochs=100, patience=20, source_ids=source_ids
    )

    # ===== VISUALIZATIONS =====
    print("\n=== Generating Visualizations ===")

    # a) Confusion matrices
    plot_confusion_matrices(mlp_y_true, mlp_y_pred, qsup_y_true, qsup_y_pred)

    # b) Training comparison
    plot_training_comparison(mlp_histories, qsup_histories)

    # c) QSUP internals
    if best_qsup_model is not None and best_qsup_scaler is not None:
        X_sample = best_qsup_scaler.transform(CCV)
        plot_qsup_internals(best_qsup_model, X_sample[:20], feature_names=feature_names)

        # d) Feature importance (gradient-based)
        y_sample = y[:20]
        plot_feature_importance(best_qsup_model, X_sample[:20], y_sample, feature_names)

    # e) t-SNE embeddings
    plot_embedding_tsne(CCV, y, title="CCV t-SNE (Control vs Alzheimer)")

    # f) ROC comparison
    plot_roc_comparison(mlp_y_true, mlp_y_proba, qsup_y_proba)

    # ===== FINAL SUMMARY TABLE =====
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: Per-Fold Accuracy and AUC")
    print("=" * 80)
    print(f"{'Fold':<6} {'MLP Acc':<11} {'MLP AUC':<11} {'QSUP Acc':<11} {'QSUP AUC':<11} {'E2E Acc':<11} {'E2E AUC':<11}")
    print("-" * 80)
    n_folds = 5
    for i in range(n_folds):
        mlp_acc_str  = f"{mlp_fold_accs[i]:.4f}"  if i < len(mlp_fold_accs)  else "N/A"
        mlp_auc_str  = f"{mlp_fold_aucs[i]:.4f}"  if i < len(mlp_fold_aucs)  else "N/A"
        qsup_acc_str = f"{qsup_fold_accs[i]:.4f}" if i < len(qsup_fold_accs) else "N/A"
        qsup_auc_str = f"{qsup_fold_aucs[i]:.4f}" if i < len(qsup_fold_aucs) else "N/A"
        e2e_acc_str  = f"{e2e_fold_accs[i]:.4f}"  if i < len(e2e_fold_accs)  else "N/A"
        e2e_auc_str  = f"{e2e_fold_aucs[i]:.4f}"  if i < len(e2e_fold_aucs)  else "N/A"
        print(f"{i+1:<6} {mlp_acc_str:<11} {mlp_auc_str:<11} {qsup_acc_str:<11} {qsup_auc_str:<11} {e2e_acc_str:<11} {e2e_auc_str:<11}")
    print("-" * 80)
    print(f"{'Mean':<6} {np.mean(mlp_fold_accs):<11.4f} {np.mean(mlp_fold_aucs):<11.4f} "
          f"{np.mean(qsup_fold_accs):<11.4f} {np.mean(qsup_fold_aucs):<11.4f} "
          f"{np.mean(e2e_fold_accs):<11.4f} {np.mean(e2e_fold_aucs):<11.4f}")
    print("=" * 80)

    # ===== META-ENSEMBLE: MLP + QSUP + E2E =====
    # All three use StratifiedKFold(random_state=5) on same data → y_true arrays align.
    # all_y_proba arrays are now (N, num_classes) matrices.
    mlp_proba_arr  = np.array(mlp_y_proba)   # (N, 3)
    qsup_proba_arr = np.array(qsup_y_proba)  # (N, 3)
    e2e_proba_arr  = np.array(e2e_y_proba)   # (N, 3)

    # 2-model ensemble (MLP + QSUP)
    meta2_proba = (mlp_proba_arr + qsup_proba_arr) / 2.0
    meta2_preds = np.argmax(meta2_proba, axis=1)
    meta2_acc   = accuracy_score(mlp_y_true, meta2_preds)
    meta2_auc   = roc_auc_score(mlp_y_true, meta2_proba[:, 1])
    print(f"\n{'META-ENSEMBLE-2 (MLP+QSUP):':42s} Accuracy={meta2_acc:.4f}  AUC={meta2_auc:.4f}")

    # 3-model ensemble (MLP + QSUP + E2E)
    meta3_proba = (mlp_proba_arr + qsup_proba_arr + e2e_proba_arr) / 3.0
    meta3_preds = np.argmax(meta3_proba, axis=1)
    meta3_acc   = accuracy_score(mlp_y_true, meta3_preds)
    meta3_auc   = roc_auc_score(mlp_y_true, meta3_proba[:, 1])
    print(f"{'META-ENSEMBLE-3 (MLP+QSUP+E2E):':42s} Accuracy={meta3_acc:.4f}  AUC={meta3_auc:.4f}")
    print(confusion_matrix(mlp_y_true, meta3_preds))
    print(classification_report(mlp_y_true, meta3_preds, target_names=['Control', 'Alzheimer']))

    end_time = time.time()
    print(f"[DONE] total runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
