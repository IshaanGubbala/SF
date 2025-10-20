import os
import io
import time
import json
from datetime import datetime
import numpy as np
import streamlit as st

# Optional heavy deps; app should degrade gracefully if missing
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# PyG is optional and separate from torch
PYG_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        from torch_geometric.nn import GCNConv, global_mean_pool
        PYG_AVAILABLE = True
    except Exception:
        PYG_AVAILABLE = False

try:
    import joblib
except Exception:  # joblib is required for sklearn models
    joblib = None

try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


MODELS_DIR = "trained_models"
PLOTS_DIR = "plots"

MLP_MODEL_PATH = os.path.join(MODELS_DIR, "mlp_model.pkl")
QSUP_MODEL_PATH = os.path.join(MODELS_DIR, "qsup_model.pth")
GCN_MODEL_PATH = os.path.join(MODELS_DIR, "gcn_model.pth")
PCA_MODEL_PATH = os.path.join(MODELS_DIR, "pca_model.joblib")

NUM_CHANNELS = 4
SAMPLING_RATE = 256
WINDOW_SECONDS = 20
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SECONDS

FREQUENCY_BANDS_9_LIST = [
    ("Delta",  (0.5, 4)),
    ("Theta1", (4, 6)),
    ("Theta2", (6, 8)),
    ("Alpha1", (8,10)),
    ("Alpha2", (10,12)),
    ("Beta1",  (12,18)),
    ("Beta2",  (18,22)),
    ("Beta3",  (22,26)),
    ("Beta4",  (26,30)),
]
FREQUENCY_BANDS_9 = dict(FREQUENCY_BANDS_9_LIST)


# -----------------------------
# Models (mirror server shapes)
# -----------------------------
if TORCH_AVAILABLE and PYG_AVAILABLE:
    class GCNNet(nn.Module):
        def __init__(self, in_channels, hidden_channels, num_classes):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.lin   = nn.Linear(hidden_channels, num_classes)

        def forward(self, x, edge_index, batch):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))
            return global_mean_pool(x, batch)

        def embed(self, x, edge_index, batch):
            return self.forward(x, edge_index, batch)
else:
    GCNNet = None  # type: ignore


if TORCH_AVAILABLE:
    class ExtendedQSUP(nn.Module):
        """
        Extended QSUP Model copied to match the training architecture in ai_model.py
        to ensure loaded state_dict aligns and predictions behave correctly.
        """
        def __init__(self, input_dim, hidden_dim, num_classes,
                     num_wavefunctions=3, partial_norm=1.5,
                     phase_per_dim=False, self_modulation_steps=2,
                     topk=8):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.num_wavefunctions = num_wavefunctions
            self.partial_norm = partial_norm
            self.phase_per_dim = phase_per_dim
            self.self_modulation_steps = self_modulation_steps
            self.topk = topk

            self.wavefunction_nets = nn.ModuleList([
                nn.Linear(input_dim, 2 * hidden_dim) for _ in range(num_wavefunctions)
            ])
            if phase_per_dim:
                self.phases = nn.Parameter(torch.zeros(num_wavefunctions, hidden_dim))
            else:
                self.phases = nn.Parameter(torch.zeros(num_wavefunctions, 1))

            if self_modulation_steps > 0:
                self.gating_net = nn.Linear(hidden_dim, hidden_dim)
            else:
                self.gating_net = None

            self.classifier = nn.Linear(hidden_dim, num_classes)
            nn.init.constant_(self.classifier.bias, 0.0)
            # Slight bias for AD as in training script
            with torch.no_grad():
                if self.classifier.bias.numel() > 1:
                    self.classifier.bias[1] = 0.75

        def forward(self, x):
            batch_size = x.size(0)
            eps = 1e-8

            wave_r_list = []
            wave_i_list = []
            for s in range(self.num_wavefunctions):
                out = self.wavefunction_nets[s](x)
                out = torch.exp(-out * out)  # ArcBell
                alpha = out[:, :self.hidden_dim]
                beta  = out[:, self.hidden_dim:]
                norm_sq = torch.sum(alpha**2 + beta**2, dim=1, keepdim=True) + eps
                factor = torch.sqrt((self.partial_norm**2) / norm_sq)
                alpha = alpha * factor
                beta  = beta  * factor

                if self.phase_per_dim:
                    phase = self.phases[s].unsqueeze(0)
                else:
                    phase = self.phases[s]
                wave_r = alpha * torch.cos(phase) - beta * torch.sin(phase)
                wave_i = alpha * torch.sin(phase) + beta * torch.cos(phase)
                wave_r_list.append(wave_r)
                wave_i_list.append(wave_i)

            real_stack = torch.stack(wave_r_list, dim=1)   # (batch, S, H)
            imag_stack = torch.stack(wave_i_list, dim=1)   # (batch, S, H)

            mean_real = torch.mean(real_stack, dim=1)      # (batch, H)
            mean_norm = torch.sqrt(torch.sum(mean_real**2, dim=1, keepdim=True)) + eps
            mean_norm = mean_norm.unsqueeze(1)
            wave_norms = torch.sqrt(torch.sum(real_stack**2, dim=2, keepdim=True)) + eps
            dot_prod = torch.sum(real_stack * mean_real.unsqueeze(1), dim=2, keepdim=True)
            cosine_sim = dot_prod / (wave_norms * mean_norm)
            cosine_sim = cosine_sim.squeeze(2)
            interference_weights = F.softmax(cosine_sim, dim=1).unsqueeze(2)

            sup_real = torch.sum(real_stack * interference_weights, dim=1)
            sup_imag = torch.sum(imag_stack * interference_weights, dim=1)

            if self.self_modulation_steps > 0 and self.gating_net is not None:
                for _ in range(self.self_modulation_steps):
                    mag = torch.sqrt(sup_real**2 + sup_imag**2 + eps)
                    gate = torch.sigmoid(self.gating_net(mag))
                    sup_real = sup_real * gate
                    sup_imag = sup_imag * gate

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
            return logits
else:
    ExtendedQSUP = None  # type: ignore


# -----------------------------
# Feature computation helpers
# -----------------------------
def adjacency_4ch():
    A = np.ones((NUM_CHANNELS, NUM_CHANNELS), dtype=np.float32)
    np.fill_diagonal(A, 0)
    return A


def _psd_numpy(sig: np.ndarray, fs: int):
    # Simple PSD via rFFT with Hann window fallback when scipy is not available
    x = sig.astype(np.float64)
    n = x.shape[-1]
    if n < 8:
        n = 8
    # window
    w = np.hanning(n)
    xw = x[-n:] * w
    X = np.fft.rfft(xw)
    psd = (np.abs(X) ** 2) / (np.sum(w**2) * fs)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    return freqs, psd


def compute_9subbands(sig: np.ndarray):
    if SCIPY_AVAILABLE:
        freqs, psd = scipy.signal.welch(sig, SAMPLING_RATE, nperseg=512)
    else:
        freqs, psd = _psd_numpy(sig, SAMPLING_RATE)
    feats = []
    for (low, high) in FREQUENCY_BANDS_9.values():
        idx = (freqs >= low) & (freqs < high)
        feats.append(float(np.sum(psd[idx])))
    return feats


def channelwise_9feats(chunk_4ch: np.ndarray):
    out = []
    for c in range(NUM_CHANNELS):
        feats9 = compute_9subbands(chunk_4ch[c])
        out.append(feats9)
    return np.array(out, dtype=np.float32)


def hjorth_params(sig):
    x = sig.flatten()
    act = np.var(x)
    dx = np.diff(x)
    mob = np.sqrt(np.var(dx) / (act + 1e-8))
    ddx = np.diff(dx)
    mob_dx = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-8))
    comp = mob_dx / (mob + 1e-8)
    return act, mob, comp


def alpha_ratio(sig):
    if SCIPY_AVAILABLE:
        freqs, psd = scipy.signal.welch(sig, SAMPLING_RATE, nperseg=512)
    else:
        freqs, psd = _psd_numpy(sig, SAMPLING_RATE)
    mask_tot = (freqs >= 0.5) & (freqs < 30)
    mask_a   = (freqs >= 8) & (freqs < 12)
    tot_pow  = np.sum(psd[mask_tot])
    alp_pow  = np.sum(psd[mask_a])
    return alp_pow / (tot_pow + 1e-12)


def spectral_entropy(sig):
    if SCIPY_AVAILABLE:
        freqs, psd = scipy.signal.welch(sig, SAMPLING_RATE, nperseg=512)
    else:
        freqs, psd = _psd_numpy(sig, SAMPLING_RATE)
    mask = (freqs >= 0.5) & (freqs < 30)
    psd_sub = psd[mask]
    psd_norm = psd_sub / (np.sum(psd_sub) + 1e-12)
    ent = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
    return ent


def aggregator_30(chunk_4ch: np.ndarray):
    alpha_ratios = []
    entropies = []
    hj_all = []
    for c in range(NUM_CHANNELS):
        arr = chunk_4ch[c]
        ar = alpha_ratio(arr)
        se = spectral_entropy(arr)
        act, mob, comp = hjorth_params(arr)
        alpha_ratios.append(ar)
        entropies.append(se)
        hj_all.extend([act, mob, comp])

    alpha_mean = float(np.mean(alpha_ratios))
    alpha_std  = float(np.std(alpha_ratios))
    ent_mean   = float(np.mean(entropies))
    ent_std    = float(np.std(entropies))

    hj_arr = np.array(hj_all).reshape(NUM_CHANNELS, 3)
    hj_act_mean  = float(np.mean(hj_arr[:, 0]))
    hj_act_std   = float(np.std(hj_arr[:, 0]))
    hj_mob_mean  = float(np.mean(hj_arr[:, 1]))
    hj_mob_std   = float(np.std(hj_arr[:, 1]))
    hj_comp_mean = float(np.mean(hj_arr[:, 2]))
    hj_comp_std  = float(np.std(hj_arr[:, 2]))

    aggregator_10 = [
        alpha_mean, alpha_std,
        ent_mean,   ent_std,
        hj_act_mean, hj_act_std,
        hj_mob_mean, hj_mob_std,
        hj_comp_mean, hj_comp_std,
    ]

    feats_20 = []
    for c in range(NUM_CHANNELS):
        feats_20.append(alpha_ratios[c])
        feats_20.append(entropies[c])
        off = c * 3
        feats_20.append(hj_all[off + 0])
        feats_20.append(hj_all[off + 1])
        feats_20.append(hj_all[off + 2])

    final_30 = np.concatenate([
        np.array(feats_20, dtype=np.float32),
        np.array(aggregator_10, dtype=np.float32),
    ])
    return final_30


def compute_energy(feats_vec_30: np.ndarray) -> float:
    return float(np.sum(feats_vec_30))


# -----------------------------
# Simulation helpers (Healthy vs Alzheimer's)
# -----------------------------
# Real-recording backed "simulation" (preferred)
def _load_labels_ds():
    try:
        import pandas as pd
        # Prefer repo-relative participants files
        p1 = os.path.join("ds004504", "participants.tsv")
        p2 = os.path.join("ds003800", "participants.tsv")
        labels = {}
        if os.path.exists(p1):
            df1 = pd.read_csv(p1, sep="\t")
            # Map A->AD(1), C->Control(0). Keep F as AD if present
            mp = {"A": 1, "C": 0, "F": 1}
            df1 = df1[df1["Group"].isin(mp.keys())]
            labels.update(df1.set_index("participant_id")["Group"].map(mp).to_dict())
        if os.path.exists(p2):
            df2 = pd.read_csv(p2, sep="\t")
            # Treat ds003800 as AD(1) unless label column present
            if "Group" in df2.columns:
                mp2 = {"A": 1, "C": 0, "F": 1}
                df2 = df2[df2["Group"].isin(mp2.keys())]
                labels.update(df2.set_index("participant_id")["Group"].map(mp2).to_dict())
            else:
                labels.update(df2.set_index("participant_id").assign(v=1)["v"].to_dict())
        return labels
    except Exception:
        return {}


@st.cache_resource(show_spinner=False)
def discover_real_recordings():
    """Scan ds004504 and ds003800 for .set files and classify by label."""
    labels = _load_labels_ds()
    pools = {"healthy": [], "alz": []}
    try:
        import glob
        # Collect .set files
        roots = ["ds004504", "ds003800"]
        files = []
        for root in roots:
            if os.path.isdir(root):
                files.extend(glob.glob(os.path.join(root, "**", "*.set"), recursive=True))
        # Prefer raw dataset paths (exclude derivatives to avoid duplicates)
        files = [f for f in files if "/derivatives/" not in f]
        # Assign by subject id prefix in filename
        for f in files:
            base = os.path.basename(f)
            subj = base.split("_")[0]
            lab = labels.get(subj)
            if lab is None:
                continue
            if lab == 0:
                pools["healthy"].append(f)
            else:
                pools["alz"].append(f)
    except Exception:
        pass
    return pools


def _pick_channels(raw, prefer=("F3","F4","P3","O2")):
    picks = []
    avail = set(raw.ch_names)
    for name in prefer:
        if name in avail:
            picks.append(name)
    # If fewer than 4, fill with any EEG channels
    if len(picks) < 4:
        eegs = [ch for ch in raw.ch_names if ch.upper().startswith("F") or ch.upper().startswith("P") or ch.upper().startswith("O") or ch.upper().startswith("C")]
        for ch in eegs:
            if ch not in picks:
                picks.append(ch)
            if len(picks) >= 4:
                break
    # Final fallback: first 4 channels
    if len(picks) < 4:
        picks = raw.ch_names[:4]
    return picks[:4]


def _load_real_signal(path: str, duration_sec: int) -> np.ndarray:
    """Load a 4xN segment at 256 Hz from a .set file."""
    import mne
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    raw.filter(1.0, 50.0, fir_design="firwin", verbose=False)
    raw.resample(SAMPLING_RATE, npad="auto")
    chs = _pick_channels(raw)
    data = raw.get_data(picks=chs)
    need = duration_sec * SAMPLING_RATE
    if data.shape[1] < need:
        reps = int(np.ceil(need / data.shape[1]))
        data = np.tile(data, reps)[:, :need]
    else:
        data = data[:, :need]
    return data.astype(np.float32)


def get_real_demo_signal(condition: str, duration_sec: int) -> np.ndarray | None:
    pools = discover_real_recordings()
    candidates = pools["healthy"] if condition == "Healthy" else pools["alz"]
    if not candidates:
        return None
    path = candidates[0]
    try:
        sig = _load_real_signal(path, duration_sec)
        return sig.T
    except Exception:
        return None


def get_real_demo_chunk(condition: str, nsamp: int, t_start_sec: float, cache_key: str = "_real_chunk") -> np.ndarray | None:
    pools = discover_real_recordings()
    candidates = pools["healthy"] if condition == "Healthy" else pools["alz"]
    if not candidates:
        return None
    path = candidates[0]
    # Cache full 5 min per condition to serve chunks quickly
    key = f"{cache_key}_{'H' if condition=='Healthy' else 'A'}"
    if key not in st.session_state:
        try:
            buf = _load_real_signal(path, duration_sec=5*60)  # 5 minutes
        except Exception:
            return None
        st.session_state[key] = buf  # shape 4 x N
        st.session_state[key+"_idx"] = 0
    buf = st.session_state[key]
    idx = st.session_state.get(key+"_idx", 0)
    # If asked with t_start_sec, align index
    idx = int(t_start_sec * SAMPLING_RATE) % buf.shape[1]
    end = idx + nsamp
    if end <= buf.shape[1]:
        chunk = buf[:, idx:end]
    else:
        part1 = buf[:, idx:]
        part2 = buf[:, : end - buf.shape[1]]
        chunk = np.concatenate([part1, part2], axis=1)
    st.session_state[key+"_idx"] = (idx + nsamp) % buf.shape[1]
    return chunk
def generate_demo_chunk(condition: str, nsamp: int, t_start_sec: float = 0.0) -> np.ndarray:
    """Generate a 4xnsamp synthetic EEG chunk.
    condition: 'Healthy' or 'Alzheimer\'s'
    """
    # Prefer real recordings; fallback to synthetic if unavailable
    try:
        ch = get_real_demo_chunk(condition, nsamp, t_start_sec)
        if ch is not None:
            return ch
    except Exception:
        pass
    t = np.arange(nsamp) / SAMPLING_RATE + t_start_sec
    # Frequency bands representative tones
    alpha_f = [10.0, 9.0, 10.5, 8.5]
    theta_f = [6.0, 5.5, 6.5, 6.2]
    delta_f = [2.0, 2.5, 1.8, 2.2]
    beta_f  = [18.0, 16.0, 20.0, 22.0]

    if condition == "Alzheimer's":
        alpha_amp = 8.0
        theta_amp = 14.0
        delta_amp = 9.0
        beta_amp  = 3.0
        noise_std = 0.8
    else:  # Healthy
        alpha_amp = 20.0
        theta_amp = 4.0
        delta_amp = 2.0
        beta_amp  = 7.0
        noise_std = 1.8

    out = []
    for ch in range(NUM_CHANNELS):
        # Mild amplitude modulation
        mod = 1.0 + 0.1 * np.sin(2 * np.pi * 0.2 * t + ch)
        sig = (
            alpha_amp * np.sin(2 * np.pi * alpha_f[ch] * t) +
            theta_amp * np.sin(2 * np.pi * theta_f[ch] * t + 0.3) +
            delta_amp * np.sin(2 * np.pi * delta_f[ch] * t + 0.6) +
            beta_amp  * np.sin(2 * np.pi * beta_f[ch]  * t + 0.1)
        ) * mod
        sig += noise_std * np.random.randn(nsamp)
        out.append(sig.astype(np.float32))
    return np.stack(out, axis=0)


def generate_demo_signal(condition: str, duration_sec: int) -> np.ndarray:
    # Prefer a real recording slice; fallback to synthetic if needed
    real = get_real_demo_signal(condition, duration_sec)
    if real is not None:
        return real
    nsamp = duration_sec * SAMPLING_RATE
    chunks = [generate_demo_chunk(condition, SAMPLING_RATE, t_start_sec=float(s)) for s in range(duration_sec)]
    full = np.concatenate(chunks, axis=1)
    return full.T


def make_feature_names_30() -> list[str]:
    chs = ["F3", "F4", "P3", "O2"]
    names = []
    for c in chs:
        names.extend([
            f"{c}_alpha_ratio",
            f"{c}_spectral_entropy",
            f"{c}_hj_activity",
            f"{c}_hj_mobility",
            f"{c}_hj_complexity",
        ])
    names.extend([
        "alpha_mean","alpha_std","entropy_mean","entropy_std",
        "hj_act_mean","hj_act_std","hj_mob_mean","hj_mob_std","hj_comp_mean","hj_comp_std",
    ])
    return names


def mlp_first_layer_repr(mlp, X: np.ndarray) -> np.ndarray | None:
    try:
        if not hasattr(mlp, "coefs_"):
            return None
        W1 = mlp.coefs_[0]  # shape (n_features, n_hidden1)
        b1 = mlp.intercepts_[0]  # shape (n_hidden1,)
        z1 = X @ W1 + b1
        # activation 'logistic' in this project
        a1 = 1.0 / (1.0 + np.exp(-z1))
        return a1.squeeze(0)
    except Exception:
        return None


def qsup_probs_repr(model, X: np.ndarray) -> np.ndarray | None:
    if not TORCH_AVAILABLE or model is None:
        return None
    try:
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float)
            # Re-implement internal steps to extract 'probs'
            hidden_dim = model.hidden_dim
            eps = 1e-8
            wave_r_list = []
            wave_i_list = []
            for s in range(model.num_wavefunctions):
                out = model.wavefunction_nets[s](x_t)
                out = torch.exp(-out * out)
                alpha = out[:, :hidden_dim]
                beta  = out[:, hidden_dim:]
                norm_sq = torch.sum(alpha**2 + beta**2, dim=1, keepdim=True) + eps
                factor = torch.sqrt((model.partial_norm**2) / norm_sq)
                alpha = alpha * factor
                beta  = beta  * factor
                phase = model.phases[s].unsqueeze(0) if model.phase_per_dim else model.phases[s]
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
            interference_weights = torch.softmax(cosine_sim, dim=1).unsqueeze(2)
            sup_real = torch.sum(real_stack * interference_weights, dim=1)
            sup_imag = torch.sum(imag_stack * interference_weights, dim=1)
            if getattr(model, "gating_net", None) is not None:
                for _ in range(model.self_modulation_steps):
                    mag = torch.sqrt(sup_real**2 + sup_imag**2 + eps)
                    gate = torch.sigmoid(model.gating_net(mag))
                    sup_real = sup_real * gate
                    sup_imag = sup_imag * gate
            mag_sq = sup_real**2 + sup_imag**2
            if model.topk > 0 and model.topk < hidden_dim:
                vals, inds = torch.topk(mag_sq, model.topk, dim=1)
                mask = torch.zeros_like(mag_sq).scatter_(1, inds, 1.0)
                masked = mag_sq * mask
                sums = torch.sum(masked, dim=1, keepdim=True) + eps
                probs = masked / sums
            else:
                sums = torch.sum(mag_sq, dim=1, keepdim=True) + eps
                probs = mag_sq / sums
        return probs.cpu().numpy().squeeze(0)
    except Exception:
        return None


# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    mlp_model = None
    pca_model = None
    gcn_model = None
    qsup_model = None

    if joblib is not None and os.path.exists(MLP_MODEL_PATH):
        try:
            mlp_model = joblib.load(MLP_MODEL_PATH)
        except Exception as e:
            st.warning(f"Failed to load MLP model: {e}")

    if joblib is not None and os.path.exists(PCA_MODEL_PATH):
        try:
            pca_model = joblib.load(PCA_MODEL_PATH)
        except Exception as e:
            st.warning(f"Failed to load PCA model: {e}")

    if TORCH_AVAILABLE and PYG_AVAILABLE and os.path.exists(GCN_MODEL_PATH):
        try:
            model = GCNNet(in_channels=9, hidden_channels=32, num_classes=2)
            sd = torch.load(GCN_MODEL_PATH, map_location="cpu")
            model.load_state_dict(sd, strict=False)
            model.eval()
            gcn_model = model
        except Exception as e:
            st.warning(f"Failed to load GCN model: {e}")

    if TORCH_AVAILABLE and os.path.exists(QSUP_MODEL_PATH):
        try:
            model = ExtendedQSUP(
                input_dim=62,
                hidden_dim=32,
                num_classes=2,
                num_wavefunctions=6,
                partial_norm=1.5,
                phase_per_dim=True,
                self_modulation_steps=2,
                topk=8,
            )
            sd = torch.load(QSUP_MODEL_PATH, map_location="cpu")
            model.load_state_dict(sd, strict=True)
            model.eval()
            qsup_model = model
        except Exception as e:
            st.warning(f"Failed to load QSUP model: {e}")

    return mlp_model, pca_model, gcn_model, qsup_model


# Optional: preprocessors (scalers) for MLP/QSUP if available
@st.cache_resource(show_spinner=False)
def load_preprocessors():
    mlp_scaler = None
    qsup_scaler = None
    if joblib is not None:
        for name in ("mlp_scaler.joblib", "scaler_mlp.joblib"):
            p = os.path.join(MODELS_DIR, name)
            if os.path.exists(p):
                try:
                    mlp_scaler = joblib.load(p)
                    break
                except Exception:
                    pass
        for name in ("qsup_scaler.joblib", "scaler_qsup.joblib"):
            p = os.path.join(MODELS_DIR, name)
            if os.path.exists(p):
                try:
                    qsup_scaler = joblib.load(p)
                    break
                except Exception:
                    pass
    return mlp_scaler, qsup_scaler


def adaptive_scale(x: np.ndarray, hist: list[np.ndarray]) -> np.ndarray:
    """Scale 1xD vector using history statistics if available, else L2 norm.
    x: shape (1, D). hist: list of arrays of shape (D,)
    """
    eps = 1e-8
    if len(hist) >= 10:
        H = np.vstack(hist[-200:])  # last 200 samples
        mu = H.mean(axis=0)
        std = H.std(axis=0)
        std[std < 1e-6] = 1.0
        return (x - mu) / std
    # Fallback: L2 normalize
    norm = np.linalg.norm(x)
    return x / (norm + eps)


def ensure_4ch_20s(signal_arr: np.ndarray) -> np.ndarray:
    arr = np.array(signal_arr, dtype=np.float32)
    # Accept (n_samples, 4) or (4, n_samples)
    if arr.ndim != 2:
        raise ValueError("Expected 2D array for EEG: (n_samples, 4) or (4, n_samples)")
    if arr.shape[1] == 4:
        arr = arr.T
    if arr.shape[0] != 4:
        raise ValueError("Expected 4 channels")
    # Resample/pad/trim to 5120 samples
    if arr.shape[1] < WINDOW_SAMPLES:
        pad = WINDOW_SAMPLES - arr.shape[1]
        arr = np.pad(arr, ((0, 0), (0, pad)), mode="edge")
    elif arr.shape[1] > WINDOW_SAMPLES:
        arr = arr[:, -WINDOW_SAMPLES:]
    return arr


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AlzDetect — EEG Alzheimer Detection by Ishaan Gubbala", page_icon="🧠", layout="wide")

# SEO and branding injection
def inject_seo():
    deploy_url = os.environ.get("ALZDETECT_URL", "")
    og_image = os.environ.get("ALZDETECT_OG_IMAGE", "")
    favicon = os.environ.get("ALZDETECT_FAVICON", "")
    meta = f"""
    <title>AlzDetect — EEG Alzheimer Detection | Ishaan Gubbala</title>
    <meta name="description" content="AlzDetect by Ishaan Gubbala and Ayaan Khan — real‑time EEG feature extraction and Alzheimer’s detection with MLP, QSUP, and GCN embeddings. Live charts, smooth UX, and research‑grade pipeline."/>
    <meta name="keywords" content="AlzDetect, Ishaan Gubbala, Ayaan Khan, EEG, Alzheimer, MCI, dementia, neural network, QSUP, GCN, Streamlit, neuroscience"/>
    <meta name="author" content="Ishaan Gubbala, Ayaan Khan"/>
    <meta property="og:title" content="AlzDetect — EEG Alzheimer Detection"/>
    <meta property="og:description" content="Real‑time EEG analysis and Alzheimer’s detection with modern ML."/>
    <meta property="og:type" content="website"/>
    <meta property="og:url" content="{deploy_url}"/>
    {f'<meta property="og:image" content="{og_image}"/>' if og_image else ''}
    <meta name="twitter:card" content="summary_large_image"/>
    <meta name="twitter:title" content="AlzDetect — EEG Alzheimer Detection"/>
    <meta name="twitter:description" content="EEG features + QSUP/MLP models with live visualization."/>
    {f'<link rel="icon" href="{favicon}">"' if favicon else ''}
    <script type="application/ld+json">
    {{
      "@context": "https://schema.org",
      "@type": "SoftwareApplication",
      "name": "AlzDetect",
      "applicationCategory": "MedicalApplication",
      "description": "EEG Alzheimer detection with ML (QSUP/MLP) and live charts.",
      "author": [
        {{"@type": "Person", "name": "Ishaan Gubbala"}},
        {{"@type": "Person", "name": "Ayaan Khan"}}
      ],
      "url": "{deploy_url}"
    }}
    </script>
    """
    st.markdown(meta, unsafe_allow_html=True)

def inject_css(theme: str):
    if theme == "aurora":
        bg = """
        background: radial-gradient(1000px 500px at 10% 10%, rgba(255,99,132,0.10), transparent 50%),
                    radial-gradient(1000px 500px at 90% 0%, rgba(54,162,235,0.10), transparent 50%),
                    linear-gradient(180deg, #0b1220 0%, #0a1730 60%, #081a3a 100%);
        color: #eaf1ff;
        """
        grad_text = "background: linear-gradient(90deg,#ff7aa2 0%, #60a5fa 50%, #34d399 100%);"
        card_bg = "background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.10);"
        primary_btn = "background: linear-gradient(90deg,#7c3aed,#06b6d4);"
    elif theme == "mono":
        bg = """
        background: linear-gradient(180deg, #0e0e10 0%, #0e0e10 100%);
        color: #e5e5e5;
        """
        grad_text = "background: linear-gradient(90deg,#d1d5db 0%, #9ca3af 100%);"
        card_bg = "background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);"
        primary_btn = "background: linear-gradient(90deg,#4b5563,#9ca3af);"
    else:  # dark default
        bg = """
        background: radial-gradient(1200px 500px at 10% 10%, rgba(120,61,255,0.08), transparent 50%),
                    radial-gradient(1200px 500px at 90% 0%, rgba(0,204,255,0.08), transparent 50%),
                    linear-gradient(180deg, #0b0f1a 0%, #0b0f1a 60%, #0b0f1a 100%);
        color: #e9edf1;
        """
        grad_text = "background: linear-gradient(90deg,#9f7aea 0%, #22d3ee 50%, #38bdf8 100%);"
        card_bg = "background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);"
        primary_btn = "background: linear-gradient(90deg,#7c3aed,#06b6d4);"

    css = f"""
    <style>
    html, body, [class*="css"]  {{ scroll-behavior: smooth; }}
    .main {{ padding-top: 1rem; }}
    .stApp {{ {bg} }}
    .alz-hero {{ padding: 0.5rem 0 1rem 0; }}
    .alz-title {{ font-size: 2.2rem; font-weight: 800; letter-spacing: 0.3px; {grad_text}
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .alz-sub  {{ color: #b6c2d0; font-size: 0.95rem; }}
    .alz-card {{ {card_bg} border-radius: 14px; padding: 1rem; box-shadow: 0 6px 18px rgba(0,0,0,0.25); }}
    .stButton>button {{ border-radius: 10px; padding: 0.6rem 1rem; font-weight: 600; }}
    .stButton>button[kind="primary"] {{ {primary_btn} border: 0; }}
    .stButton>button:hover {{ filter: brightness(1.08); }}
    .stMetric {{ background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px; padding: .5rem; }}
    .alz-footer {{ color:#9fb0c3; font-size:.85rem; margin-top:.5rem; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_seo()
# Theme selection persistence via query params
params = st.query_params
default_theme = params.get("theme") or "dark"
if "theme" not in st.session_state:
    st.session_state.theme = default_theme if default_theme in ("dark","aurora","mono") else "dark"
inject_css(st.session_state.theme)

# Header
st.markdown("""
<div class="alz-hero">
  <div class="alz-title">AlzDetect — EEG Alzheimer Detection</div>
  <div class="alz-sub">Real‑time EEG features, QSUP/MLP models, live visualization, and a research‑grade pipeline.</div>
</div>
""", unsafe_allow_html=True)
st.caption("Made by Ishaan Gubbala and Ayaan Khan")

# Header theme toggle (mirrors sidebar, persists via query param)
ht_cols = st.columns([6,2])
with ht_cols[1]:
    sel = st.selectbox(
        "Theme",
        options=["dark","aurora","mono"],
        index=["dark","aurora","mono"].index(st.session_state.theme),
        key="theme_header",
        label_visibility="collapsed",
    )
    if sel != st.session_state.theme:
        st.session_state.theme = sel
        st.query_params["theme"] = sel

def _nav_to(target_page: str, mode: str|None=None):
    st.session_state["page_selector"] = target_page
    if mode is not None:
        st.session_state["input_mode"] = mode
    # persist theme in URL
    st.query_params["theme"] = st.session_state.get("theme","dark")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Home", "Inference", "Training", "Plots", "About"],
        index=0,
        key="page_selector",
    )
    st.selectbox(
        "Theme",
        options=["dark","aurora","mono"],
        format_func=lambda k: {"dark":"Dark","aurora":"Aurora","mono":"Mono"}[k],
        key="theme",
        on_change=lambda: st.query_params.__setitem__("theme", st.session_state.get("theme","dark"))
    )
    st.markdown("---")
    st.caption("Models loaded from `trained_models/`")

mlp_model, pca_model, gcn_model, qsup_model = load_models()
mlp_scaler, qsup_scaler = load_preprocessors()

if page == "Home":
    st.subheader("Welcome to AlzDetect")
    st.markdown("Use the cards below to jump into key experiences.")

    # Compact KPI bar
    k1, k2, k3, k4 = st.columns(4)
    # Model availability
    avail = {
        "MLP": bool(mlp_model),
        "QSUP": bool(qsup_model),
        "GCN": bool(gcn_model),
        "PCA": bool(pca_model),
    }
    with k1:
        st.metric("Models", f"{sum(avail.values())}/4", delta=", ".join([k for k,v in avail.items() if v]) or "none")
    # Device info
    device_label = "CPU"
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            device_label = torch.cuda.get_device_name(0)
        except Exception:
            device_label = "CUDA GPU"
    with k2:
        st.metric("Compute", device_label)
    # Last trained date from latest file in trained_models
    last_trained = "n/a"
    if os.path.isdir(MODELS_DIR):
        try:
            files = [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR)]
            files = [f for f in files if os.path.isfile(f)]
            if files:
                latest = max(files, key=os.path.getmtime)
                last_trained = datetime.fromtimestamp(os.path.getmtime(latest)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
    with k3:
        st.metric("Last Trained", last_trained)
    # App version/simple tag
    with k4:
        st.metric("AlzDetect", "v1")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='alz-card'><h4>Try Inference</h4><p>Upload EEG (CSV/NPY) and get predictions.</p></div>", unsafe_allow_html=True)
        st.button("Go to Inference", key="cta_inf", type="primary", use_container_width=True, on_click=_nav_to, args=("Inference",))
    with c2:
        st.markdown("<div class='alz-card'><h4>Live Demo</h4><p>Stream synthetic EEG and watch live predictions.</p></div>", unsafe_allow_html=True)
        st.button("Start Live Demo", key="cta_live", type="primary", use_container_width=True, on_click=_nav_to, args=("Inference","Live demo"))
    with c3:
        st.markdown("<div class='alz-card'><h4>See Plots</h4><p>Explore training ROC and loss figures.</p></div>", unsafe_allow_html=True)
        st.button("Open Plots", key="cta_plots", use_container_width=True, on_click=_nav_to, args=("Plots",))

if page == "Inference":
    st.subheader("Upload 4‑channel EEG, use demo, or run live mode")
    cols = st.columns([2, 1])

    with cols[0]:
        mode = st.radio("Input mode", ["Upload", "Demo signal", "Live demo", "Live playback"], horizontal=True, key="input_mode")
        eeg_array = None

        if mode == "Upload":
            up = st.file_uploader("Upload CSV or NPY with 4 channels", type=["csv", "npy"])
            if up is not None:
                if up.name.endswith(".csv"):
                    try:
                        import pandas as pd
                        df = pd.read_csv(up)
                        eeg_array = df.values.astype(np.float32)
                    except Exception:
                        # Fallback to numpy loadtxt
                        up.seek(0)
                        eeg_array = np.loadtxt(up, delimiter=",").astype(np.float32)
                else:
                    data = up.read()
                    by = io.BytesIO(data)
                    eeg_array = np.load(by)
        elif mode == "Demo signal":
            dur = st.slider("Demo duration (seconds)", min_value=5, max_value=600, value=60, step=5)
            cond = st.radio("Simulated condition", ["Healthy", "Alzheimer's"], horizontal=True)
            # Show which subject is used (for debugging/traceability)
            pools = discover_real_recordings()
            cand = (pools["healthy"] if cond == "Healthy" else pools["alz"]) or []
            if cand:
                subj = os.path.basename(cand[0]).split('_')[0]
                st.caption(f"Simulated source: {subj}")
            eeg_array = generate_demo_signal(cond, dur)
        elif mode == "Live demo":
            # Initialize / controls
            if "live_running" not in st.session_state:
                st.session_state.live_running = False
                st.session_state.live_buf = np.zeros((NUM_CHANNELS, 0), dtype=np.float32)
                st.session_state.t_idx = 0
                st.session_state.conf_hist = []
                st.session_state.last_tick = 0.0
            cond = st.radio("Simulated condition", ["Healthy", "Alzheimer's"], horizontal=True)
            pools = discover_real_recordings()
            cand = (pools["healthy"] if cond == "Healthy" else pools["alz"]) or []
            if cand:
                subj = os.path.basename(cand[0]).split('_')[0]
                st.caption(f"Simulated source: {subj}")
            dur_total = st.slider("Total live duration (seconds)", 10, 900, 120, step=10)
            view_secs = st.slider("Chart window (seconds)", 5, 60, 20, step=5)
            plot_rate = st.slider("Plot rate (Hz)", 16, 128, 64, step=16)
            start = st.button("Start live demo", disabled=st.session_state.live_running)
            stop = st.button("Stop live demo")
            chart_ph = st.empty()
            pred_ph = st.empty()
            conf_chart_ph = st.empty()
            last_update_ph = st.empty()
            if stop and st.session_state.live_running:
                st.session_state.live_running = False
            if start:
                st.session_state.live_running = True
                st.session_state.live_buf = np.zeros((NUM_CHANNELS, 0), dtype=np.float32)
                st.session_state.t_idx = 0
                st.session_state.conf_hist = []
                st.session_state.last_tick = 0.0

            # Blocking loop with controlled plotting and computation
            if st.session_state.live_running:
                try:
                    for _ in range(dur_total):
                        nsamp = SAMPLING_RATE
                        t0_sec = (st.session_state.t_idx / SAMPLING_RATE)
                        chunk = generate_demo_chunk(cond, nsamp, t_start_sec=t0_sec)
                        st.session_state.live_buf = np.concatenate((st.session_state.live_buf, chunk), axis=1)
                        st.session_state.t_idx += nsamp

                        # Plot last view window with decimation
                        seg = st.session_state.live_buf[:, -view_secs*SAMPLING_RATE:]
                        if seg.size > 0:
                            step = max(1, SAMPLING_RATE // plot_rate)
                            seg_ds = seg[:, ::step]
                            import pandas as pd
                            df = pd.DataFrame(seg_ds.T, columns=["F3", "F4", "P3", "O2"])
                            chart_ph.line_chart(df, height=250)

                        # Predict after warmup
                        if st.session_state.live_buf.shape[1] >= WINDOW_SAMPLES:
                            last = st.session_state.live_buf[:, -WINDOW_SAMPLES:]
                            gf_30 = aggregator_30(last).reshape(1, 30)
                            node_feats = channelwise_9feats(last)
                            gcn_emb = np.zeros((1, 32), dtype=np.float32)
                            if TORCH_AVAILABLE and gcn_model is not None:
                                try:
                                    A = adjacency_4ch()
                                    edge_idx = np.array(np.nonzero(A))
                                    edge_idx = torch.tensor(edge_idx, dtype=torch.long)
                                    x_tensor = torch.tensor(node_feats, dtype=torch.float)
                                    batch = torch.zeros(x_tensor.shape[0], dtype=torch.long)
                                    with torch.no_grad():
                                        emb = gcn_model.embed(x_tensor, edge_index=edge_idx, batch=batch)
                                        gcn_emb = emb.cpu().numpy().reshape(1, -1)
                                except Exception:
                                    pass
                        full_vec = np.hstack([gf_30, gcn_emb])
                        # Maintain CCV history for adaptive scaling
                        if "ccv_hist" not in st.session_state:
                            st.session_state.ccv_hist = []
                        st.session_state.ccv_hist.append(full_vec.flatten())
                        if len(st.session_state.ccv_hist) > 500:
                            st.session_state.ccv_hist = st.session_state.ccv_hist[-500:]
                            pred_text = ""
                            mlp_conf_val = None
                            qsup_conf_val = None
                        if mlp_model is not None:
                            try:
                                x_in = full_vec
                                if mlp_scaler is not None:
                                    x_in = mlp_scaler.transform(x_in)
                                else:
                                    x_in = adaptive_scale(x_in, st.session_state.ccv_hist)
                                proba = mlp_model.predict_proba(x_in)[0]
                                mlp_conf_val = float(proba[1])
                                pred = int(mlp_conf_val >= 0.5)
                                pred_text += f"MLP: {pred} (conf {mlp_conf_val:.2f})  "
                            except Exception:
                                pass
                        if TORCH_AVAILABLE and qsup_model is not None:
                            try:
                                with torch.no_grad():
                                    x_in = full_vec
                                    if qsup_scaler is not None:
                                        x_in = qsup_scaler.transform(x_in)
                                    else:
                                        x_in = adaptive_scale(x_in, st.session_state.ccv_hist)
                                    logits = qsup_model(torch.tensor(x_in, dtype=torch.float))
                                    p = torch.softmax(logits, dim=1).cpu().numpy()[0]
                                    qsup_conf_val = float(p[1])
                                    pred = int(qsup_conf_val >= 0.5)
                                    pred_text += f"QSUP: {pred} (conf {qsup_conf_val:.2f})  "
                            except Exception:
                                pass
                            pred_text += f"Energy: {compute_energy(gf_30[0]):.1f}"
                            pred_ph.info(pred_text)
                            t_sec = st.session_state.t_idx / SAMPLING_RATE
                            st.session_state.conf_hist.append({
                                "t": t_sec,
                                "MLP": mlp_conf_val if mlp_conf_val is not None else np.nan,
                                "QSUP": qsup_conf_val if qsup_conf_val is not None else np.nan,
                            })
                            if len(st.session_state.conf_hist) > 600:
                                st.session_state.conf_hist = st.session_state.conf_hist[-600:]
                            try:
                                import pandas as pd
                                dfh = pd.DataFrame(st.session_state.conf_hist).set_index("t")
                                conf_chart_ph.line_chart(dfh, height=200)
                            except Exception:
                                pass
                            last_update_ph.caption(f"Last prediction at t={t_sec:.1f}s")

                        time.sleep(1.0)
                except Exception as e:
                    st.error(f"Live demo encountered an error: {e}")
                finally:
                    st.session_state.live_running = False

        elif mode == "Live playback":
            up = st.file_uploader("Upload CSV/NPY to play back live", type=["csv", "npy"])
            view_secs = st.slider("Chart window (seconds)", 5, 60, 20, step=5)
            plot_rate = st.slider("Plot rate (Hz)", 16, 128, 64, step=16)
            start = st.button("Start playback")
            stop = st.button("Stop playback")
            chart_ph = st.empty()
            pred_ph = st.empty()
            conf_chart_ph = st.empty()
            last_update_ph = st.empty()
            if "play_running" not in st.session_state:
                st.session_state.play_running = False
                st.session_state.play_buf = np.zeros((NUM_CHANNELS, 0), dtype=np.float32)
                st.session_state.play_idx = 0
                st.session_state.play_conf_hist = []
                st.session_state.play_last_tick = 0.0
                st.session_state.play_arr = None
            if stop and st.session_state.play_running:
                st.session_state.play_running = False
            if start and up is not None:
                # load file into session
                if up.name.endswith(".csv"):
                    import pandas as pd
                    df = pd.read_csv(up)
                    arr = df.values.astype(np.float32)
                else:
                    arr = np.load(io.BytesIO(up.read()))
                if arr.shape[1] == 4:
                    arr = arr.T
                if arr.shape[0] != 4:
                    st.error("Expected 4 channels in file")
                else:
                    st.session_state.play_arr = arr
                    st.session_state.play_running = True
                    st.session_state.play_buf = np.zeros((NUM_CHANNELS, 0), dtype=np.float32)
                    st.session_state.play_idx = 0
                    st.session_state.play_conf_hist = []
                    st.session_state.play_last_tick = 0.0

            if st.session_state.play_running and st.session_state.play_arr is not None:
                now = time.time()
                # if more data remains and tick elapsed
                total_secs = st.session_state.play_arr.shape[1] // SAMPLING_RATE
                if st.session_state.play_idx >= total_secs:
                    st.session_state.play_running = False
                elif now - st.session_state.play_last_tick >= 1.0:
                    s = st.session_state.play_idx
                    chunk = st.session_state.play_arr[:, s*SAMPLING_RATE:(s+1)*SAMPLING_RATE]
                    st.session_state.play_buf = np.concatenate((st.session_state.play_buf, chunk), axis=1)
                    st.session_state.play_idx += 1
                    st.session_state.play_last_tick = now

                seg = st.session_state.play_buf[:, -view_secs*SAMPLING_RATE:]
                if seg.size > 0:
                    step = max(1, SAMPLING_RATE // plot_rate)
                    seg_ds = seg[:, ::step]
                    import pandas as pd
                    dfc = pd.DataFrame(seg_ds.T, columns=["F3", "F4", "P3", "O2"])  # example labels
                    chart_ph.line_chart(dfc, height=250)
                if st.session_state.play_buf.shape[1] >= WINDOW_SAMPLES:
                    last = st.session_state.play_buf[:, -WINDOW_SAMPLES:]
                    gf_30 = aggregator_30(last).reshape(1, 30)
                    node_feats = channelwise_9feats(last)
                    gcn_emb = np.zeros((1, 32), dtype=np.float32)
                    if TORCH_AVAILABLE and gcn_model is not None:
                        try:
                            A = adjacency_4ch()
                            edge_idx = np.array(np.nonzero(A))
                            edge_idx = torch.tensor(edge_idx, dtype=torch.long)
                            x_tensor = torch.tensor(node_feats, dtype=torch.float)
                            batch = torch.zeros(x_tensor.shape[0], dtype=torch.long)
                            with torch.no_grad():
                                emb = gcn_model.embed(x_tensor, edge_index=edge_idx, batch=batch)
                                gcn_emb = emb.cpu().numpy().reshape(1, -1)
                        except Exception:
                            pass
                    full_vec = np.hstack([gf_30, gcn_emb])
                    if "ccv_hist" not in st.session_state:
                        st.session_state.ccv_hist = []
                    st.session_state.ccv_hist.append(full_vec.flatten())
                    if len(st.session_state.ccv_hist) > 500:
                        st.session_state.ccv_hist = st.session_state.ccv_hist[-500:]
                    pred_text = ""
                    mlp_conf_val = None
                    qsup_conf_val = None
                    if mlp_model is not None:
                        try:
                            x_in = full_vec
                            if mlp_scaler is not None:
                                x_in = mlp_scaler.transform(x_in)
                            else:
                                x_in = adaptive_scale(x_in, st.session_state.ccv_hist)
                            proba = mlp_model.predict_proba(x_in)[0]
                            mlp_conf_val = float(proba[1])
                            pred = int(mlp_conf_val >= 0.5)
                            pred_text += f"MLP: {pred} (conf {mlp_conf_val:.2f})  "
                        except Exception:
                            pass
                    if TORCH_AVAILABLE and qsup_model is not None:
                        try:
                            with torch.no_grad():
                                x_in = full_vec
                                if qsup_scaler is not None:
                                    x_in = qsup_scaler.transform(x_in)
                                else:
                                    x_in = adaptive_scale(x_in, st.session_state.ccv_hist)
                                logits = qsup_model(torch.tensor(x_in, dtype=torch.float))
                                p = torch.softmax(logits, dim=1).cpu().numpy()[0]
                                qsup_conf_val = float(p[1])
                                pred = int(qsup_conf_val >= 0.5)
                                pred_text += f"QSUP: {pred} (conf {qsup_conf_val:.2f})  "
                        except Exception:
                            pass
                    pred_text += f"Energy: {compute_energy(gf_30[0]):.1f}"
                    pred_ph.info(pred_text)
                    t_sec = st.session_state.play_idx
                    st.session_state.play_conf_hist.append({
                        "t": t_sec,
                        "MLP": mlp_conf_val if mlp_conf_val is not None else np.nan,
                        "QSUP": qsup_conf_val if qsup_conf_val is not None else np.nan,
                    })
                    if len(st.session_state.play_conf_hist) > 600:
                        st.session_state.play_conf_hist = st.session_state.play_conf_hist[-600:]
                    try:
                        import pandas as pd
                        dfh = pd.DataFrame(st.session_state.play_conf_hist).set_index("t")
                        conf_chart_ph.line_chart(dfh, height=200)
                    except Exception:
                        pass
                    last_update_ph.caption(f"Last prediction at t={t_sec:.1f}s")

                # End of playback loop: no rerun, script completes naturally

        run = st.button("Compute features and predict", use_container_width=True, type="primary")

    with cols[1]:
        st.markdown("**Model availability**")
        st.write({
            "MLP": bool(mlp_model),
            "QSUP": bool(qsup_model),
            "GCN": bool(gcn_model),
            "PCA": bool(pca_model),
            "SciPy": SCIPY_AVAILABLE,
            "Torch": TORCH_AVAILABLE,
        })

    if run:
        if eeg_array is None:
            st.error("Please provide input data first.")
        else:
            try:
                chunk = ensure_4ch_20s(eeg_array)
            except Exception as e:
                st.error(str(e))
                st.stop()

            with st.spinner("Extracting features..."):
                try:
                    gf_30 = aggregator_30(chunk).reshape(1, 30)
                    node_feats = channelwise_9feats(chunk)
                except Exception as e:
                    st.error(f"Feature extraction failed: {e}")
                    st.stop()

            # Build GCN embedding if available
            gcn_emb = np.zeros((1, 32), dtype=np.float32)
            if TORCH_AVAILABLE and gcn_model is not None:
                try:
                    A = adjacency_4ch()
                    edge_idx = np.array(np.nonzero(A))
                    edge_idx = torch.tensor(edge_idx, dtype=torch.long)
                    x_tensor = torch.tensor(node_feats, dtype=torch.float)
                    batch = torch.zeros(x_tensor.shape[0], dtype=torch.long)
                    with torch.no_grad():
                        emb = gcn_model.embed(x_tensor, edge_idx, batch)
                        gcn_emb = emb.cpu().numpy().reshape(1, -1)
                except Exception as e:
                    st.warning(f"GCN embedding failed; using zeros. Reason: {e}")

            full_vec = np.hstack([gf_30, gcn_emb])  # (1, 62)

            # Maintain CCV history for adaptive scaling
            if "ccv_hist" not in st.session_state:
                st.session_state.ccv_hist = []
            st.session_state.ccv_hist.append(full_vec.flatten())
            if len(st.session_state.ccv_hist) > 500:
                st.session_state.ccv_hist = st.session_state.ccv_hist[-500:]

            # PCA for visualization (optional)
            pc1, pc2 = 0.0, 0.0
            if pca_model is not None:
                try:
                    pc = pca_model.transform(full_vec)[0]
                    pc1, pc2 = float(pc[0]), float(pc[1])
                except Exception:
                    pass

            # Inference
            pred_mlp = conf_mlp = None
            if mlp_model is not None:
                try:
                    x_in = full_vec
                    if mlp_scaler is not None:
                        x_in = mlp_scaler.transform(x_in)
                    else:
                        x_in = adaptive_scale(x_in, st.session_state.ccv_hist)
                    proba = mlp_model.predict_proba(x_in)[0]
                    conf_mlp = float(proba[1])
                    pred_mlp = int(conf_mlp >= 0.5)
                except Exception as e:
                    st.warning(f"MLP inference failed: {e}")

            pred_qsup = conf_qsup = None
            if TORCH_AVAILABLE and qsup_model is not None:
                try:
                    with torch.no_grad():
                        x_in = full_vec
                        if qsup_scaler is not None:
                            x_in = qsup_scaler.transform(x_in)
                        else:
                            x_in = adaptive_scale(x_in, st.session_state.ccv_hist)
                        logits = qsup_model(torch.tensor(x_in, dtype=torch.float))
                        p = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        conf_qsup = float(p[1])
                        pred_qsup = int(conf_qsup >= 0.5)
                except Exception as e:
                    st.warning(f"QSUP inference failed: {e}")

            # Display
            st.success("Prediction complete")
            met_cols = st.columns(3)
            with met_cols[0]:
                st.metric("MLP (1=AD)", value=str(pred_mlp) if pred_mlp is not None else "-",
                          delta=f"conf {conf_mlp:.2f}" if conf_mlp is not None else "")
            with met_cols[1]:
                st.metric("QSUP (1=AD)", value=str(pred_qsup) if pred_qsup is not None else "-",
                          delta=f"conf {conf_qsup:.2f}" if conf_qsup is not None else "")
            with met_cols[2]:
                st.metric("Energy", value=f"{compute_energy(gf_30[0]):.1f}")

            with st.expander("Feature values (30 agg) and CCV (62)"):
                try:
                    import pandas as pd
                    names30 = make_feature_names_30()
                    df30 = pd.DataFrame([gf_30[0]], columns=names30)
                    st.dataframe(df30, use_container_width=True)
                except Exception:
                    st.write(gf_30.tolist())
                st.caption("CCV (62 dims): 30 aggregator + 32 GCN (zeros if GCN unavailable)")
                st.bar_chart(full_vec.flatten(), height=160)

            # Model representations
            rep_cols = st.columns(2)
            with rep_cols[0]:
                st.markdown("**MLP first-layer activations**")
                a1 = mlp_first_layer_repr(mlp_model, full_vec) if mlp_model is not None else None
                if a1 is not None:
                    st.bar_chart(a1, height=160)
                else:
                    st.caption("Not available.")
            with rep_cols[1]:
                st.markdown("**QSUP internal probabilities (probs)**")
                qprob = qsup_probs_repr(qsup_model, full_vec) if (TORCH_AVAILABLE and qsup_model is not None) else None
                if qprob is not None:
                    st.bar_chart(qprob, height=160)
                else:
                    st.caption("Not available.")

            st.caption("PC1/PC2 (if PCA available): %.3f, %.3f" % (pc1, pc2))

elif page == "Training":
    st.subheader("Run training on saved features (heavy)")
    st.caption("Uses processed_features/* to train GCN embeddings, MLP and QSUP. Saves outputs to trained_models/ and plots/.")
    warn = st.warning("Training can take significant time and requires all dependencies (torch, torch_geometric, imblearn, mne, etc.).", icon="⚠️")
    do_train = st.button("Start training (ai_model.main)", type="primary")
    if do_train:
        start = time.time()
        with st.spinner("Training pipeline running..."):
            try:
                import ai_model  # local module
                ai_model.main()
                st.success("Training completed.")
            except Exception as e:
                st.error(f"Training failed: {e}")
        st.caption(f"Elapsed: {time.time() - start:.1f}s")

elif page == "Plots":
    st.subheader("Saved plots")
    if not os.path.isdir(PLOTS_DIR):
        st.info("plots/ directory not found.")
    else:
        imgs = [f for f in os.listdir(PLOTS_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not imgs:
            st.info("No plots found in plots/.")
        else:
            for fn in sorted(imgs):
                st.image(os.path.join(PLOTS_DIR, fn), caption=fn, use_container_width=True)

elif page == "About":
    st.subheader("About this app")
    st.markdown(
        """
        - Inference: Upload a 4‑channel EEG segment (CSV or NPY). AlzDetect computes 30 aggregator features and, if available, a 32‑D GCN embedding. Predictions use trained MLP and QSUP models.
        - Training: Runs `ai_model.py` on features under `processed_features/`, saving checkpoints to `trained_models/` and plots to `plots/`.
        - Plots: Displays PNG/JPG figures in `plots/`.
        """
    )
    st.markdown("**Overview**")
    st.markdown(
        """
        AlzDetect by Ishaan Gubbala and Ayaan Khan is a modern EEG analysis application for Alzheimer’s detection. It blends handcrafted EEG features, graph‑based embeddings (GCN), and a quantum‑inspired neural network (QSUP) for robust classification. Designed for real‑time inference and smooth UX, AlzDetect offers:
        - Real‑time sliding‑window predictions with live confidence charts.
        - Research‑grade feature pipeline with Hjorth parameters, spectral entropy, and band powers.
        - Cross‑validation training flows with saved plots and model checkpoints.
        """
    )
    st.markdown("**Results**")
    st.markdown(
        """
        On internal experiments with pre‑extracted features, the MLP and QSUP models achieve competitive accuracy in distinguishing Alzheimer’s vs. Control subjects. Performance depends on dataset composition and preprocessing; use the Training tab to reproduce your own metrics. The Plots tab renders per‑fold ROC curves and loss trajectories.
        
        Keywords: AlzDetect, Ishaan Gubbala, EEG Alzheimer detection, QSUP, GCN, MLP, dementia, neural signal processing.
        """
    )
    st.caption("Made by Ishaan Gubbala and Ayaan Khan")
