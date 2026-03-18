"""
server.py

A FastAPI WebSocket server that:
 • Receives 4-channel cleaned EEG data (from the client) in 1-second chunks.
 • Maintains a 20-second rolling buffer (5120 samples at 256Hz).
 • Computes 30 aggregator features + 42 extra GNN features + 64 GCN embeddings = 136-dim CCV.
 • Feeds the CCV into pre-trained MLP and QSUP models to produce predictions.
 • Returns a JSON response with predictions, aggregator features, PCA (pc1, pc2) and an energy measure.
 • Sends a static 3D energy landscape once upon client connection.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations
import joblib
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any
import uvicorn
import scipy.signal

# Model file paths
MODELS_DIR = "trained_models"
GCN_MODEL_PATH  = os.path.join(MODELS_DIR, "gcn_model.pth")
MLP_MODEL_PATH  = os.path.join(MODELS_DIR, "mlp_model.pkl")
QSUP_MODEL_PATH = os.path.join(MODELS_DIR, "qsup_model.pth")
PCA_MODEL_PATH  = os.path.join(MODELS_DIR, "pca_model.joblib")

NUM_CHANNELS   = 4
SAMPLING_RATE  = 256
WINDOW_SECONDS = 20
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SECONDS  # 5120 samples

FREQUENCY_BANDS_9 = {
    "Delta":  (0.5, 4),
    "Theta1": (4, 6),
    "Theta2": (6, 8),
    "Alpha1": (8,10),
    "Alpha2": (10,12),
    "Beta1":  (12,18),
    "Beta2":  (18,22),
    "Beta3":  (22,26),
    "Beta4":  (26,30)
}

# Pydantic models
class EEGSample(BaseModel):
    channels: List[float]

class EEGData(BaseModel):
    data: List[EEGSample]

class PredictionResponse(BaseModel):
    prediction_mlp: int
    confidence_mlp: float
    prediction_qsup: int
    confidence_qsup: float
    prediction_gcn: int
    confidence_gcn: float
    features: Dict[str, float]
    stats: Dict[str, Dict[str, float]]
    pc1: float
    pc2: float
    energy: float

class EnergyLandscape(BaseModel):
    energy_landscape: Dict[str, List[float]]

# ─── Extra engineered features from band-power matrix ───────────────────────
def extract_extra_features_from_gnn(gnn_feat: np.ndarray) -> np.ndarray:
    """
    42-dim channel-aggregated features from (N_ch, 9) band-power matrix.
    Works for any N_ch (4 during live inference, 19 during training).
    Band order: Delta(0), Theta1(1), Theta2(2), Alpha1(3), Alpha2(4),
                Beta1(5), Beta2(6), Gamma1(7), Gamma2(8)
    """
    eps = 1e-12
    band_means = np.mean(gnn_feat, axis=0).astype(np.float64)
    band_stds  = np.std(gnn_feat,  axis=0).astype(np.float64)
    log_means  = np.log1p(np.clip(band_means, 0, None))
    total_mean = float(np.sum(band_means)) + eps
    norm_bands = band_means / total_mean
    delta = band_means[0];  theta = band_means[1] + band_means[2]
    alpha = band_means[3] + band_means[4]
    beta  = band_means[5] + band_means[6]; gamma = band_means[7] + band_means[8]
    ratios = np.array([
        theta / (alpha + eps), alpha / (beta + eps), delta / (alpha + eps),
        (alpha + theta) / (total_mean + eps), delta / (total_mean + eps),
        gamma / (beta + eps),
    ], dtype=np.float64)
    return np.concatenate([band_means, band_stds, log_means, norm_bands, ratios]).astype(np.float32)

# Model classes
class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, bias=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, bias=False)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, bias=False)
        self.lin   = nn.Linear(hidden_channels, num_classes, bias=False)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return global_mean_pool(x, batch)
    def embed(self, x, edge_index, batch):
        return self.forward(x, edge_index, batch)

class ExtendedQSUP(nn.Module):
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

        self.wavefunction_nets = nn.ModuleList([
            parametrizations.spectral_norm(nn.Linear(wf_input_dim, 2 * hidden_dim)) for _ in range(num_wavefunctions)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(2 * hidden_dim) for _ in range(num_wavefunctions)])
        self.temperature = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(0.1)
        if phase_per_dim:
            self.phases = nn.Parameter(torch.zeros(num_wavefunctions, hidden_dim))
        else:
            self.phases = nn.Parameter(torch.zeros(num_wavefunctions, 1))

        if self_modulation_steps > 0:
            self.gating_net = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.gating_net = None

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.constant_(self.classifier[-1].bias, 0.0)

    def forward(self, x):
        batch_size = x.size(0)
        eps = 1e-8

        if self.use_proj:
            x = self.input_proj(x)

        wave_r_list = []
        wave_i_list = []
        for s in range(self.num_wavefunctions):
            out = self.wavefunction_nets[s](x)    # shape: (batch, 2*hidden_dim)
            out = self.layer_norms[s](out)
            out = torch.exp(-out * out)           # ArcBell
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

        probs = self.dropout(probs)
        logits = self.classifier(probs)
        return logits

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

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    gcn_model = GCNNet(in_channels=9, hidden_channels=64, num_classes=2)
    sd_gcn = torch.load(GCN_MODEL_PATH, map_location="cpu")
    gcn_model.load_state_dict(sd_gcn, strict=True)
    gcn_model.eval()
    print("[INFO] GCN model loaded.")
except Exception as e:
    print("[ERROR] GCN model:", e)
    gcn_model = None

mlp_model = None
mlp_is_torch = False
MLP_PTH_PATH = os.path.join(MODELS_DIR, "mlp_model.pth")
try:
    torch_mlp = TorchMLP(input_dim=136, num_classes=2)
    sd_mlp = torch.load(MLP_PTH_PATH, map_location="cpu", weights_only=False)
    torch_mlp.load_state_dict(sd_mlp, strict=True)
    torch_mlp.eval()
    mlp_model = torch_mlp
    mlp_is_torch = True
    print("[INFO] TorchMLP model loaded from .pth")
except Exception as e:
    print(f"[WARN] TorchMLP load failed ({e}), trying sklearn fallback...")
    try:
        mlp_model = joblib.load(MLP_MODEL_PATH)
        print("[INFO] sklearn MLP model loaded from .pkl")
    except Exception as e2:
        print("[ERROR] MLP model:", e2)
        mlp_model = None

try:
    qsup_model = ExtendedQSUP(
        input_dim=136, hidden_dim=48, num_classes=2,
        num_wavefunctions=8, partial_norm=1.5,
        phase_per_dim=True, self_modulation_steps=3, topk=12
    )
    sd_qsup = torch.load(QSUP_MODEL_PATH, map_location="cpu", weights_only=False)
    qsup_model.load_state_dict(sd_qsup, strict=True)
    qsup_model.eval()
    print("[INFO] QSUP model loaded.")
except Exception as e:
    print("[ERROR] QSUP model:", e)
    qsup_model = None

try:
    pca_model = joblib.load(PCA_MODEL_PATH)
    print("[INFO] PCA model loaded.")
except Exception as e:
    print("[ERROR] PCA model:", e)
    pca_model = None

def generate_energy_landscape_3d():
    x_vals = np.linspace(-2,2,50)
    y_vals = np.linspace(-2,2,50)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.exp(-(X**2 + Y**2))
    return {"energy_landscape": {
        "x": X.flatten().tolist(),
        "y": Y.flatten().tolist(),
        "z": Z.flatten().tolist()
    }}

energy_landscape_data = generate_energy_landscape_3d()

# Aggregator functions
def adjacency_4ch():
    A = np.ones((4,4), dtype=np.float32)
    np.fill_diagonal(A, 0)
    return A

def compute_9subbands(sig: np.ndarray):
    freqs, psd = scipy.signal.welch(sig, SAMPLING_RATE, nperseg=512)
    feats = []
    for (low, high) in FREQUENCY_BANDS_9.values():
        idx = (freqs >= low) & (freqs < high)
        feats.append(np.sum(psd[idx]))
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
    freqs, psd = scipy.signal.welch(sig, SAMPLING_RATE, nperseg=512)
    mask_tot = (freqs >= 0.5) & (freqs < 30)
    mask_a   = (freqs >= 8) & (freqs < 12)
    tot_pow  = np.sum(psd[mask_tot])
    alp_pow  = np.sum(psd[mask_a])
    return alp_pow / (tot_pow + 1e-12)

def spectral_entropy(sig):
    freqs, psd = scipy.signal.welch(sig, SAMPLING_RATE, nperseg=512)
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
    alpha_mean = np.mean(alpha_ratios)
    alpha_std  = np.std(alpha_ratios)
    ent_mean   = np.mean(entropies)
    ent_std    = np.std(entropies)
    hj_arr = np.array(hj_all).reshape(NUM_CHANNELS, 3)
    hj_act_mean = np.mean(hj_arr[:,0])
    hj_act_std  = np.std(hj_arr[:,0])
    hj_mob_mean = np.mean(hj_arr[:,1])
    hj_mob_std  = np.std(hj_arr[:,1])
    hj_comp_mean = np.mean(hj_arr[:,2])
    hj_comp_std  = np.std(hj_arr[:,2])
    aggregator_10 = [
        alpha_mean, alpha_std,
        ent_mean,   ent_std,
        hj_act_mean, hj_act_std,
        hj_mob_mean, hj_mob_std,
        hj_comp_mean, hj_comp_std
    ]
    feats_20 = []
    for c in range(NUM_CHANNELS):
        feats_20.append(alpha_ratios[c])
        feats_20.append(entropies[c])
        off = c * 3
        feats_20.append(hj_all[off+0])
        feats_20.append(hj_all[off+1])
        feats_20.append(hj_all[off+2])
    final_30 = np.concatenate([
        np.array(feats_20, dtype=np.float32),
        np.array(aggregator_10, dtype=np.float32)
    ])
    return final_30

def compute_energy(feats: Dict[str, float]) -> float:
    return sum(feats.values())

def compute_stats(feats: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    out = {}
    for k, v in feats.items():
        out[k] = {"mean": v, "std": 0.0}
    return out

# Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[Any] = []
        self.buffer = np.zeros((NUM_CHANNELS, 0), dtype=np.float32)
        self.energy_landscape = self.generate_energy_landscape()

    def generate_energy_landscape(self):
        x_vals = np.linspace(-2, 2, 50)
        y_vals = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.exp(-(X**2 + Y**2))
        return {"energy_landscape": {
            "x": X.flatten().tolist(),
            "y": Y.flatten().tolist(),
            "z": Z.flatten().tolist()
        }}

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
        print("[INFO] client connected")
        await ws.send_json(self.energy_landscape)

    async def disconnect(self, ws: WebSocket):
        self.active_connections.remove(ws)

    async def receive_data(self, ws: WebSocket, data: EEGData):
        # The client sends only the cleaned 4-channel EEG (F3, F4, P3, O2)
        chunk = np.array([s.channels for s in data.data], dtype=np.float32).T  # shape (4, n_samples)
        self.buffer = np.concatenate((self.buffer, chunk), axis=1)
        if self.buffer.shape[1] > WINDOW_SAMPLES:
            self.buffer = self.buffer[:, -WINDOW_SAMPLES:]
        if self.buffer.shape[1] >= WINDOW_SAMPLES:
            gf_30 = aggregator_30(self.buffer).reshape(1, 30)
            node_feats = channelwise_9feats(self.buffer)   # (4, 9)
            extra_42 = extract_extra_features_from_gnn(node_feats).reshape(1, 42)
            A = adjacency_4ch()
            edge_idx = np.array(np.nonzero(A))
            edge_idx = torch.tensor(edge_idx, dtype=torch.long)
            x_tensor = torch.tensor(node_feats, dtype=torch.float)
            batch = torch.zeros(x_tensor.shape[0], dtype=torch.long)
            gcn_emb = np.zeros((1, 64), dtype=np.float32)
            if gcn_model:
                with torch.no_grad():
                    data_obj = Data(x=x_tensor, edge_index=edge_idx, y=torch.zeros(1, dtype=torch.long), batch=batch)
                    emb = gcn_model.embed(data_obj.x, data_obj.edge_index, data_obj.batch)
                    gcn_emb = emb.cpu().numpy().reshape(1, -1)
            full_vec = np.hstack([gf_30, extra_42, gcn_emb])  # (1, 136)
            # Classes: 0=Control, 1=FTD, 2=Alzheimer
            CLASS_LABELS = ["Control", "Alzheimer"]
            mlp_pred, mlp_conf, mlp_label = 0, 0.0, "Control"
            if mlp_model:
                try:
                    if mlp_is_torch:
                        with torch.no_grad():
                            t_in = torch.tensor(full_vec, dtype=torch.float)
                            p_m = torch.softmax(mlp_model(t_in), dim=1).cpu().numpy()[0]
                        mlp_pred = int(np.argmax(p_m))
                        mlp_conf = float(p_m[mlp_pred])
                    else:
                        p_m = mlp_model.predict_proba(full_vec)[0]
                        mlp_pred = int(np.argmax(p_m))
                        mlp_conf = float(p_m[mlp_pred])
                    mlp_label = CLASS_LABELS[mlp_pred]
                except Exception:
                    pass
            qsup_pred, qsup_conf, qsup_label = 0, 0.0, "Control"
            if qsup_model:
                with torch.no_grad():
                    t_in = torch.tensor(full_vec, dtype=torch.float)
                    p_q = torch.softmax(qsup_model(t_in), dim=1).cpu().numpy()[0]
                qsup_pred = int(np.argmax(p_q))
                qsup_conf = float(p_q[qsup_pred])
                qsup_label = CLASS_LABELS[qsup_pred]
            pc1, pc2 = 0.0, 0.0
            if pca_model:
                try:
                    pc = pca_model.transform(full_vec)[0]
                    pc1, pc2 = float(pc[0]), float(pc[1])
                except Exception:
                    pass
            feats_dict = {f"agg{i}": float(gf_30[0, i]) for i in range(30)}
            e_val = compute_energy(feats_dict)
            stats_dict = compute_stats(feats_dict)
            response = {
                "prediction_mlp": mlp_pred,
                "confidence_mlp": mlp_conf,
                "label_mlp": mlp_label,
                "prediction_qsup": qsup_pred,
                "confidence_qsup": qsup_conf,
                "label_qsup": qsup_label,
                "prediction_gcn": 0,
                "confidence_gcn": 0.0,
                "features": feats_dict,
                "stats": stats_dict,
                "pc1": pc1,
                "pc2": pc2,
                "energy": e_val
            }
            await ws.send_json(response)
            print(f"[INFO] MLP={mlp_label}({mlp_conf:.2f}), QSUP={qsup_label}({qsup_conf:.2f})")

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            msg = await ws.receive_text()
            try:
                data_obj = EEGData.parse_raw(msg)
                await manager.receive_data(ws, data_obj)
            except ValidationError as ve:
                await ws.send_json({"error": f"ValidationError: {ve}"})
            except Exception as e:
                await ws.send_json({"error": f"Failed to process data: {e}"})
    except WebSocketDisconnect:
        await manager.disconnect(ws)
    except Exception as e:
        await manager.disconnect(ws)
        print("[ERROR]", e)

def run_server():
    """Entry point for alzdetect-server CLI."""
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    run_server()
