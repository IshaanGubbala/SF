#!/usr/bin/env python
"""
Generate animated GIF visualizations for MLP and QSUP inference.

1. MLP: Classic neural network diagram with activations flowing through layers.
2. QSUP: 3D surface plot of wavefunction interference landscape across samples.
"""

import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import joblib
import mne

MODELS_DIR = "trained_models"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Model definitions (must match ai_model.py) ──────────────────────

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


class ExtendedQSUP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes,
                 num_wavefunctions=3, partial_norm=1.5,
                 phase_per_dim=False, self_modulation_steps=2, topk=8):
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

        self.wavefunction_nets = nn.ModuleList()
        for _ in range(num_wavefunctions):
            layer = nn.Linear(wf_input_dim, 2 * hidden_dim)
            layer = parametrizations.spectral_norm(layer)
            self.wavefunction_nets.append(layer)

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(2 * hidden_dim) for _ in range(num_wavefunctions)
        ])
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
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden_dim, num_classes),
        )
        nn.init.constant_(self.classifier[-1].bias, 0.0)
        self.classifier[-1].bias.data[1] = 0.75

    def forward_with_internals(self, x):
        eps = 1e-8
        if self.use_proj:
            x = self.input_proj(x)

        wave_r_list, wave_i_list, wave_magnitudes = [], [], []
        for s in range(self.num_wavefunctions):
            out = self.wavefunction_nets[s](x)
            out = self.layer_norms[s](out)
            out = torch.exp(-out * out)
            alpha = out[:, :self.hidden_dim]
            beta = out[:, self.hidden_dim:]
            norm_sq = torch.sum(alpha**2 + beta**2, dim=1, keepdim=True) + eps
            factor = torch.sqrt((self.partial_norm**2) / norm_sq)
            alpha, beta = alpha * factor, beta * factor
            wave_magnitudes.append(torch.sqrt(alpha**2 + beta**2))

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
        cosine_sim = (dot_prod / (wave_norms * mean_norm)).squeeze(2)
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

        return logits, {
            'wave_magnitudes': torch.stack(wave_magnitudes, dim=1),
            'interference_weights': interference_weights.squeeze(2),
            'sup_real': sup_real,
            'sup_imag': sup_imag,
            'probs': probs,
            'phases': self.phases.detach(),
        }


# ── Helper functions (mirrors ai_model.py) ──────────────────────────

FREQUENCY_BANDS = {
    "Delta": (0.5, 4), "Theta1": (4, 6), "Theta2": (6, 8),
    "Alpha1": (8, 10), "Alpha2": (10, 12),
    "Beta1": (12, 20), "Beta2": (20, 30),
    "Gamma1": (30, 40), "Gamma2": (40, 50),
}

def extract_extra_features_from_gnn(gnn_feat):
    eps = 1e-12
    band_means = np.mean(gnn_feat, axis=0).astype(np.float64)
    band_stds  = np.std(gnn_feat,  axis=0).astype(np.float64)
    log_means  = np.log1p(np.clip(band_means, 0, None))
    total_mean = float(np.sum(band_means)) + eps
    norm_bands = band_means / total_mean
    delta = band_means[0]
    theta = band_means[1] + band_means[2]
    alpha = band_means[3] + band_means[4]
    beta  = band_means[5] + band_means[6]
    gamma = band_means[7] + band_means[8]
    ratios = np.array([
        theta / (alpha + eps), alpha / (beta + eps),
        delta / (alpha + eps), (alpha + theta) / (total_mean + eps),
        delta / (total_mean + eps), gamma / (beta + eps),
    ], dtype=np.float64)
    return np.concatenate([band_means, band_stds, log_means, norm_bands, ratios]).astype(np.float32)


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


# ── Load models and data ────────────────────────────────────────────

def load_sample_data(n=20):
    """Load n CCV samples (136-dim: 30 handcrafted + 42 extra GNN + 64 GCN emb)."""
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data

    HANDCRAFTED_DIR = "processed_features/handcrafted"
    GNN_DIR         = "processed_features/gnn"
    CHANNELS_DIR    = "processed_features/channels"

    class GCNNet(nn.Module):
        def __init__(self, in_channels, hidden_channels, num_classes):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels, bias=False)
            self.conv2 = GCNConv(hidden_channels, hidden_channels, bias=False)
            self.conv3 = GCNConv(hidden_channels, hidden_channels, bias=False)
            self.lin = nn.Linear(hidden_channels, num_classes, bias=False)

        def embed(self, x, edge_index, batch):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))
            return global_mean_pool(x, batch)

    gcn = GCNNet(9, 64, 2)
    gcn.load_state_dict(torch.load(os.path.join(MODELS_DIR, "gcn_model.pth"), map_location="cpu"))
    gcn.eval()

    handcrafted_files = sorted(glob.glob(os.path.join(HANDCRAFTED_DIR, "*_handcrafted.npy")))
    ccv_list = []
    for f in handcrafted_files:
        if len(ccv_list) >= n:
            break
        subj_id = os.path.basename(f).split('_')[0]
        gnn_file = os.path.join(GNN_DIR, f"{subj_id}_gnn.npy")
        ch_file  = os.path.join(CHANNELS_DIR, f"{subj_id}_channels.json")
        if not os.path.exists(gnn_file) or not os.path.exists(ch_file):
            continue
        hc       = np.load(f).astype(np.float32)
        gnn_feat = np.load(gnn_file).astype(np.float32)
        with open(ch_file) as fp:
            ch_names = json.load(fp)

        A = compute_adjacency_matrix(ch_names)
        if A is None:
            continue

        extra = extract_extra_features_from_gnn(gnn_feat)

        # Subset gnn_feat to valid 10-20 channels (matches A dimensions)
        montage = mne.channels.make_standard_montage('standard_1020')
        pos_dict = montage.get_positions()['ch_pos']
        valid_mask = [i for i, ch in enumerate(ch_names) if ch in pos_dict]
        gnn_valid = gnn_feat[valid_mask]

        x_tensor = torch.tensor(gnn_valid, dtype=torch.float)
        edge_index = torch.tensor(np.array(np.nonzero(A)), dtype=torch.long)
        batch = torch.zeros(x_tensor.shape[0], dtype=torch.long)
        with torch.no_grad():
            emb = gcn.embed(x_tensor, edge_index, batch).numpy().flatten()

        # Source one-hot (4-dim): ds004504=0, ds003800=1, ds006036=2, zenodo=3
        if subj_id.startswith('bsub'):
            src = 1
        elif subj_id.startswith('zsub'):
            src = 2
        elif subj_id.startswith('sub'):
            src = 0
        else:
            src = 3
        src_onehot = np.zeros(4, dtype=np.float32)
        src_onehot[src] = 1.0
        ccv = np.concatenate([hc, extra, emb, src_onehot])  # 30+42+64+4 = 140
        ccv_list.append(ccv)

    return np.array(ccv_list, dtype=np.float32)


# ── MLP Visualization ───────────────────────────────────────────────

def get_mlp_layer_activations(model, x_sample):
    """Extract activations at each meaningful layer of the MLP."""
    model.eval()
    activations = [x_sample.numpy().flatten()]  # input layer

    x = x_sample.unsqueeze(0)
    with torch.no_grad():
        for layer in model.net:
            x = layer(x)
            if isinstance(layer, (nn.Linear, nn.GELU)):
                activations.append(x.squeeze(0).numpy().copy())

    return activations


def create_mlp_animation(model, samples, scaler=None, output_path="plots/mlp_inference.gif"):
    """Create animated GIF of MLP inference showing activations flowing through network."""
    # Layer sizes for the diagram
    layer_sizes = [140, 128, 64, 32, 2]
    layer_labels = ["Input\n(CCV 140d)", "Hidden 1\n(128, BN+GELU)", "Hidden 2\n(64, BN+GELU)",
                    "Hidden 3\n(32, GELU)", "Output\n(2 classes)"]
    # Max nodes to draw per layer (subsample large layers)
    max_display = [20, 20, 16, 16, 2]

    n_layers = len(layer_sizes)
    n_samples = len(samples)

    # Pre-compute all activations
    all_activations = []
    all_preds = []
    for i in range(n_samples):
        x = samples[i:i+1]
        if scaler is not None:
            x = scaler.transform(x)
        x_t = torch.tensor(x, dtype=torch.float)
        acts = get_mlp_layer_activations(model, x_t.squeeze(0))
        all_activations.append(acts)
        with torch.no_grad():
            logits = model(x_t)
            pred = torch.softmax(logits, dim=1).numpy().flatten()
        all_preds.append(pred)

    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#0a0a1a')

    def draw_frame(frame_idx):
        ax.clear()
        ax.set_facecolor('#0a0a1a')
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('auto')
        ax.axis('off')

        sample_idx = frame_idx // 3  # 3 frames per sample (input, propagate, output)
        phase = frame_idx % 3
        activations = all_activations[sample_idx]
        pred = all_preds[sample_idx]

        # Title
        pred_label = "Alzheimer" if pred[1] > 0.5 else "Control"
        conf = max(pred) * 100
        ax.set_title(
            f"MLP Inference  |  Sample {sample_idx+1}/{n_samples}  |  "
            f"Prediction: {pred_label} ({conf:.0f}%)",
            color='white', fontsize=14, fontweight='bold', pad=15
        )

        # Node positions
        positions = []
        for l_idx in range(n_layers):
            n_draw = max_display[l_idx]
            ys = np.linspace(-0.9, 0.9, n_draw)
            positions.append([(l_idx, y) for y in ys])

        # Determine which layers are "lit up" based on animation phase
        if phase == 0:
            active_layers = {0}
        elif phase == 1:
            active_layers = {0, 1, 2}
        else:
            active_layers = set(range(n_layers))

        # Draw connections first (behind nodes)
        for l_idx in range(n_layers - 1):
            alpha_conn = 0.08 if l_idx + 1 not in active_layers else 0.2
            pos_from = positions[l_idx]
            pos_to = positions[l_idx + 1]
            # Draw subset of connections
            step_from = max(1, len(pos_from) // 6)
            step_to = max(1, len(pos_to) // 6)
            for i in range(0, len(pos_from), step_from):
                for j in range(0, len(pos_to), step_to):
                    x_coords = [pos_from[i][0], pos_to[j][0]]
                    y_coords = [pos_from[i][1], pos_to[j][1]]
                    ax.plot(x_coords, y_coords, color='#3a5fcd',
                            alpha=alpha_conn, linewidth=0.3, zorder=1)

        # Draw nodes
        for l_idx in range(n_layers):
            n_actual = layer_sizes[l_idx]
            n_draw = max_display[l_idx]
            pos = positions[l_idx]

            if l_idx in active_layers:
                act = activations[l_idx] if l_idx < len(activations) else np.zeros(n_actual)
                # Normalize activations to [0, 1]
                act_abs = np.abs(act)
                act_max = act_abs.max() + 1e-8
                act_norm = act_abs / act_max

                # Subsample activations to match display nodes
                if len(act_norm) > n_draw:
                    indices = np.linspace(0, len(act_norm) - 1, n_draw).astype(int)
                    act_sub = act_norm[indices]
                else:
                    act_sub = act_norm

                for i, (px, py) in enumerate(pos):
                    val = act_sub[i] if i < len(act_sub) else 0
                    # Color: dark blue (inactive) to bright cyan (active)
                    r = val * 0.2
                    g = val * 0.8 + 0.1
                    b = val * 1.0 + 0.15
                    color = (min(r, 1), min(g, 1), min(b, 1))
                    size = 15 + val * 40
                    ax.scatter(px, py, s=size, c=[color], edgecolors='#4488ff',
                              linewidth=0.3, zorder=3, alpha=0.9)
            else:
                for px, py in pos:
                    ax.scatter(px, py, s=12, c='#1a1a3a', edgecolors='#333366',
                              linewidth=0.3, zorder=3, alpha=0.5)

            # Layer label
            ax.text(l_idx, -1.1, layer_labels[l_idx], ha='center', va='top',
                    color='#8899cc', fontsize=7, fontweight='bold')

            # Show "..." if subsampled
            if layer_sizes[l_idx] > max_display[l_idx]:
                ax.text(l_idx + 0.12, 0, f"({layer_sizes[l_idx]})", ha='left',
                        va='center', color='#556688', fontsize=6, style='italic')

        # Output labels
        if phase == 2:
            output_pos = positions[-1]
            labels = ["Control", "Alzheimer"]
            for i, (px, py) in enumerate(output_pos):
                conf_val = pred[i] * 100
                color = '#00ff88' if pred[i] == max(pred) else '#ff4466'
                ax.text(px + 0.15, py, f"{labels[i]}\n{conf_val:.1f}%",
                        ha='left', va='center', color=color, fontsize=9, fontweight='bold')

    total_frames = n_samples * 3
    print(f"[MLP VIZ] Generating {total_frames} frames for {n_samples} samples...")

    anim = animation.FuncAnimation(fig, draw_frame, frames=total_frames, interval=400)
    anim.save(output_path, writer='pillow', fps=3, dpi=120)
    plt.close()
    print(f"[MLP VIZ] Saved to {output_path}")


# ── QSUP 3D Surface Visualization ──────────────────────────────────

def create_qsup_animation(model, samples, scaler=None, output_path="plots/qsup_inference_3d.gif"):
    """Create animated 3D surface showing QSUP wavefunction landscape per sample."""
    model.eval()
    n_samples = len(samples)

    # Pre-compute all internals
    all_internals = []
    all_preds = []
    for i in range(n_samples):
        x = samples[i:i+1]
        if scaler is not None:
            x = scaler.transform(x)
        x_t = torch.tensor(x, dtype=torch.float)
        with torch.no_grad():
            logits, internals = model.forward_with_internals(x_t)
            pred = torch.softmax(logits, dim=1).numpy().flatten()
        all_preds.append(pred)
        all_internals.append({k: v.numpy() if isinstance(v, torch.Tensor) else v
                              for k, v in internals.items()})

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor('#0a0a1a')

    def draw_frame(sample_idx):
        fig.clear()
        fig.patch.set_facecolor('#0a0a1a')

        internals = all_internals[sample_idx]
        pred = all_preds[sample_idx]
        pred_label = "Alzheimer" if pred[1] > 0.5 else "Control"
        conf = max(pred) * 100

        fig.suptitle(
            f"QSUP Inference  |  Sample {sample_idx+1}/{n_samples}  |  "
            f"Prediction: {pred_label} ({conf:.0f}%)",
            color='white', fontsize=13, fontweight='bold', y=0.98
        )

        # ── Panel 1: 3D wavefunction interference surface ──
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.set_facecolor('#0a0a1a')

        wave_mags = internals['wave_magnitudes'][0]  # (num_wf, hidden_dim)
        num_wf, hidden_dim = wave_mags.shape

        # Create 2D surface: X = hidden dim index, Y = wavefunction index
        X_grid = np.arange(hidden_dim)
        Y_grid = np.arange(num_wf)
        X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)

        # Interpolate to make smoother surface
        from scipy.ndimage import zoom
        Z_raw = wave_mags
        zoom_factor = (3, max(1, 32 // hidden_dim))
        Z_smooth = zoom(Z_raw, zoom_factor, order=3)
        X_s = np.linspace(0, hidden_dim - 1, Z_smooth.shape[1])
        Y_s = np.linspace(0, num_wf - 1, Z_smooth.shape[0])
        X_sm, Y_sm = np.meshgrid(X_s, Y_s)

        surf = ax1.plot_surface(X_sm, Y_sm, Z_smooth, cmap='plasma',
                                alpha=0.85, edgecolor='none', antialiased=True)
        ax1.set_xlabel('Hidden Dim', color='#8899cc', fontsize=7, labelpad=2)
        ax1.set_ylabel('Wavefunction', color='#8899cc', fontsize=7, labelpad=2)
        ax1.set_zlabel('Magnitude', color='#8899cc', fontsize=7, labelpad=2)
        ax1.set_title('Wavefunction Landscape', color='#ccddff', fontsize=9, pad=5)
        ax1.tick_params(colors='#556688', labelsize=5)
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        ax1.view_init(elev=25, azim=45 + sample_idx * 8)

        # ── Panel 2: Interference + superposition polar ──
        ax2 = fig.add_subplot(132, projection='polar')
        ax2.set_facecolor('#0a0a1a')

        interf_w = internals['interference_weights'][0]  # (num_wf,)
        phases = internals['phases']  # (num_wf, dim) or (num_wf, 1)
        sup_r = internals['sup_real'][0]  # (hidden_dim,)
        sup_i = internals['sup_imag'][0]  # (hidden_dim,)

        # Plot each wavefunction as a polar vector
        colors_wf = cm.Set1(np.linspace(0, 1, num_wf))
        for wf_idx in range(num_wf):
            phase_vals = phases[wf_idx].flatten()
            weight = interf_w[wf_idx]
            # Mean phase angle
            mean_phase = np.mean(phase_vals)
            ax2.annotate('', xy=(mean_phase, weight * 3),
                         xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color=colors_wf[wf_idx],
                                         lw=2.5 * weight + 0.5))
            ax2.text(mean_phase, weight * 3 + 0.3, f'WF{wf_idx}\n({weight:.2f})',
                     ha='center', va='bottom', color=colors_wf[wf_idx], fontsize=6)

        # Superposition vector (mean direction)
        sup_angles = np.arctan2(sup_i, sup_r)
        sup_mags = np.sqrt(sup_r**2 + sup_i**2)
        mean_angle = np.mean(sup_angles)
        mean_mag = np.mean(sup_mags)
        ax2.annotate('', xy=(mean_angle, mean_mag * 2),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='#00ffaa', lw=3))
        ax2.text(mean_angle, mean_mag * 2 + 0.3, 'SUP', ha='center',
                 color='#00ffaa', fontsize=8, fontweight='bold')

        ax2.set_title('Phase Interference', color='#ccddff', fontsize=9, pad=15)
        ax2.tick_params(colors='#556688', labelsize=5)

        # ── Panel 3: Measurement probabilities (bar chart) ──
        ax3 = fig.add_subplot(133)
        ax3.set_facecolor('#0a0a1a')

        probs = internals['probs'][0]  # (hidden_dim,)
        n_bars = len(probs)
        bar_colors = np.zeros((n_bars, 4))
        for j in range(n_bars):
            intensity = min(probs[j] * 5, 1.0)
            bar_colors[j] = (0.1 + intensity * 0.2, 0.4 + intensity * 0.5,
                             0.8 + intensity * 0.2, 0.8)

        ax3.bar(range(n_bars), probs, color=bar_colors, edgecolor='#334466', linewidth=0.3)
        ax3.set_xlabel('Hidden Dim', color='#8899cc', fontsize=7)
        ax3.set_ylabel('Probability', color='#8899cc', fontsize=7)
        ax3.set_title('Measurement Collapse', color='#ccddff', fontsize=9, pad=8)
        ax3.tick_params(colors='#556688', labelsize=5)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_color('#334466')
        ax3.spines['left'].set_color('#334466')

        # Output prediction bar at bottom
        fig.text(0.5, 0.02,
                 f"Control: {pred[0]*100:.1f}%    |    Alzheimer: {pred[1]*100:.1f}%",
                 ha='center', color='#00ff88' if pred[0] > pred[1] else '#ff6644',
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e', edgecolor='#334466'))

        fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.1, wspace=0.3)

    print(f"[QSUP VIZ] Generating {n_samples} frames...")
    anim = animation.FuncAnimation(fig, draw_frame, frames=n_samples, interval=800)
    anim.save(output_path, writer='pillow', fps=2, dpi=110)
    plt.close()
    print(f"[QSUP VIZ] Saved to {output_path}")


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading models...")
    mlp = TorchMLP(input_dim=140, num_classes=2)
    mlp.load_state_dict(torch.load(os.path.join(MODELS_DIR, "mlp_model.pth"), map_location="cpu"))
    mlp.eval()

    qsup = ExtendedQSUP(input_dim=140, hidden_dim=48, num_classes=2,
                         num_wavefunctions=8, partial_norm=1.5,
                         phase_per_dim=True, self_modulation_steps=3, topk=12)
    qsup.load_state_dict(torch.load(os.path.join(MODELS_DIR, "qsup_model.pth"),
                                     map_location="cpu", weights_only=False))
    qsup.eval()

    # Load scalers
    mlp_scaler = None
    qsup_scaler = None
    try:
        mlp_scaler = joblib.load(os.path.join(MODELS_DIR, "mlp_scaler.joblib"))
    except Exception:
        pass
    try:
        qsup_scaler = joblib.load(os.path.join(MODELS_DIR, "qsup_scaler.joblib"))
    except Exception:
        pass

    print("Loading sample data...")
    samples = load_sample_data(n=15)
    print(f"Loaded {len(samples)} CCV samples of shape {samples.shape}")

    print("\n=== Generating MLP Inference Animation ===")
    create_mlp_animation(mlp, samples, scaler=mlp_scaler,
                         output_path=os.path.join(PLOTS_DIR, "mlp_inference.gif"))

    print("\n=== Generating QSUP 3D Inference Animation ===")
    create_qsup_animation(qsup, samples, scaler=qsup_scaler,
                          output_path=os.path.join(PLOTS_DIR, "qsup_inference_3d.gif"))

    print("\nDone! Check plots/ directory for GIFs.")
