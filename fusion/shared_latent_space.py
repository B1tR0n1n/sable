#!/usr/bin/env python3
"""
Project PARALLAX — Phase 4: Shared Latent State Space
=======================================================
The bridge between three pillars.

One vector per component. 128 dimensions. Every pillar reads from it
and writes to it. Cross-attention learns when to trust which pillar.

Components:
  - ProjectionHead: maps pillar native output → shared 128-dim space
  - ReadingHead: maps shared space → pillar native input
  - CrossAttentionFusion: attention-weighted combination of three perspectives
  - SharedStateSpace: the full fusion layer (170K params)
  - DiagnosticLoop: iterative reasoning across all three pillars

Usage:
    python shared_latent_space.py --device cuda
"""

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar3"))

from sable_sim.core.states import N_STATES, STATE_NAMES, STATE_MAP
from generate_temporal_data import NODE_FEAT_DIM

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"

# Shared space dimensions
Z_DIM = 128
Z_CONSENSUS = 96   # Aligned across pillars
Z_PERSPECTIVE = 32  # Unique per pillar

# Pillar native dimensions
GNN_DIM = 256       # GNN node embedding
POMDP_DIM = 8       # [4 state probs, confidence, obs_age, contradiction, hub_centrality]
MAMBA_STATE_DIM = 25 # Per-node state features (health + state_onehot + type_onehot)


# ── Cross-Attention ──────────────────────────────────────────────────────


class CrossAttentionFusion(nn.Module):
    """Cross-attention over three pillar perspectives."""

    def __init__(self, d_model: int = Z_DIM, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, perspectives: torch.Tensor) -> torch.Tensor:
        B, N, P, D = perspectives.shape
        x = perspectives.reshape(B * N, P, D)
        Q = self.q_proj(x).view(B * N, P, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B * N, P, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B * N, P, self.n_heads, self.d_head).transpose(1, 2)
        attn = F.softmax((Q @ K.transpose(-2, -1)) / self.scale, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ V).transpose(1, 2).reshape(B * N, P, D)
        fused = out.mean(dim=1)
        fused = self.out_proj(fused)
        fused = self.norm(fused)
        self._last_attn = attn.detach().reshape(B, N, self.n_heads, P, P)
        return fused.reshape(B, N, D)

    def get_pillar_weights(self):
        if not hasattr(self, '_last_attn'):
            return None
        return self._last_attn.mean(dim=2).mean(dim=-1)


class GatedFusion(nn.Module):
    """Learned gating over three pillar perspectives.

    Instead of cross-attention (which degenerates to uniform with 3 tokens),
    this uses a gating network that sees ALL three projections concatenated
    and produces per-pillar weights. The gate is input-dependent: different
    nodes get different pillar weightings based on what each pillar sees.
    """

    def __init__(self, d_model: int = Z_DIM, dropout: float = 0.1):
        super().__init__()
        # Gate network: sees all 3 projections, produces 3 weights
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
            # No softmax here — we use a learnable temperature
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, perspectives: torch.Tensor) -> torch.Tensor:
        """
        Args:
            perspectives: (B, N, 3, Z_DIM)

        Returns:
            (B, N, Z_DIM)
        """
        B, N, P, D = perspectives.shape

        # Concatenate all perspectives as gate input
        gate_input = perspectives.reshape(B, N, P * D)  # (B, N, 3*128)

        # Produce per-pillar weights with learnable temperature
        gate_logits = self.gate(gate_input)  # (B, N, 3)
        weights = F.softmax(gate_logits / self.temperature.clamp(min=0.1), dim=-1)  # (B, N, 3)

        # Weighted sum of perspectives
        # weights: (B, N, 3, 1), perspectives: (B, N, 3, D)
        fused = (weights.unsqueeze(-1) * perspectives).sum(dim=2)  # (B, N, D)
        fused = self.out_proj(fused)
        fused = self.norm(fused)

        self._last_weights = weights.detach()

        return fused

    def get_pillar_weights(self) -> torch.Tensor:
        if hasattr(self, '_last_weights'):
            return self._last_weights
        return None


# ── Projection Heads ──────────────────────────────────────────────────────


class GNNProjection(nn.Module):
    """GNN embedding (256-dim) → shared space (128-dim)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(GNN_DIM, Z_DIM),
            nn.LayerNorm(Z_DIM),
        )

    def forward(self, gnn_emb: torch.Tensor) -> torch.Tensor:
        """gnn_emb: (B, N, 256) → (B, N, 128)"""
        return self.proj(gnn_emb)


class POMDPProjection(nn.Module):
    """POMDP belief vector (8-dim) → shared space (128-dim)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(POMDP_DIM, 64),
            nn.GELU(),
            nn.Linear(64, Z_DIM),
            nn.LayerNorm(Z_DIM),
        )

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        """belief: (B, N, 8) → (B, N, 128)"""
        return self.proj(belief)


class MambaProjection(nn.Module):
    """Mamba state prediction → shared space (128-dim)."""

    def __init__(self, mamba_dim: int = MAMBA_STATE_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(mamba_dim, Z_DIM),
            nn.LayerNorm(Z_DIM),
        )

    def forward(self, mamba_out: torch.Tensor) -> torch.Tensor:
        """mamba_out: (B, N, mamba_dim) → (B, N, 128)"""
        return self.proj(mamba_out)


# ── Reading Heads ─────────────────────────────────────────────────────────


class GNNReading(nn.Module):
    """Shared space → conditioning signal for GNN re-embedding."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(Z_DIM, GNN_DIM)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


class POMDPReading(nn.Module):
    """Shared space → belief update signal for POMDP."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(Z_DIM, 32),
            nn.GELU(),
            nn.Linear(32, POMDP_DIM),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


class MambaReading(nn.Module):
    """Shared space → conditioning signal for Mamba prediction."""

    def __init__(self, mamba_dim: int = MAMBA_STATE_DIM):
        super().__init__()
        self.proj = nn.Linear(Z_DIM, mamba_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


# ── Task Head ─────────────────────────────────────────────────────────────


class StateClassifier(nn.Module):
    """Classify from fused Z + all three pillar projections.

    Takes the fused representation AND each pillar's individual projection,
    so the model can never be worse than the best single pillar. The fusion
    adds signal on top of what each pillar already knows.
    """

    def __init__(self):
        super().__init__()
        # Input: Z (128) + z_gnn (128) + z_pomdp (128) + z_mamba (128) = 512
        self.head = nn.Sequential(
            nn.Linear(Z_DIM * 4, Z_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(Z_DIM, N_STATES),
        )

    def forward(self, z_fused, z_gnn, z_pomdp, z_mamba):
        """All inputs (B, N, 128) → (B, N, 4) logits"""
        combined = torch.cat([z_fused, z_gnn, z_pomdp, z_mamba], dim=-1)
        return self.head(combined)


# ── Full Shared State Space ──────────────────────────────────────────────


class SharedStateSpace(nn.Module):
    """The complete fusion layer. 170K params.

    Three pillars project in → cross-attend → fused Z → classify + read back.
    """

    def __init__(self, mamba_dim: int = MAMBA_STATE_DIM):
        super().__init__()
        # Write projections
        self.gnn_proj = GNNProjection()
        self.pomdp_proj = POMDPProjection()
        self.mamba_proj = MambaProjection(mamba_dim)

        # Fusion
        self.cross_attn = CrossAttentionFusion()

        # Read projections
        self.gnn_read = GNNReading()
        self.pomdp_read = POMDPReading()
        self.mamba_read = MambaReading(mamba_dim)

        # Task head
        self.classifier = StateClassifier()

    def forward(
        self,
        gnn_emb: torch.Tensor,    # (B, N, 256)
        pomdp_belief: torch.Tensor, # (B, N, 8)
        mamba_pred: torch.Tensor,   # (B, N, mamba_dim)
    ) -> dict:
        """Full fusion forward pass."""
        # Project into shared space
        z_gnn = self.gnn_proj(gnn_emb)       # (B, N, 128)
        z_pomdp = self.pomdp_proj(pomdp_belief) # (B, N, 128)
        z_mamba = self.mamba_proj(mamba_pred)   # (B, N, 128)

        # Stack perspectives: (B, N, 3, 128)
        perspectives = torch.stack([z_gnn, z_pomdp, z_mamba], dim=2)

        # Cross-attention fusion
        Z = self.cross_attn(perspectives)  # (B, N, 128)

        # Task: classify from fused Z + individual projections (residual)
        state_logits = self.classifier(Z, z_gnn, z_pomdp, z_mamba)  # (B, N, 4)

        # Read back: conditioning signals for each pillar
        gnn_condition = self.gnn_read(Z)     # (B, N, 256)
        pomdp_update = self.pomdp_read(Z)    # (B, N, 8)
        mamba_condition = self.mamba_read(Z)  # (B, N, mamba_dim)

        # Pillar weights for interpretability
        pillar_weights = self.cross_attn.get_pillar_weights()

        return {
            "Z": Z,
            "state_logits": state_logits,
            "gnn_condition": gnn_condition,
            "pomdp_update": pomdp_update,
            "mamba_condition": mamba_condition,
            "pillar_weights": pillar_weights,
            "z_gnn": z_gnn,
            "z_pomdp": z_pomdp,
            "z_mamba": z_mamba,
        }

    def param_count(self) -> dict:
        counts = {}
        for name, module in [
            ("gnn_proj", self.gnn_proj),
            ("pomdp_proj", self.pomdp_proj),
            ("mamba_proj", self.mamba_proj),
            ("cross_attn", self.cross_attn),
            ("gnn_read", self.gnn_read),
            ("pomdp_read", self.pomdp_read),
            ("mamba_read", self.mamba_read),
            ("classifier", self.classifier),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(counts.values())
        return counts


# ── Training ──────────────────────────────────────────────────────────────


def alignment_loss(z_gnn, z_pomdp, z_mamba, states, margin=1.0):
    """Contrastive alignment on consensus dimensions.

    Same-state projections should be close. Different-state should be far.
    Only applies to first 96 dims (consensus subspace).
    """
    z_g = z_gnn[:, :, :Z_CONSENSUS]
    z_p = z_pomdp[:, :, :Z_CONSENSUS]
    z_m = z_mamba[:, :, :Z_CONSENSUS]

    B, N, D = z_g.shape

    # Pairwise distances between pillar projections of same node
    d_gp = F.pairwise_distance(z_g.reshape(-1, D), z_p.reshape(-1, D))
    d_gm = F.pairwise_distance(z_g.reshape(-1, D), z_m.reshape(-1, D))
    d_pm = F.pairwise_distance(z_p.reshape(-1, D), z_m.reshape(-1, D))

    # Same-node projections should be close
    pull_loss = (d_gp.pow(2) + d_gm.pow(2) + d_pm.pow(2)).mean()

    # Different-state nodes should be far apart in consensus space
    # Sample random pairs
    flat_z = ((z_g + z_p + z_m) / 3).reshape(B * N, D)
    flat_states = states.reshape(B * N)

    idx = torch.randperm(B * N, device=z_g.device)[:min(B * N, 1000)]
    if len(idx) < 2:
        return pull_loss

    pairs_a = idx[:len(idx) // 2]
    pairs_b = idx[len(idx) // 2:len(idx) // 2 * 2]

    same_state = (flat_states[pairs_a] == flat_states[pairs_b]).float()
    dist = F.pairwise_distance(flat_z[pairs_a], flat_z[pairs_b])

    # Contrastive: same state → close, different state → far
    push_loss = (
        same_state * dist.pow(2) +
        (1 - same_state) * F.relu(margin - dist).pow(2)
    ).mean()

    return pull_loss + push_loss


def _compute_fusion_batch_loss(shared_space, aux_heads, out, s, mask, state_weights, device):
    """Compute combined loss for one batch: task + auxiliary + alignment + diversity."""
    aux_gnn_head, aux_pomdp_head, aux_mamba_head = aux_heads

    # 1. Fused task loss
    logits = out["state_logits"]
    task_loss = F.cross_entropy(
        logits.reshape(-1, N_STATES), s.reshape(-1).long(),
        weight=state_weights, reduction="none"
    )
    task_loss = (task_loss * mask.reshape(-1)).sum() / mask.sum()

    # 2. Auxiliary pillar-specific losses
    aux_loss = torch.tensor(0.0, device=device)
    for head, z_key in [(aux_gnn_head, "z_gnn"), (aux_pomdp_head, "z_pomdp"), (aux_mamba_head, "z_mamba")]:
        pillar_logits = head(out[z_key])
        pillar_loss = F.cross_entropy(
            pillar_logits.reshape(-1, N_STATES), s.reshape(-1).long(), reduction="none"
        )
        aux_loss = aux_loss + (pillar_loss * mask.reshape(-1)).sum() / mask.sum()
    aux_loss = aux_loss / 3.0

    # 3. Alignment loss
    align = alignment_loss(out["z_gnn"], out["z_pomdp"], out["z_mamba"], s)

    # 4. Attention diversity loss
    weights = out["pillar_weights"]
    if weights is not None:
        w_entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1)
        diversity_loss = (w_entropy * mask).sum() / mask.sum()
    else:
        diversity_loss = torch.tensor(0.0, device=device)

    return task_loss + 0.3 * aux_loss + 0.1 * align + 0.05 * diversity_loss


def _validate_fusion(shared_space, val_data, device, batch_size):
    """Validate shared state space, returning (macro_f1, per_class_f1)."""
    shared_space.eval()
    class_tp = torch.zeros(N_STATES)
    class_fp = torch.zeros(N_STATES)
    class_fn = torch.zeros(N_STATES)
    n_val = val_data["gnn"].size(0)

    with torch.no_grad():
        for i in range(0, n_val, batch_size):
            g = val_data["gnn"][i:i+batch_size].to(device)
            p = val_data["pomdp"][i:i+batch_size].to(device)
            m = val_data["mamba"][i:i+batch_size].to(device)
            s = val_data["states"][i:i+batch_size].to(device)
            mask = val_data["mask"][i:i+batch_size].to(device)

            out = shared_space(g, p, m)
            preds = out["state_logits"].argmax(dim=-1)
            valid = mask > 0

            for c in range(N_STATES):
                ct = (s.long() == c) & valid
                cp = (preds == c) & valid
                class_tp[c] += (ct & cp).sum().item()
                class_fp[c] += (~ct & cp).sum().item()
                class_fn[c] += (ct & ~cp).sum().item()

    class_f1 = []
    for c in range(N_STATES):
        prec = class_tp[c] / max(class_tp[c] + class_fp[c], 1)
        rec = class_tp[c] / max(class_tp[c] + class_fn[c], 1)
        class_f1.append(2 * prec * rec / max(prec + rec, 1e-8))
    macro_f1 = sum(class_f1) / N_STATES
    return macro_f1, class_f1


def train_fusion(
    shared_space: SharedStateSpace,
    train_data: dict,
    val_data: dict,
    device: str = "cuda",
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.001,
):
    """Train the shared state space on simulator data."""

    state_counts = torch.bincount(train_data["states"].flatten().long(), minlength=N_STATES).float().clamp(min=1)
    state_weights = (train_data["states"].numel() / (N_STATES * state_counts)).clamp(max=8.0).to(device)

    aux_gnn_head = nn.Linear(Z_DIM, N_STATES).to(device)
    aux_pomdp_head = nn.Linear(Z_DIM, N_STATES).to(device)
    aux_mamba_head = nn.Linear(Z_DIM, N_STATES).to(device)
    aux_heads = (aux_gnn_head, aux_pomdp_head, aux_mamba_head)
    aux_params = list(aux_gnn_head.parameters()) + list(aux_pomdp_head.parameters()) + list(aux_mamba_head.parameters())

    all_params = list(shared_space.parameters()) + aux_params
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    n_train = train_data["gnn"].size(0)

    best_val_macro = 0.0
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        shared_space.train()
        for h in aux_heads:
            h.train()

        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            g = train_data["gnn"][idx].to(device)
            p = train_data["pomdp"][idx].to(device)
            m = train_data["mamba"][idx].to(device)
            s = train_data["states"][idx].to(device)
            mask = train_data["mask"][idx].to(device)

            out = shared_space(g, p, m)
            loss = _compute_fusion_batch_loss(shared_space, aux_heads, out, s, mask, state_weights, device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            macro_f1, class_f1 = _validate_fusion(shared_space, val_data, device, batch_size)

            improved = macro_f1 > best_val_macro
            if improved:
                best_val_macro = macro_f1
                best_state = {k: v.clone() for k, v in shared_space.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if epoch <= 10 or epoch % 10 == 0 or improved:
                marker = f"{C_SUCCESS}*{C_RESET}" if improved else " "
                per_class = " ".join(f"{f:.3f}" for f in class_f1)
                print(
                    f"  {C_TEXT}{epoch:4d}{C_RESET}  "
                    f"loss={epoch_loss/n_batches:.4f}  "
                    f"macro={macro_f1:.4f}  "
                    f"[{per_class}]  {marker}"
                )

            if patience >= 8:
                print(f"  {C_DIM}Early stopping at epoch {epoch}{C_RESET}")
                break

    if best_state:
        shared_space.load_state_dict(best_state)
    shared_space.eval()

    return best_val_macro


# ── Data Generation ───────────────────────────────────────────────────────


def generate_fusion_data(count: int, device: str = "cuda", seed: int = 42):
    """Generate training data for the shared state space.

    For each simulated scenario, produces:
      - GNN embeddings (from running GNN on the topology)
      - POMDP belief vectors (from running belief propagation)
      - Mamba state features (raw state vectors)
      - Ground truth states (from simulator)
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent / "pillar1"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "pillar2"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "pillar3"))

    from cortex_gnn_model import SableGNN
    from pomcp import BeliefState
    from generate_temporal_data import (
        build_random_topology, encode_system_state, NODE_FEAT_DIM,
    )
    from build_infra_dataset import (
        NODE_TYPES, EDGE_TYPES, N_NODE_TYPES, N_EDGE_TYPES,
        NODE_FEAT_DIM as INFRA_NODE_FEAT_DIM, EDGE_FEAT_DIM as INFRA_EDGE_FEAT_DIM,
    )
    from sable_sim.core.component import ComponentState
    from sable_sim.core.state import SystemState
    from sable_sim.simulation.propagation import PropagationEngine
    from sable_sim.simulation.failure_injection import FailureInjector
    from sable_sim.simulation.fog import FogOfWar
    from sable_sim.utils.random import SeededRandom

    # Load GNN — infra-trained with native 10-dim node / 7-dim edge features
    gnn_ckpt_path = str(Path(__file__).parent.parent / "pillar1" / "checkpoints" / "best_model.pt")
    ckpt = torch.load(gnn_ckpt_path, weights_only=False, map_location=device)
    mc = ckpt["config"]
    gnn = SableGNN(
        in_dim=mc["in_dim"], hidden_dim=mc["hidden_dim"],
        edge_dim=mc["edge_dim"], num_layers=mc["num_layers"],
        heads=mc["heads"], dropout=mc["dropout"],
    ).to(device)
    model_state = gnn.state_dict()
    for k, v in ckpt["model_state_dict"].items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
    gnn.load_state_dict(model_state)
    gnn.eval()

    # Component type → infra node type mapping (same as build_infra_dataset)
    COMP_TYPE_MAP = {
        "ComponentType.CORE_SWITCH": NODE_TYPES["router"],
        "ComponentType.ACCESS_SWITCH": NODE_TYPES["switch"],
        "ComponentType.FIREWALL": NODE_TYPES["router"],
        "ComponentType.ROUTER": NODE_TYPES["router"],
        "ComponentType.LOAD_BALANCER": NODE_TYPES["switch"],
        "ComponentType.SERVER_PHYSICAL": NODE_TYPES["server"],
        "ComponentType.SERVER_VIRTUAL": NODE_TYPES["server"],
        "ComponentType.HYPERVISOR": NODE_TYPES["server"],
        "ComponentType.STORAGE_ARRAY": NODE_TYPES["storage"],
        "ComponentType.STORAGE_TARGET": NODE_TYPES["storage"],
        "ComponentType.VDI_BROKER": NODE_TYPES["service"],
        "ComponentType.VDI_HOST": NODE_TYPES["server"],
        "ComponentType.DNS_SERVER": NODE_TYPES["service"],
        "ComponentType.DHCP_SERVER": NODE_TYPES["service"],
        "ComponentType.DOMAIN_CONTROLLER": NODE_TYPES["service"],
        "ComponentType.CERTIFICATE_AUTHORITY": NODE_TYPES["service"],
        "ComponentType.MONITORING_SERVER": NODE_TYPES["service"],
        "ComponentType.WAN_LINK": NODE_TYPES["gateway"],
        "ComponentType.INTERNET_GATEWAY": NODE_TYPES["gateway"],
        "ComponentType.APPLICATION_SERVICE": NODE_TYPES["server"],
    }
    DEP_TYPE_MAP = {
        "DependencyType.HARD": EDGE_TYPES["backbone"],
        "DependencyType.SOFT": EDGE_TYPES["access"],
        "DependencyType.SERVICE": EDGE_TYPES["service_dep"],
        "DependencyType.RESOURCE": EDGE_TYPES["storage_dep"],
    }

    rng = SeededRandom(seed)
    max_nodes = 40

    # Output tensors
    all_gnn = torch.zeros(count, max_nodes, GNN_DIM)
    all_pomdp = torch.zeros(count, max_nodes, POMDP_DIM)
    all_mamba = torch.zeros(count, max_nodes, NODE_FEAT_DIM)
    all_states = torch.zeros(count, max_nodes, dtype=torch.long)
    all_mask = torch.zeros(count, max_nodes)

    generated = 0
    t0 = time.time()

    while generated < count:
        graph = build_random_topology(rng, 15, 40)
        components = graph.get_all_components()
        if len(components) < 15:
            continue
        component_ids = [c.id for c in components]
        n = len(component_ids)

        # Inject failure
        state = SystemState(graph)
        injector = FailureInjector(rng)
        try:
            injections = injector.generate_failures(
                state, difficulty=rng.weighted_choice(["easy", "medium", "hard"], [0.2, 0.5, 0.3])
            )
        except Exception:
            continue
        if not injections:
            continue

        use_deg = rng.random() < 0.4
        for inj in injections:
            injector.inject(state, inj)
            if use_deg:
                comp = graph.get_component(inj.component_id)
                if comp and comp.state == ComponentState.FAILED:
                    comp.state = ComponentState.DEGRADED
                    comp.health = rng.uniform(0.25, 0.45)

        # Snapshot PRE-propagation state (what the operator sees early)
        pre_prop_state = encode_system_state(graph, component_ids)

        # Propagate to get ground truth OUTCOME
        engine = PropagationEngine(max_ticks=30, soft_impact_factor=0.4 if use_deg else 0.3)
        engine.propagate(state)

        # 1. GNN embeddings — infra-native 10-dim features matching trained GNN
        cid_to_idx = {cid: i for i, cid in enumerate(component_ids)}
        degrees = {}
        for comp in components:
            degrees[comp.id] = len(comp.dependencies_in) + len(comp.dependencies_out)
        max_deg = max(degrees.values()) if degrees else 1

        node_features = np.zeros((n, INFRA_NODE_FEAT_DIM), dtype=np.float32)
        pre_healths = []
        for i, comp in enumerate(components):
            pre_health = pre_prop_state[i * NODE_FEAT_DIM]
            pre_healths.append(pre_health)
            ntype = COMP_TYPE_MAP.get(str(comp.type), NODE_TYPES["unknown"])
            node_features[i, ntype] = 1.0  # type one-hot
            node_features[i, N_NODE_TYPES] = degrees[comp.id] / max(max_deg, 1)  # degree norm
            node_features[i, N_NODE_TYPES + 1] = pre_health  # health
        x = torch.tensor(node_features, dtype=torch.float32).to(device)

        sources, targets, edge_feats = [], [], []
        for comp in components:
            si = cid_to_idx[comp.id]
            for dep_id in comp.dependencies_in:
                ti = cid_to_idx.get(dep_id)
                if ti is not None:
                    dep = graph.get_dependency(comp.id, dep_id)
                    if dep:
                        etype = DEP_TYPE_MAP.get(str(dep.type), EDGE_TYPES["unknown"])
                        feat = np.zeros(INFRA_EDGE_FEAT_DIM, dtype=np.float32)
                        feat[etype] = 1.0
                        feat[N_EDGE_TYPES] = 1.0  # weight
                        sources.append(si)
                        targets.append(ti)
                        edge_feats.append(feat)

        if sources:
            ei = torch.tensor([sources, targets], dtype=torch.long).to(device)
            ea = torch.tensor(np.array(edge_feats), dtype=torch.float32).to(device)
            with torch.no_grad():
                gnn_emb = gnn.encode(x, ei, ea)  # (n, hidden_dim=128)
            # Pad to GNN_DIM if hidden_dim < GNN_DIM
            emb_cpu = gnn_emb.cpu()
            if emb_cpu.size(1) < GNN_DIM:
                padded = torch.zeros(n, GNN_DIM)
                padded[:, :emb_cpu.size(1)] = emb_cpu
                all_gnn[generated, :n] = padded
            else:
                all_gnn[generated, :n] = emb_cpu[:, :GNN_DIM]

        # 2. POMDP belief vectors
        fog = FogOfWar(monitoring_coverage=0.75, rng=SeededRandom(seed + generated))
        operator_view = fog.generate_operator_view(state)

        belief = BeliefState(component_ids)
        for obs in operator_view.get("observations", []):
            belief.update_from_observation(obs["component_id"], obs["observed_state"], 0.85)
        belief.propagate_beliefs(graph)

        for i, cid in enumerate(component_ids):
            b = belief.beliefs[cid]
            conf = 1.0 - belief.entropy(cid) / 2.0  # Normalize entropy to confidence
            obs_age = float(belief.observation_age.get(cid, 5))
            has_contra = 1.0 if cid in [c[0] for c in belief.root_cause_candidates.items() if c[1] > 1] else 0.0
            # Hub centrality: number of dependents
            dependents = graph.get_dependents(cid)
            hub = min(len(dependents) / 10.0, 1.0)

            all_pomdp[generated, i] = torch.tensor([
                b[0], b[1], b[2], b[3],  # State probabilities
                conf, obs_age / 10.0, has_contra, hub,
            ])

        # 3. Mamba state features — PRE-propagation (partial observation, not ground truth)
        for i in range(n):
            start = i * NODE_FEAT_DIM
            all_mamba[generated, i] = torch.tensor(pre_prop_state[start:start + NODE_FEAT_DIM])

        # 4. Ground truth states
        for i, cid in enumerate(component_ids):
            comp = graph.get_component(cid)
            if comp:
                all_states[generated, i] = float(STATE_MAP.get(comp.state, 0))

        # 5. Mask
        all_mask[generated, :n] = 1.0

        generated += 1
        if generated % 1000 == 0:
            print(f"    {C_DIM}{generated}/{count} ({time.time()-t0:.1f}s){C_RESET}")

    print(f"  {C_TEXT}Generated {generated} samples in {time.time()-t0:.1f}s{C_RESET}")

    # Split
    perm = torch.randperm(generated)
    n_train = int(generated * 0.8)

    train_data = {
        "gnn": all_gnn[perm[:n_train]],
        "pomdp": all_pomdp[perm[:n_train]],
        "mamba": all_mamba[perm[:n_train]],
        "states": all_states[perm[:n_train]],
        "mask": all_mask[perm[:n_train]],
    }
    val_data = {
        "gnn": all_gnn[perm[n_train:]],
        "pomdp": all_pomdp[perm[n_train:]],
        "mamba": all_mamba[perm[n_train:]],
        "states": all_states[perm[n_train:]],
        "mask": all_mask[perm[n_train:]],
    }

    return train_data, val_data


# ── Full Test ─────────────────────────────────────────────────────────────


def run_fusion_experiment(device="cuda", n_samples=10000):
    print(f"\n{C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║   Phase 4: Shared Latent State Space              ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║   170K params. Three perspectives, one truth.      ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════╝{C_RESET}\n")

    # Generate data
    print(f"  {C_INFO}Generating training data from simulator...{C_RESET}")
    train_data, val_data = generate_fusion_data(count=n_samples, device=device)

    # Build fusion layer
    shared_space = SharedStateSpace(mamba_dim=NODE_FEAT_DIM).to(device)
    params = shared_space.param_count()

    print(f"\n  {C_INFO}Shared State Space Architecture:{C_RESET}")
    for name, count in params.items():
        bar = "█" * (count // 1000)
        print(f"    {C_DIM}{name:<15s}{C_RESET} {C_TEXT}{count:>8,}{C_RESET} {C_GOLD}{bar}{C_RESET}")

    # Also train single-pillar baselines for comparison
    print(f"\n  {C_INFO}Training baselines (single-pillar classifiers)...{C_RESET}")

    def _train_baseline(head, data_key, train_data, device, n_epochs=30):
        """Train a single-pillar baseline classifier."""
        opt = torch.optim.Adam(head.parameters(), lr=0.001, weight_decay=1e-5)
        for _ in range(n_epochs):
            head.train()
            perm = torch.randperm(train_data[data_key].size(0))
            for i in range(0, len(perm), 128):
                idx = perm[i:i+128]
                x = train_data[data_key][idx].to(device)
                s = train_data["states"][idx].to(device).long()
                mask = train_data["mask"][idx].to(device)
                logits = head(x)
                loss = F.cross_entropy(logits.reshape(-1, N_STATES), s.reshape(-1), reduction="none")
                loss = (loss * mask.reshape(-1)).sum() / mask.sum()
                opt.zero_grad(); loss.backward(); opt.step()

    gnn_head = nn.Linear(GNN_DIM, N_STATES).to(device)
    _train_baseline(gnn_head, "gnn", train_data, device)

    pomdp_head = nn.Sequential(nn.Linear(POMDP_DIM, 32), nn.GELU(), nn.Linear(32, N_STATES)).to(device)
    _train_baseline(pomdp_head, "pomdp", train_data, device)

    mamba_head = nn.Linear(NODE_FEAT_DIM, N_STATES).to(device)
    _train_baseline(mamba_head, "mamba", train_data, device)

    # Train fusion
    print(f"\n  {C_INFO}Training shared state space...{C_RESET}")
    print(f"  {C_DIM}{'ep':>6s}  {'loss':>10s}  {'acc':>8s}  [hlthy  dgrad  faild  unrch]{C_RESET}")
    print(f"  {C_DIM}{'─' * 60}{C_RESET}")

    t0 = time.time()
    train_fusion(shared_space, train_data, val_data, device, epochs=80)
    train_time = time.time() - t0

    # ── Evaluate all on validation set ──
    print(f"\n  {C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║              FUSION RESULTS                        ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════╝{C_RESET}")

    def eval_model(model_fn, val_data, device):
        class_tp = torch.zeros(N_STATES)
        class_fn = torch.zeros(N_STATES)
        class_fp = torch.zeros(N_STATES)
        with torch.no_grad():
            for i in range(0, val_data["states"].size(0), 128):
                s = val_data["states"][i:i+128].to(device).long()
                mask = val_data["mask"][i:i+128].to(device)
                logits = model_fn(i, i + 128, device)
                preds = logits.argmax(dim=-1)
                valid = mask > 0
                for c in range(N_STATES):
                    ct = (s == c) & valid
                    cp = (preds == c) & valid
                    class_tp[c] += (ct & cp).sum().item()
                    class_fp[c] += (~ct & cp).sum().item()
                    class_fn[c] += (ct & ~cp).sum().item()
        results = {}
        for c in range(N_STATES):
            p = class_tp[c] / max(class_tp[c] + class_fp[c], 1)
            r = class_tp[c] / max(class_tp[c] + class_fn[c], 1)
            f1 = 2 * p * r / max(p + r, 1e-8)
            results[STATE_NAMES[c]] = {"f1": float(f1), "recall": float(r), "n": int(class_tp[c] + class_fn[c])}
        results["macro_f1"] = sum(results[s]["f1"] for s in STATE_NAMES) / N_STATES
        return results

    gnn_head.eval(); pomdp_head.eval(); mamba_head.eval(); shared_space.eval()

    gnn_results = eval_model(
        lambda i, j, d: gnn_head(val_data["gnn"][i:j].to(d)),
        val_data, device
    )
    pomdp_results = eval_model(
        lambda i, j, d: pomdp_head(val_data["pomdp"][i:j].to(d)),
        val_data, device
    )
    mamba_results = eval_model(
        lambda i, j, d: mamba_head(val_data["mamba"][i:j].to(d)),
        val_data, device
    )
    fusion_results = eval_model(
        lambda i, j, d: shared_space(
            val_data["gnn"][i:j].to(d),
            val_data["pomdp"][i:j].to(d),
            val_data["mamba"][i:j].to(d),
        )["state_logits"],
        val_data, device
    )

    # Display
    print(f"\n  {C_DIM}{'Model':<18s} {'Macro F1':>8s}", end="")
    for s in STATE_NAMES:
        print(f" {s[:6]:>8s}", end="")
    print(f"{C_RESET}")
    print(f"  {C_DIM}{'─' * 60}{C_RESET}")

    all_results = [
        ("GNN only", gnn_results),
        ("POMDP only", pomdp_results),
        ("Mamba only", mamba_results),
        ("FUSED (shared)", fusion_results),
    ]

    for name, res in all_results:
        is_fused = "FUSED" in name
        c = C_SUCCESS if is_fused else C_TEXT
        b = C_BOLD if is_fused else ""
        line = f"  {c}{b}{name:<18s}{C_RESET} {res['macro_f1']:8.4f}"
        for s in STATE_NAMES:
            line += f" {res[s]['f1']:8.4f}"
        print(line)

    # Deltas
    print(f"\n  {C_INFO}Fusion vs best single pillar:{C_RESET}")
    for s in STATE_NAMES:
        best_single = max(gnn_results[s]["f1"], pomdp_results[s]["f1"], mamba_results[s]["f1"])
        fused_f1 = fusion_results[s]["f1"]
        delta = fused_f1 - best_single
        if gnn_results[s]["f1"] == best_single:
            which = "GNN"
        elif pomdp_results[s]["f1"] == best_single:
            which = "POMDP"
        else:
            which = "Mamba"

        if delta > 0.02:
            c = C_SUCCESS
        elif delta < -0.02:
            c = C_DANGER
        else:
            c = C_DIM
        print(f"    {c}{s:<15s} fused={fused_f1:.4f} best_single={best_single:.4f} ({which}) delta={delta:+.4f}{C_RESET}")

    macro_best = max(gnn_results["macro_f1"], pomdp_results["macro_f1"], mamba_results["macro_f1"])
    macro_fused = fusion_results["macro_f1"]
    macro_delta = macro_fused - macro_best

    # Pillar weights analysis
    print(f"\n  {C_INFO}Pillar attention weights (how much each pillar was trusted):{C_RESET}")
    with torch.no_grad():
        sample_g = val_data["gnn"][:100].to(device)
        sample_p = val_data["pomdp"][:100].to(device)
        sample_m = val_data["mamba"][:100].to(device)
        out = shared_space(sample_g, sample_p, sample_m)
        weights = out["pillar_weights"]  # (B, N, 3)
        mask = val_data["mask"][:100].to(device)
        # Average weights across valid nodes
        w_sum = (weights * mask.unsqueeze(-1)).sum(dim=(0, 1))
        w_count = mask.sum() * 1.0
        avg_weights = w_sum / w_count
        print(f"    GNN:   {avg_weights[0].item():.4f}")
        print(f"    POMDP: {avg_weights[1].item():.4f}")
        print(f"    Mamba: {avg_weights[2].item():.4f}")

    # Verdict
    print(f"\n  {C_GOLD}{C_BOLD}  VERDICT{C_RESET}")
    print(f"  {C_DIM}{'─' * 50}{C_RESET}")
    print(f"  {C_TEXT}Macro F1: fused={macro_fused:.4f} vs best_single={macro_best:.4f} ({macro_delta:+.4f}){C_RESET}")
    print(f"  {C_TEXT}Training time: {train_time:.1f}s{C_RESET}")
    print(f"  {C_TEXT}Fusion params: {params['total']:,}{C_RESET}")

    wins = sum(1 for s in STATE_NAMES
               if fusion_results[s]["f1"] > max(
                   gnn_results[s]["f1"], pomdp_results[s]["f1"], mamba_results[s]["f1"]
               ) + 0.01)

    if macro_delta > 0.02 and wins >= 2:
        print(f"\n  {C_SUCCESS}{C_BOLD}FUSION VALIDATED.{C_RESET}")
        print(f"  {C_SUCCESS}The shared latent space improves on the best single pillar.{C_RESET}")
        print(f"  {C_SUCCESS}Three perspectives, one truth. The architecture thesis holds.{C_RESET}")
    elif macro_delta > 0 and wins >= 1:
        print(f"\n  {C_GOLD}{C_BOLD}PARTIAL FUSION.{C_RESET}")
        print(f"  {C_GOLD}Some metrics improve. The signal exists but isn't dominant.{C_RESET}")
    else:
        print(f"\n  {C_DANGER}{C_BOLD}FUSION NOT DEMONSTRATED.{C_RESET}")
        print(f"  {C_TEXT}The shared space doesn't beat the best individual pillar.{C_RESET}")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    run_fusion_experiment(args.device, args.samples)
