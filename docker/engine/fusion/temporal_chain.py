#!/usr/bin/env -S python -u
"""
Project PARALLAX — Temporal Chain: Verdict Feedback Loop
==========================================================
Wires the fusion output back to the next inference cycle.
Verdict at t becomes input at t+1. Context accumulates.
The system doesn't just classify — it tracks trajectories.

Components:
  - TemporalState: dataclass persisting between cycles (~160 KB)
  - ContextMixer: augments pillar inputs with prior context (residual gated)
  - RevisionGate: compares current verdict to trajectory, revises if warranted
  - TemporalChainFusion: wraps SharpRoutedFusion without modifying it

Total new parameters: ~72K. System goes from 9.2M → 9.27M.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar3"))

from sable_sim.core.states import N_STATES
from shared_latent_space import Z_DIM, GNN_DIM, POMDP_DIM
from generate_temporal_data import NODE_FEAT_DIM

MAMBA_DIM = NODE_FEAT_DIM  # 25
TRAJ_K = 8      # History depth: last 8 verdicts per node
TRAJ_FEAT = 5   # Per-verdict: [state, confidence, cycle_norm, changed, direction]
BOTTLENECK = 32  # TemporalGate bottleneck dim

# Feature clamp range — anything outside this is sensor garbage
_FEAT_CLAMP = 1e6


def _sanitize(x: torch.Tensor) -> torch.Tensor:
    """Replace NaN with 0, clamp Inf to finite range. Zero-cost on clean data."""
    return torch.nan_to_num(x, nan=0.0, posinf=_FEAT_CLAMP, neginf=-_FEAT_CLAMP)


# ── Temporal State ────────────────────────────────────────────────────────


@dataclass
class TemporalState:
    """Persists between inference cycles. One per monitored topology."""

    # Mamba SSM hidden states — one per MambaBlock layer
    mamba_h: list[torch.Tensor] | None = None  # list of (1, d_inner, d_state)

    # Fused latent Z from previous cycle
    prev_z: torch.Tensor | None = None  # (1, N, Z_DIM=128)

    # Per-node verdict trajectory ring buffer
    trajectory: torch.Tensor | None = None  # (N, TRAJ_K, TRAJ_FEAT)

    # POMDP belief vectors carried forward
    belief_state: torch.Tensor | None = None  # (N, POMDP_DIM)

    # Cycle counter
    cycle: int = 0

    # Ring buffer write pointer per node
    traj_ptr: torch.Tensor | None = None  # (N,) int

    @staticmethod
    def cold_start(n_nodes: int, device: torch.device = torch.device("cpu")) -> "TemporalState":
        """Initialize empty temporal state for a topology."""
        return TemporalState(
            mamba_h=None,
            prev_z=torch.zeros(1, n_nodes, Z_DIM, device=device),
            trajectory=torch.zeros(n_nodes, TRAJ_K, TRAJ_FEAT, device=device),
            belief_state=torch.zeros(n_nodes, POMDP_DIM, device=device),
            cycle=0,
            traj_ptr=torch.zeros(n_nodes, dtype=torch.long, device=device),
        )

    def update(self, logits: torch.Tensor, confidence: torch.Tensor,
               z_fused: torch.Tensor | None = None,
               mamba_h: list[torch.Tensor] | None = None):
        """Update state after an inference cycle.

        Args:
            logits: (1, N, 4) — current verdict logits
            confidence: (1, N, 1) — revision confidence
            z_fused: (1, N, 128) — fused latent from this cycle
            mamba_h: list of SSM hidden states from this cycle
        """
        N = logits.size(1)
        device = logits.device

        # Initialize trajectory on first call if needed
        if self.trajectory is None or self.trajectory.size(0) != N:
            self.trajectory = torch.zeros(N, TRAJ_K, TRAJ_FEAT, device=device)
            self.traj_ptr = torch.zeros(N, dtype=torch.long, device=device)

        # Current verdict
        state_class = logits[0].argmax(dim=-1).float()  # (N,)
        conf = confidence[0].squeeze(-1)  # (N,)
        cycle_norm = min(self.cycle / 100.0, 1.0)

        # Detect change from previous verdict
        if self.cycle > 0:
            prev_ptr = (self.traj_ptr - 1) % TRAJ_K
            prev_state = self.trajectory[torch.arange(N, device=device), prev_ptr, 0]
            changed = (state_class != prev_state).float()
            # Direction: positive = deteriorating (state index increasing)
            direction = torch.sign(state_class - prev_state)
        else:
            changed = torch.zeros(N, device=device)
            direction = torch.zeros(N, device=device)

        # Write to ring buffer
        entry = torch.stack([
            state_class,           # 0: state class
            conf,                  # 1: confidence
            torch.full((N,), cycle_norm, device=device),  # 2: normalized cycle
            changed,               # 3: changed flag
            direction,             # 4: direction [-1, 0, 1]
        ], dim=-1)  # (N, 5)

        # Clone trajectory to avoid in-place modification during BPTT
        new_traj = self.trajectory.clone()
        new_traj[torch.arange(N, device=device), self.traj_ptr, :] = entry
        self.trajectory = new_traj
        self.traj_ptr = (self.traj_ptr + 1) % TRAJ_K

        # Update other state
        if z_fused is not None:
            self.prev_z = z_fused.detach()
        if mamba_h is not None:
            self.mamba_h = [h.detach() for h in mamba_h]

        self.cycle += 1


# ── Temporal Gate ─────────────────────────────────────────────────────────


class TemporalGate(nn.Module):
    """Residual gated injection of temporal context.

    output = x + sigmoid(gate) * temporal_signal

    Gate initialized near-zero (bias=-3) so model starts equivalent
    to stateless. Learns when temporal context helps.
    """

    def __init__(self, native_dim: int, traj_dim: int, z_dim: int = Z_DIM,
                 bottleneck: int = BOTTLENECK):
        super().__init__()
        context_dim = traj_dim + z_dim

        self.context_linear1 = nn.Linear(context_dim, bottleneck)
        self.context_linear2 = nn.Linear(bottleneck, native_dim)
        self.gate = nn.Sequential(
            nn.Linear(native_dim + bottleneck, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, native_dim),
        )
        # Initialize gate bias negative → sigmoid ≈ 0.05 at init
        nn.init.constant_(self.gate[-1].bias, -3.0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, native_dim) — current pillar output
            context: (B, N, traj_dim + z_dim) — temporal context
        """
        context_bottleneck = F.gelu(self.context_linear1(context))  # (B, N, bottleneck)
        temporal_signal = self.context_linear2(context_bottleneck)  # (B, N, native_dim)

        gate_input = torch.cat([x, context_bottleneck], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))

        return x + gate * temporal_signal


# ── Context Mixer ─────────────────────────────────────────────────────────


class ContextMixer(nn.Module):
    """Augments current pillar outputs with temporal context.

    For each pillar: concatenates trajectory summary + prev_z as context,
    then applies residual gated injection.
    """

    def __init__(self, traj_k: int = TRAJ_K, traj_feat: int = TRAJ_FEAT):
        super().__init__()
        traj_summary_dim = BOTTLENECK  # 32

        self.traj_encoder = nn.Sequential(
            nn.Linear(traj_k * traj_feat, 64),
            nn.GELU(),
            nn.Linear(64, traj_summary_dim),
        )

        context_dim = traj_summary_dim + Z_DIM  # 32 + 128 = 160
        self.gnn_gate = TemporalGate(GNN_DIM, traj_summary_dim)
        self.pomdp_gate = TemporalGate(POMDP_DIM, traj_summary_dim)
        self.mamba_gate = TemporalGate(MAMBA_DIM, traj_summary_dim)

    def forward(self, gnn_raw: torch.Tensor, pomdp_raw: torch.Tensor,
                mamba_raw: torch.Tensor, temporal_state: TemporalState | None):
        """
        Returns augmented versions of each pillar's output (same shapes).
        On cold start (temporal_state=None), passes through unchanged.
        """
        if temporal_state is None or temporal_state.cycle == 0:
            return gnn_raw, pomdp_raw, mamba_raw

        B, N, _ = gnn_raw.shape
        device = gnn_raw.device

        # Encode trajectory: (N, K*5) → (N, 32)
        traj_flat = temporal_state.trajectory.reshape(N, -1).to(device)
        traj_summary = self.traj_encoder(traj_flat)  # (N, 32)
        traj_summary = traj_summary.unsqueeze(0).expand(B, -1, -1)  # (B, N, 32)

        prev_z = temporal_state.prev_z.to(device).expand(B, -1, -1)  # (B, N, 128)
        context = torch.cat([traj_summary, prev_z], dim=-1)  # (B, N, 160)

        gnn_aug = self.gnn_gate(gnn_raw, context)
        pomdp_aug = self.pomdp_gate(pomdp_raw, context)
        mamba_aug = self.mamba_gate(mamba_raw, context)

        return gnn_aug, pomdp_aug, mamba_aug


# ── Revision Gate ─────────────────────────────────────────────────────────


class RevisionGate(nn.Module):
    """Compares current verdict against trajectory history.

    Produces:
      - revised_logits: (B, N, N_STATES) — revision-adjusted predictions
      - confidence: (B, N, 1) — calibrated confidence (stability-aware)
      - transition: (B, N, 3) — [improving, stable, deteriorating]
    """

    # Stability features derived from trajectory (not learned — computed directly)
    # [0] n_flips / K          — flip rate (0=stable, 1=every tick)
    # [1] ticks_in_current     — how long in current state / K
    # [2] avg_past_confidence  — mean of past confidence values
    # [3] max_logit_margin     — how decisive current prediction is
    N_STABILITY_FEATS = 4

    def __init__(self, traj_k: int = TRAJ_K, traj_feat: int = TRAJ_FEAT):
        super().__init__()
        base_dim = N_STATES + traj_k * traj_feat + Z_DIM
        conf_dim = base_dim + self.N_STABILITY_FEATS

        self.revision_head = nn.Sequential(
            nn.Linear(base_dim, 64),
            nn.GELU(),
            nn.Linear(64, N_STATES),
        )

        # Confidence head gets extra stability features — direct signal for calibration
        self.confidence_head = nn.Sequential(
            nn.Linear(conf_dim, 48),
            nn.GELU(),
            nn.Linear(48, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.transition_head = nn.Sequential(
            nn.Linear(base_dim, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

    def _compute_stability(self, current_logits: torch.Tensor,
                           trajectory: torch.Tensor) -> torch.Tensor:
        """Compute explicit stability features from trajectory. No learning needed.

        Returns: (N, N_STABILITY_FEATS)
        """
        N = trajectory.size(0)
        device = trajectory.device
        K = trajectory.size(1)

        states = trajectory[:, :, 0]      # (N, K) — past state classes
        confidences = trajectory[:, :, 1]  # (N, K) — past confidences
        changed = trajectory[:, :, 3]      # (N, K) — change flags

        # [0] Flip rate: how often has this node changed state?
        flip_rate = changed.sum(dim=1) / max(K, 1)  # (N,)

        # [1] Ticks in current state: how long since last change?
        # Count backward from most recent entry
        current_state = current_logits.argmax(dim=-1).float()  # (N,) from (B=1, N, S)
        if current_state.dim() > 1:
            current_state = current_state[0]
        ticks_same = torch.zeros(N, device=device)
        for k in range(K - 1, -1, -1):
            still_same = (states[:, k] == current_state) & (trajectory[:, k, :].abs().sum(dim=-1) > 0)
            ticks_same += still_same.float()
            # Stop counting once we hit a different state
            break_mask = (states[:, k] != current_state) & (trajectory[:, k, :].abs().sum(dim=-1) > 0)
            ticks_same[break_mask] = ticks_same[break_mask]  # freeze
        ticks_in_current = ticks_same / max(K, 1)

        # [2] Average past confidence
        valid_mask = trajectory[:, :, :].abs().sum(dim=-1) > 0  # (N, K)
        conf_sum = (confidences * valid_mask.float()).sum(dim=1)
        conf_count = valid_mask.float().sum(dim=1).clamp(min=1)
        avg_confidence = conf_sum / conf_count

        # [3] Current prediction decisiveness (margin between top 2 logits)
        if current_logits.dim() == 3:
            logits_2d = current_logits[0]  # (N, S)
        else:
            logits_2d = current_logits
        sorted_logits, _ = logits_2d.sort(dim=-1, descending=True)
        margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).clamp(0, 10) / 10.0

        return torch.stack([flip_rate, ticks_in_current, avg_confidence, margin], dim=-1)

    def forward(self, current_logits: torch.Tensor, trajectory: torch.Tensor,
                prev_z: torch.Tensor):
        """
        Args:
            current_logits: (B, N, N_STATES) from fusion
            trajectory: (N, K, 5)
            prev_z: (1, N, 128)
        """
        B, N, _ = current_logits.shape
        device = current_logits.device

        traj_flat = trajectory.reshape(N, -1).to(device)
        traj_flat = traj_flat.unsqueeze(0).expand(B, -1, -1)
        pz = prev_z.to(device).expand(B, -1, -1)

        base = torch.cat([current_logits, traj_flat, pz], dim=-1)

        # Stability features for confidence calibration
        stability = self._compute_stability(current_logits, trajectory)
        stability = stability.unsqueeze(0).expand(B, -1, -1)  # (B, N, 4)
        conf_input = torch.cat([base, stability], dim=-1)

        revised = self.revision_head(base)
        confidence = self.confidence_head(conf_input)
        transition = self.transition_head(base)

        # Blend: confidence determines how much revision matters
        final_logits = confidence * revised + (1 - confidence) * current_logits

        return final_logits, confidence, transition


# ── Temporal Chain Fusion ─────────────────────────────────────────────────


class TemporalChainFusion(nn.Module):
    """Wraps SharpRoutedFusion with temporal chaining.

    Does NOT modify the base fusion — wraps it with ContextMixer (before)
    and RevisionGate (after). The base fusion's trained weights are preserved.

    Forward signature is backward-compatible: without temporal_state,
    output matches base fusion exactly.
    """

    def __init__(self, base_fusion: nn.Module, traj_k: int = TRAJ_K):
        super().__init__()
        self.base_fusion = base_fusion
        self.context_mixer = ContextMixer(traj_k=traj_k)
        self.revision_gate = RevisionGate(traj_k=traj_k)
        self.traj_k = traj_k

    def forward(self, gnn_raw: torch.Tensor, pomdp_raw: torch.Tensor,
                mamba_raw: torch.Tensor, temporal_state: TemporalState | None = None,
                hard_route: bool = False) -> dict:
        """
        Args:
            gnn_raw: (B, N, GNN_DIM)
            pomdp_raw: (B, N, POMDP_DIM)
            mamba_raw: (B, N, MAMBA_DIM)
            temporal_state: TemporalState from previous cycle, or None for cold start
            hard_route: passed to base fusion

        Returns:
            dict with all base fusion outputs plus:
              'revised_logits': (B, N, 4) — temporal-revised predictions
              'confidence': (B, N, 1) — revision confidence
              'transition': (B, N, 3) — [improving, stable, deteriorating]
              'temporal_state': TemporalState for next cycle (call .update() yourself
                               or use update_temporal_state helper)
        """
        # Step 0: Sanitize inputs — kill NaN/Inf before they propagate
        gnn_raw = _sanitize(gnn_raw)
        pomdp_raw = _sanitize(pomdp_raw)
        mamba_raw = _sanitize(mamba_raw)

        # Step 1: Augment inputs with temporal context
        gnn_aug, pomdp_aug, mamba_aug = self.context_mixer(
            gnn_raw, pomdp_raw, mamba_raw, temporal_state
        )

        # Step 2: Run base fusion (architecture unchanged)
        base_out = self.base_fusion(gnn_aug, pomdp_aug, mamba_aug, hard_route)

        # Step 3: Revision gate — compare verdict to trajectory
        if temporal_state is not None and temporal_state.cycle > 0:
            revised_logits, confidence, transition = self.revision_gate(
                base_out["logits"], temporal_state.trajectory, temporal_state.prev_z
            )
        else:
            revised_logits = base_out["logits"]
            B, N = base_out["logits"].shape[:2]
            device = base_out["logits"].device
            confidence = torch.ones(B, N, 1, device=device) * 0.5
            transition = torch.zeros(B, N, 3, device=device)

        # Step 4: Extract z_fused for state storage
        # Re-compute from projections (base fusion doesn't return z_fused)
        z_gnn = self.base_fusion.gnn_proj(gnn_aug)
        z_pomdp = self.base_fusion.pomdp_proj(pomdp_aug)
        z_mamba = self.base_fusion.mamba_proj(mamba_aug)
        perspectives = torch.stack([z_gnn, z_pomdp, z_mamba], dim=2)
        z_fused = self.base_fusion.fusion(perspectives)  # (B, N, Z_DIM)

        return {
            **base_out,
            "revised_logits": revised_logits,
            "confidence": confidence,
            "transition": transition,
            "z_fused": z_fused,
        }

    def n_temporal_params(self) -> int:
        """Count only temporal chain parameters (not base fusion)."""
        return (sum(p.numel() for p in self.context_mixer.parameters())
                + sum(p.numel() for p in self.revision_gate.parameters()))
