#!/usr/bin/env python3
"""
Project PARALLAX — Pillar 3: Mamba Cascade Predictor (Final)
==============================================================
Given the first 2 ticks of a cascade, predict cascade outcome.

Fixes from v3:
  - Delta features: explicit change signal between ticks
  - Focal loss for state classification
  - Two-stage prediction: affected detection → state classification
  - Stronger regularization against overfitting

Usage:
    python sable_mamba_final.py --data temporal_v3.pt --epochs 100
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent))
from sable_sim.core.states import N_STATES, STATE_NAMES
from sable_mamba import SelectiveSSM, MambaBlock

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               gamma: float = 2.0, weights: torch.Tensor = None,
               reduction: str = "none") -> torch.Tensor:
    """Focal loss for multi-class classification."""
    ce = F.cross_entropy(logits, targets, weight=weights, reduction="none")
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    if reduction == "mean":
        return focal.mean()
    return focal


class SableMambaFinal(nn.Module):
    """Final Mamba cascade predictor with delta features and two-stage heads."""

    def __init__(
        self,
        state_dim: int = 1000,
        max_nodes: int = 40,
        node_feat_dim: int = 25,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        n_states: int = N_STATES,
        input_ticks: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.max_nodes = max_nodes
        self.n_states = n_states

        # Input: original ticks + delta features
        # For 2 input ticks: [tick0, tick1, tick1-tick0] = 3 * state_dim
        augmented_dim = state_dim * (input_ticks + 1)  # +1 for delta

        # Input projection with bottleneck
        self.input_proj = nn.Sequential(
            nn.Linear(augmented_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # Mamba encoder
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Head 1: Per-node affected detection (binary: affected vs healthy)
        self.affected_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, max_nodes),
        )

        # Head 2: Per-node state classification (4-class, for affected nodes)
        self.state_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, max_nodes * n_states),
        )

        # Head 3: Per-node health regression
        self.health_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, max_nodes),
            nn.Sigmoid(),
        )

        # Head 4: Cascade severity
        self.severity_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _encode(self, x: torch.Tensor, h_list: list[torch.Tensor] | None = None,
                return_state: bool = False):
        """Shared encoder logic for forward and forward_stateful."""
        B = x.size(0)

        # Compute delta features: difference between consecutive ticks
        deltas = x[:, -1, :] - x[:, 0, :]  # (B, state_dim) — total change

        # Concatenate: [all ticks flattened, delta]
        x_flat = x.reshape(B, -1)  # (B, T * state_dim)
        augmented = torch.cat([x_flat, deltas], dim=-1)  # (B, (T+1) * state_dim)

        # Project and add sequence dim for Mamba (expects 3D)
        h = self.input_proj(augmented).unsqueeze(1)  # (B, 1, d_model)

        # Mamba blocks — with optional state persistence
        h_finals = []
        for i, layer in enumerate(self.layers):
            h_init = h_list[i] if h_list is not None else None
            if return_state:
                h, h_final = layer(h, h_init=h_init, return_state=True)
                h_finals.append(h_final)
            else:
                h = layer(h, h_init=h_init)
        h = self.norm(h)

        summary = h.squeeze(1)  # (B, d_model)
        return summary, h_finals if return_state else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Stateless forward pass (backward compatible).

        Args:
            x: (B, T, state_dim)
        """
        summary, _ = self._encode(x)
        B = x.size(0)

        return {
            "affected_logits": self.affected_head(summary),
            "state_logits": self.state_head(summary).view(B, self.max_nodes, self.n_states),
            "health": self.health_head(summary),
            "severity": self.severity_head(summary).squeeze(-1),
        }

    def forward_stateful(self, x: torch.Tensor,
                         h_list: list[torch.Tensor] | None = None,
                         ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
        """Stateful forward pass for temporal chaining.

        Carries SSM hidden state across inference cycles.

        Args:
            x: (B, T, state_dim)
            h_list: list of (B, d_inner, d_state) per layer, or None for cold start.

        Returns:
            (outputs_dict, h_list_new) where h_list_new can be passed to next cycle.
        """
        summary, h_finals = self._encode(x, h_list=h_list, return_state=True)
        B = x.size(0)

        outputs = {
            "affected_logits": self.affected_head(summary),
            "state_logits": self.state_head(summary).view(B, self.max_nodes, self.n_states),
            "health": self.health_head(summary),
            "severity": self.severity_head(summary).squeeze(-1),
        }
        return outputs, h_finals


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.0005
    weight_decay: float = 1e-3
    patience: int = 20
    d_model: int = 256
    n_layers: int = 4
    d_state: int = 16
    dropout: float = 0.2
    focal_gamma: float = 2.0
    affected_weight: float = 1.0
    state_weight: float = 1.0
    health_weight: float = 0.5
    severity_weight: float = 0.3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train(data_path: str = "temporal_v3.pt", config: TrainConfig = None,
          checkpoint_dir: str = "checkpoints"):
    if config is None:
        config = TrainConfig()

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Pillar 3: Mamba Final{C_RESET}")
    print(f"  {C_DIM}{'═' * 50}{C_RESET}\n")

    data = torch.load(data_path, weights_only=False)
    X_input = data["X_input"]
    Y_nodes = data["Y_nodes"]
    Y_severity = data["Y_severity"]
    N_nodes = data["N_nodes"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    max_nodes = data["max_nodes"]
    state_dim = data["state_dim"]
    feat_dim = data["feat_dim"]
    input_ticks = data["input_ticks"]

    Y_health = Y_nodes[:, :, 0]
    Y_state = Y_nodes[:, :, 1].long()
    # Binary affected: 1 if not healthy (state != 0)
    Y_affected = (Y_state != 0).float()

    print(f"  {C_TEXT}samples:     {C_BRIGHT}{X_input.size(0)}{C_RESET}")
    print(f"  {C_TEXT}input ticks: {C_BRIGHT}{input_ticks}{C_RESET}")
    print(f"  {C_TEXT}state dim:   {C_BRIGHT}{state_dim}{C_RESET}")
    print(f"  {C_TEXT}max nodes:   {C_BRIGHT}{max_nodes}{C_RESET}")
    print(f"  {C_TEXT}device:      {C_BRIGHT}{config.device}{C_RESET}")

    # Compute class balance for weighted sampling
    # Oversample high-severity cascades
    sample_weights = torch.ones(len(train_idx))
    for i, idx in enumerate(train_idx):
        sev = Y_severity[idx].item()
        # Weight by severity: low severity gets 1x, high gets 5x
        sample_weights[i] = 1.0 + 4.0 * sev

    sampler = WeightedRandomSampler(sample_weights, len(train_idx), replacement=True)

    model = SableMambaFinal(
        state_dim=state_dim,
        max_nodes=max_nodes,
        node_feat_dim=feat_dim,
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_state=config.d_state,
        dropout=config.dropout,
        input_ticks=input_ticks,
    ).to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {C_TEXT}params:      {C_BRIGHT}{n_params:,}{C_RESET}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    # Cosine annealing instead of plateau — smoother, less aggressive
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    # State class weights for focal loss
    state_counts = torch.bincount(Y_state[train_idx].flatten(), minlength=N_STATES).float().clamp(min=1)
    state_weights = (Y_state[train_idx].numel() / (N_STATES * state_counts)).to(config.device)
    # Cap weights to prevent explosion on rare classes
    state_weights = state_weights.clamp(max=10.0)

    # Affected class weights
    n_affected = Y_affected[train_idx].sum().item()
    n_healthy = (Y_affected[train_idx] == 0).sum().item()
    affected_pos_weight = torch.tensor([n_healthy / max(n_affected, 1)], device=config.device).clamp(max=15.0)

    print(f"\n  {C_INFO}Class balance:{C_RESET}")
    print(f"    {C_DIM}healthy nodes:  {int(n_healthy)} ({n_healthy/(n_healthy+n_affected)*100:.1f}%){C_RESET}")
    print(f"    {C_DIM}affected nodes: {int(n_affected)} ({n_affected/(n_healthy+n_affected)*100:.1f}%){C_RESET}")
    print(f"    {C_DIM}affected pos_weight: {affected_pos_weight.item():.1f}x{C_RESET}")
    for i, name in enumerate(STATE_NAMES):
        if i < len(state_weights):
            print(f"    {C_DIM}{name}: weight={state_weights[i]:.2f}{C_RESET}")

    # DataLoaders
    train_ds = TensorDataset(
        X_input[train_idx], Y_health[train_idx], Y_state[train_idx],
        Y_affected[train_idx], Y_severity[train_idx], N_nodes[train_idx],
    )
    val_ds = TensorDataset(
        X_input[val_idx], Y_health[val_idx], Y_state[val_idx],
        Y_affected[val_idx], Y_severity[val_idx], N_nodes[val_idx],
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_val_metric = -1.0  # Track by affected F1, not loss
    patience_counter = 0

    print(f"\n  {C_DIM}{'ep':>4s} {'loss':>7s} {'aff_f1':>7s} {'st_acc':>7s} {'hlth':>7s} {'v_loss':>7s} {'v_aff':>7s} {'v_st':>7s} {'v_fail':>7s}{C_RESET}")
    print(f"  {C_DIM}{'─' * 72}{C_RESET}")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_aff_tp = 0
        epoch_aff_fp = 0
        epoch_aff_fn = 0
        epoch_n = 0

        for bx, bh, bs, ba, bsev, bn in train_loader:
            bx = bx.to(config.device)
            bh = bh.to(config.device)
            bs = bs.to(config.device)
            ba = ba.to(config.device)
            bsev = bsev.to(config.device)
            bn = bn.to(config.device)

            out = model(bx)

            # Node mask
            mask = torch.zeros(bx.size(0), max_nodes, device=config.device)
            for i, n in enumerate(bn):
                mask[i, :n] = 1.0

            # Loss 1: Affected detection (binary, focal-style with pos_weight)
            aff_bce = F.binary_cross_entropy_with_logits(
                out["affected_logits"], ba,
                pos_weight=affected_pos_weight.expand(max_nodes),
                reduction="none"
            )
            # Focal modulation
            aff_probs = torch.sigmoid(out["affected_logits"])
            pt = ba * aff_probs + (1 - ba) * (1 - aff_probs)
            focal_w = (1 - pt) ** config.focal_gamma
            aff_loss = (aff_bce * focal_w * mask).sum() / mask.sum()

            # Loss 2: State classification (focal loss)
            logits_flat = out["state_logits"].reshape(-1, N_STATES)
            targets_flat = bs.reshape(-1)
            mask_flat = mask.reshape(-1)
            fl = focal_loss(logits_flat, targets_flat, gamma=config.focal_gamma, weights=state_weights)
            state_loss = (fl * mask_flat).sum() / mask_flat.sum()

            # Loss 3: Health regression (MSE, masked)
            health_loss = ((out["health"] - bh) ** 2 * mask).sum() / mask.sum()

            # Loss 4: Severity
            severity_loss = F.mse_loss(out["severity"], bsev)

            loss = (
                config.affected_weight * aff_loss
                + config.state_weight * state_loss
                + config.health_weight * health_loss
                + config.severity_weight * severity_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track affected F1
            aff_preds = (out["affected_logits"] > 0).float()
            epoch_aff_tp += ((aff_preds == 1) & (ba == 1) & (mask == 1)).sum().item()
            epoch_aff_fp += ((aff_preds == 1) & (ba == 0) & (mask == 1)).sum().item()
            epoch_aff_fn += ((aff_preds == 0) & (ba == 1) & (mask == 1)).sum().item()
            epoch_loss += loss.item() * mask.sum().item()
            epoch_n += mask.sum().item()

        train_loss = epoch_loss / max(epoch_n, 1)
        aff_prec = epoch_aff_tp / max(epoch_aff_tp + epoch_aff_fp, 1)
        aff_rec = epoch_aff_tp / max(epoch_aff_tp + epoch_aff_fn, 1)
        train_aff_f1 = 2 * aff_prec * aff_rec / max(aff_prec + aff_rec, 1e-8)

        scheduler.step()

        # Validate
        model.eval()
        val_aff_tp = val_aff_fp = val_aff_fn = 0
        val_state_correct = val_state_total = 0
        val_fail_correct = val_fail_total = 0

        with torch.no_grad():
            for bx, bh, bs, ba, bsev, bn in val_loader:
                bx = bx.to(config.device)
                bh = bh.to(config.device)
                bs = bs.to(config.device)
                ba = ba.to(config.device)
                bsev = bsev.to(config.device)

                out = model(bx)
                mask = torch.zeros(bx.size(0), max_nodes, device=config.device)
                for i, n in enumerate(bn):
                    mask[i, :n] = 1.0

                # Affected
                aff_preds = (out["affected_logits"] > 0).float()
                val_aff_tp += ((aff_preds == 1) & (ba == 1) & (mask == 1)).sum().item()
                val_aff_fp += ((aff_preds == 1) & (ba == 0) & (mask == 1)).sum().item()
                val_aff_fn += ((aff_preds == 0) & (ba == 1) & (mask == 1)).sum().item()

                # State accuracy (overall + failed-class)
                state_preds = out["state_logits"].argmax(dim=-1)
                val_state_correct += ((state_preds == bs) * mask).sum().item()
                val_state_total += mask.sum().item()

                # Failed-class recall
                fail_mask = (bs == 2) & (mask > 0)
                val_fail_total += fail_mask.sum().item()
                val_fail_correct += ((state_preds == 2) & fail_mask).sum().item()

        val_aff_prec = val_aff_tp / max(val_aff_tp + val_aff_fp, 1)
        val_aff_rec = val_aff_tp / max(val_aff_tp + val_aff_fn, 1)
        val_aff_f1 = 2 * val_aff_prec * val_aff_rec / max(val_aff_prec + val_aff_rec, 1e-8)
        val_state_acc = val_state_correct / max(val_state_total, 1)
        val_fail_rec = val_fail_correct / max(val_fail_total, 1)

        improved = val_aff_f1 > best_val_metric + 0.001
        marker = f"{C_SUCCESS}*{C_RESET}" if improved else " "

        if epoch <= 5 or epoch % 5 == 0 or improved or epoch == config.epochs:
            print(
                f"  {C_TEXT}{epoch:4d}{C_RESET} "
                f"{train_loss:7.4f} "
                f"{train_aff_f1:7.4f} "
                f"{'—':>7s} "
                f"{'—':>7s} "
                f"{'—':>7s} "
                f"{val_aff_f1:7.4f} "
                f"{val_state_acc:7.4f} "
                f"{val_fail_rec:7.4f} {marker}"
            )

        if improved:
            best_val_metric = val_aff_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_aff_f1": val_aff_f1,
                "val_state_acc": val_state_acc,
                "val_fail_recall": val_fail_rec,
                "config": {
                    "state_dim": state_dim, "max_nodes": max_nodes,
                    "d_model": config.d_model, "n_layers": config.n_layers,
                    "d_state": config.d_state, "dropout": config.dropout,
                    "node_feat_dim": feat_dim, "input_ticks": input_ticks,
                },
                "n_params": n_params,
            }, checkpoint_path / "best_mamba_final.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\n  {C_DIM}Early stopping at epoch {epoch}{C_RESET}")
                break

    # ── Test ──
    print(f"\n  {C_GOLD}{C_BOLD}  Test Set Evaluation{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")

    ckpt = torch.load(checkpoint_path / "best_mamba_final.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = TensorDataset(
        X_input[test_idx], Y_health[test_idx], Y_state[test_idx],
        Y_affected[test_idx], Y_severity[test_idx], N_nodes[test_idx],
    )
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    # Per-class metrics
    class_tp = torch.zeros(N_STATES)
    class_fp = torch.zeros(N_STATES)
    class_fn = torch.zeros(N_STATES)
    aff_tp = aff_fp = aff_fn = aff_tn = 0
    health_errors = []

    with torch.no_grad():
        for bx, bh, bs, ba, bsev, bn in test_loader:
            bx = bx.to(config.device)
            out = model(bx)

            mask = torch.zeros(bx.size(0), max_nodes)
            for i, n in enumerate(bn):
                mask[i, :n] = 1.0

            # Affected
            aff_preds = (out["affected_logits"].cpu() > 0).float()
            aff_tp += ((aff_preds == 1) & (ba == 1) & (mask == 1)).sum().item()
            aff_fp += ((aff_preds == 1) & (ba == 0) & (mask == 1)).sum().item()
            aff_fn += ((aff_preds == 0) & (ba == 1) & (mask == 1)).sum().item()
            aff_tn += ((aff_preds == 0) & (ba == 0) & (mask == 1)).sum().item()

            # State
            state_preds = out["state_logits"].cpu().argmax(dim=-1)
            for c in range(N_STATES):
                c_true = (bs == c) & (mask > 0)
                c_pred = (state_preds == c) & (mask > 0)
                class_tp[c] += (c_true & c_pred).sum().item()
                class_fp[c] += (~c_true & c_pred).sum().item()
                class_fn[c] += (c_true & ~c_pred).sum().item()

            # Health
            h_err = ((out["health"].cpu() - bh) ** 2 * mask).sum() / mask.sum()
            health_errors.append(h_err.item())

    # Report
    print(f"\n  {C_INFO}Affected Node Detection (binary):{C_RESET}")
    aff_prec = aff_tp / max(aff_tp + aff_fp, 1)
    aff_rec = aff_tp / max(aff_tp + aff_fn, 1)
    aff_f1 = 2 * aff_prec * aff_rec / max(aff_prec + aff_rec, 1e-8)
    print(f"    {C_TEXT}Precision:  {C_BRIGHT}{aff_prec:.4f}{C_RESET}")
    print(f"    {C_TEXT}Recall:     {C_BRIGHT}{aff_rec:.4f}{C_RESET}")
    print(f"    {C_TEXT}F1:         {C_BRIGHT}{aff_f1:.4f}{C_RESET}")
    print(f"    {C_DIM}TP={int(aff_tp)} FP={int(aff_fp)} FN={int(aff_fn)} TN={int(aff_tn)}{C_RESET}")

    print(f"\n  {C_INFO}Per-State Classification:{C_RESET}")
    for c in range(N_STATES):
        prec = class_tp[c] / max(class_tp[c] + class_fp[c], 1)
        rec = class_tp[c] / max(class_tp[c] + class_fn[c], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        support = int(class_tp[c] + class_fn[c])
        bar = "█" * int(f1 * 25)
        print(f"    {C_DIM}{STATE_NAMES[c]:15s}{C_RESET} "
              f"P={C_TEXT}{prec:.3f}{C_RESET} "
              f"R={C_TEXT}{rec:.3f}{C_RESET} "
              f"F1={C_TEXT}{f1:.3f}{C_RESET} "
              f"(n={support:5d}) {C_GOLD}{bar}{C_RESET}")

    macro_f1 = sum(
        2 * (class_tp[c] / max(class_tp[c] + class_fp[c], 1)) *
        (class_tp[c] / max(class_tp[c] + class_fn[c], 1)) /
        max((class_tp[c] / max(class_tp[c] + class_fp[c], 1)) +
            (class_tp[c] / max(class_tp[c] + class_fn[c], 1)), 1e-8)
        for c in range(N_STATES)
    ) / N_STATES

    print(f"\n  {C_TEXT}Macro F1:    {C_BRIGHT}{macro_f1:.4f}{C_RESET}")
    print(f"  {C_TEXT}Health MSE:  {C_BRIGHT}{np.mean(health_errors):.6f}{C_RESET}")

    # Baselines
    print(f"\n  {C_INFO}Baselines:{C_RESET}")
    baseline_healthy_acc = (class_tp[0] + class_fn[0]) / sum(class_tp[c] + class_fn[c] for c in range(N_STATES))
    print(f"    {C_DIM}All-healthy accuracy: {baseline_healthy_acc:.4f} (catches 0% of failures){C_RESET}")
    fail_recall = class_tp[2] / max(class_tp[2] + class_fn[2], 1)
    print(f"    {C_DIM}Mamba failed-node recall: {fail_recall:.4f}{C_RESET}")

    if aff_f1 > 0.3:
        print(f"\n  {C_SUCCESS}{C_BOLD}TEMPORAL REASONING VALIDATED{C_RESET}")
        print(f"  {C_TEXT}Mamba predicts cascade outcomes from 2 ticks of early signal.{C_RESET}")
    else:
        print(f"\n  {C_DANGER}Affected F1 below threshold — needs more work{C_RESET}")

    print(f"\n  {C_SUCCESS}{C_BOLD}Best checkpoint:{C_RESET} {C_TEXT}{checkpoint_path / 'best_mamba_final.pt'}{C_RESET}")
    print(f"  {C_TEXT}Epoch: {ckpt['epoch']}  Affected F1: {ckpt['val_aff_f1']:.4f}{C_RESET}")
    print(f"  {C_TEXT}Parameters: {C_BRIGHT}{n_params:,}{C_RESET}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="temporal_v3.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, d_model=args.d_model, n_layers=args.n_layers,
        dropout=args.dropout, device=args.device,
    )
    train(data_path=args.data, config=config, checkpoint_dir=args.checkpoint_dir)
