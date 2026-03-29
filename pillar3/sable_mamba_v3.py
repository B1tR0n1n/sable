#!/usr/bin/env python3
"""
Project PARALLAX — Pillar 3: Mamba Cascade Predictor v3
=========================================================
Given the first 2 ticks of an infrastructure cascade, predict:
  1. Per-node outcome: will each node be healthy, degraded, or failed?
  2. Per-node health: what health value will each node have?
  3. Cascade severity: what fraction of nodes will be affected?

This is the correct training signal — temporal reasoning means
"given early signs, predict the full outcome."

Usage:
    python sable_mamba_v3.py --data temporal_v3.pt --epochs 80
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
from torch.utils.data import DataLoader, TensorDataset

from sable_mamba import SelectiveSSM, MambaBlock  # Reuse SSM from v1

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


# ── Model ──────────────────────────────────────────────────────────────────


class SableMambaV3(nn.Module):
    """Mamba-based cascade outcome predictor.

    Input: (B, T, state_dim) — first T ticks of a cascade
    Output: three prediction heads:
        - node_health: (B, max_nodes) — predicted health per node
        - node_state: (B, max_nodes, 4) — predicted state class logits per node
        - severity: (B, 1) — predicted cascade severity
    """

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
        dropout: float = 0.1,
        n_states: int = 4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.max_nodes = max_nodes
        self.node_feat_dim = node_feat_dim
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(state_dim, d_model)

        # Mamba temporal encoder
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Head 1: Per-node health prediction (regression)
        self.health_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, max_nodes),
            nn.Sigmoid(),  # Health is [0, 1]
        )

        # Head 2: Per-node state classification (4-class)
        self.state_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, max_nodes * n_states),
        )
        self.n_states = n_states

        # Head 3: Cascade severity prediction (scalar regression)
        self.severity_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Severity is [0, 1]
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, state_dim) — input sequence (first T ticks)

        Returns:
            dict with health, state_logits, severity predictions
        """
        B = x.size(0)

        # Project and encode
        h = self.input_proj(x)  # (B, T, d_model)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        # Use last position's encoding as the cascade summary
        summary = h[:, -1, :]  # (B, d_model)

        # Predictions
        health = self.health_head(summary)  # (B, max_nodes)
        state_logits = self.state_head(summary).view(B, self.max_nodes, self.n_states)  # (B, max_nodes, 4)
        severity = self.severity_head(summary).squeeze(-1)  # (B,)

        return {
            "health": health,
            "state_logits": state_logits,
            "severity": severity,
        }


# ── Training ──────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 15
    d_model: int = 256
    n_layers: int = 4
    d_state: int = 16
    dropout: float = 0.1
    health_weight: float = 1.0
    state_weight: float = 1.0
    severity_weight: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train(data_path: str = "temporal_v3.pt", config: TrainConfig = None,
          checkpoint_dir: str = "checkpoints"):
    if config is None:
        config = TrainConfig()

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Pillar 3: Cascade Predictor v3{C_RESET}")
    print(f"  {C_DIM}{'═' * 55}{C_RESET}\n")

    data = torch.load(data_path, weights_only=False)
    X_input = data["X_input"]       # (N, T, state_dim)
    Y_nodes = data["Y_nodes"]       # (N, max_nodes, 2) — [health, state_class]
    Y_severity = data["Y_severity"] # (N,)
    N_nodes = data["N_nodes"]       # (N,)
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    max_nodes = data["max_nodes"]
    state_dim = data["state_dim"]
    feat_dim = data["feat_dim"]

    print(f"  {C_TEXT}samples:    {C_BRIGHT}{X_input.size(0)}{C_RESET}")
    print(f"  {C_TEXT}input:      {C_BRIGHT}{X_input.shape}{C_RESET}")
    print(f"  {C_TEXT}max nodes:  {C_BRIGHT}{max_nodes}{C_RESET}")
    print(f"  {C_TEXT}state dim:  {C_BRIGHT}{state_dim}{C_RESET}")
    print(f"  {C_TEXT}device:     {C_BRIGHT}{config.device}{C_RESET}")

    # Split targets
    Y_health = Y_nodes[:, :, 0]      # (N, max_nodes) — health values
    Y_state = Y_nodes[:, :, 1].long() # (N, max_nodes) — state classes

    # Build model
    model = SableMambaV3(
        state_dim=state_dim,
        max_nodes=max_nodes,
        node_feat_dim=feat_dim,
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_state=config.d_state,
        dropout=config.dropout,
    ).to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {C_TEXT}params:     {C_BRIGHT}{n_params:,}{C_RESET}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    # DataLoaders
    train_ds = TensorDataset(
        X_input[train_idx], Y_health[train_idx], Y_state[train_idx],
        Y_severity[train_idx], N_nodes[train_idx],
    )
    val_ds = TensorDataset(
        X_input[val_idx], Y_health[val_idx], Y_state[val_idx],
        Y_severity[val_idx], N_nodes[val_idx],
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    # Class weights for state prediction (healthy is dominant)
    state_counts = torch.bincount(Y_state[train_idx].flatten(), minlength=4).float()
    state_counts = state_counts.clamp(min=1)
    state_weights = (Y_state[train_idx].numel() / (4 * state_counts)).to(config.device)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n  {C_DIM}{'epoch':>5s}  {'loss':>8s}  {'health':>8s}  {'state':>8s}  {'sever':>8s}  {'val_loss':>8s}  {'val_acc':>7s}{C_RESET}")
    print(f"  {C_DIM}{'─' * 65}{C_RESET}")

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = {"total": 0, "health": 0, "state": 0, "severity": 0, "n": 0}

        for bx, bh, bs, bsev, bn in train_loader:
            bx = bx.to(config.device)
            bh = bh.to(config.device)
            bs = bs.to(config.device)
            bsev = bsev.to(config.device)
            bn = bn.to(config.device)

            out = model(bx)

            # Node mask — only count real nodes
            mask = torch.zeros(bx.size(0), max_nodes, device=config.device)
            for i, n in enumerate(bn):
                mask[i, :n] = 1.0

            # Health loss (MSE, masked)
            health_loss = ((out["health"] - bh) ** 2 * mask).sum() / mask.sum()

            # State classification loss (CE, masked)
            logits_flat = out["state_logits"].view(-1, 4)  # (B*max_nodes, 4)
            targets_flat = bs.view(-1)  # (B*max_nodes,)
            mask_flat = mask.view(-1)  # (B*max_nodes,)
            ce_per_node = F.cross_entropy(logits_flat, targets_flat, weight=state_weights, reduction="none")
            state_loss = (ce_per_node * mask_flat).sum() / mask_flat.sum()

            # Severity loss (MSE)
            severity_loss = F.mse_loss(out["severity"], bsev)

            # Combined
            loss = (
                config.health_weight * health_loss
                + config.state_weight * state_loss
                + config.severity_weight * severity_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            b_size = mask.sum().item()
            losses["total"] += loss.item() * b_size
            losses["health"] += health_loss.item() * b_size
            losses["state"] += state_loss.item() * b_size
            losses["severity"] += severity_loss.item() * b_size
            losses["n"] += b_size

        n = losses["n"]
        train_loss = losses["total"] / n

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for bx, bh, bs, bsev, bn in val_loader:
                bx = bx.to(config.device)
                bh = bh.to(config.device)
                bs = bs.to(config.device)
                bsev = bsev.to(config.device)
                bn = bn.to(config.device)

                out = model(bx)

                mask = torch.zeros(bx.size(0), max_nodes, device=config.device)
                for i, nn_val in enumerate(bn):
                    mask[i, :nn_val] = 1.0

                health_loss = ((out["health"] - bh) ** 2 * mask).sum() / mask.sum()
                logits_flat = out["state_logits"].view(-1, 4)
                targets_flat = bs.view(-1)
                mask_flat = mask.view(-1)
                ce_per_node = F.cross_entropy(logits_flat, targets_flat, weight=state_weights, reduction="none")
                state_loss = (ce_per_node * mask_flat).sum() / mask_flat.sum()
                severity_loss = F.mse_loss(out["severity"], bsev)

                loss = (
                    config.health_weight * health_loss
                    + config.state_weight * state_loss
                    + config.severity_weight * severity_loss
                )
                val_loss_sum += loss.item() * mask.sum().item()

                # Accuracy on state prediction
                preds = out["state_logits"].argmax(dim=-1)  # (B, max_nodes)
                correct = ((preds == bs) * mask).sum().item()
                val_correct += correct
                val_total += mask.sum().item()

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss - 1e-5
        marker = f"{C_SUCCESS}*{C_RESET}" if improved else " "

        if epoch <= 5 or epoch % 5 == 0 or improved or epoch == config.epochs:
            print(
                f"  {C_TEXT}{epoch:5d}{C_RESET}  "
                f"{train_loss:8.4f}  "
                f"{losses['health']/n:8.4f}  "
                f"{losses['state']/n:8.4f}  "
                f"{losses['severity']/n:8.4f}  "
                f"{val_loss:8.4f}  "
                f"{val_acc:7.4f} {marker}"
            )

        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": {
                    "state_dim": state_dim, "max_nodes": max_nodes,
                    "d_model": config.d_model, "n_layers": config.n_layers,
                    "d_state": config.d_state, "dropout": config.dropout,
                    "node_feat_dim": feat_dim,
                },
                "n_params": n_params,
            }, checkpoint_path / "best_mamba_v3.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\n  {C_DIM}Early stopping at epoch {epoch}{C_RESET}")
                break

    # ── Test Evaluation ──
    print(f"\n  {C_GOLD}{C_BOLD}  Test Set Evaluation{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")

    ckpt = torch.load(checkpoint_path / "best_mamba_v3.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = TensorDataset(
        X_input[test_idx], Y_health[test_idx], Y_state[test_idx],
        Y_severity[test_idx], N_nodes[test_idx],
    )
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    # Per-class accuracy
    class_correct = torch.zeros(4)
    class_total = torch.zeros(4)
    health_errors = []
    severity_errors = []

    with torch.no_grad():
        for bx, bh, bs, bsev, bn in test_loader:
            bx = bx.to(config.device)
            out = model(bx)

            mask = torch.zeros(bx.size(0), max_nodes)
            for i, nn_val in enumerate(bn):
                mask[i, :nn_val] = 1.0

            preds = out["state_logits"].cpu().argmax(dim=-1)
            for c in range(4):
                c_mask = (bs == c) & (mask > 0)
                class_total[c] += c_mask.sum().item()
                class_correct[c] += ((preds == c) & c_mask).sum().item()

            health_err = ((out["health"].cpu() - bh) ** 2 * mask).sum() / mask.sum()
            health_errors.append(health_err.item())

            sev_err = ((out["severity"].cpu() - bsev) ** 2).mean()
            severity_errors.append(sev_err.item())

    state_names = ["healthy", "degraded", "failed", "unreachable"]
    print(f"\n  {C_INFO}Per-state classification accuracy:{C_RESET}")
    for c in range(4):
        acc = class_correct[c] / max(class_total[c], 1)
        bar = "█" * int(acc * 30)
        print(f"    {C_DIM}{state_names[c]:15s}{C_RESET} {C_TEXT}{acc:.3f}{C_RESET} "
              f"({int(class_correct[c])}/{int(class_total[c])}) {C_GOLD}{bar}{C_RESET}")

    overall_acc = class_correct.sum() / class_total.sum()
    mean_health_mse = np.mean(health_errors)
    mean_sev_mse = np.mean(severity_errors)

    print(f"\n  {C_TEXT}Overall state accuracy: {C_BRIGHT}{overall_acc:.4f}{C_RESET}")
    print(f"  {C_TEXT}Health MSE:             {C_BRIGHT}{mean_health_mse:.6f}{C_RESET}")
    print(f"  {C_TEXT}Severity MSE:           {C_BRIGHT}{mean_sev_mse:.6f}{C_RESET}")

    # Baseline: predict all healthy (majority class)
    baseline_acc = class_total[0] / class_total.sum()
    print(f"\n  {C_INFO}Baseline (predict all healthy):{C_RESET}")
    print(f"    {C_DIM}accuracy: {baseline_acc:.4f}{C_RESET}")

    if overall_acc > baseline_acc:
        delta = overall_acc - baseline_acc
        print(f"\n  {C_SUCCESS}{C_BOLD}MAMBA BEATS BASELINE by {delta:.4f} ({delta/baseline_acc*100:.1f}%){C_RESET}")
    else:
        print(f"\n  {C_DANGER}Mamba does not beat majority baseline{C_RESET}")

    print(f"\n  {C_SUCCESS}{C_BOLD}Best checkpoint:{C_RESET} {C_TEXT}{checkpoint_path / 'best_mamba_v3.pt'}{C_RESET}")
    print(f"  {C_TEXT}Epoch: {ckpt['epoch']}  Val acc: {ckpt['val_acc']:.4f}{C_RESET}")
    print(f"  {C_TEXT}Parameters: {C_BRIGHT}{n_params:,}{C_RESET}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="temporal_v3.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, d_model=args.d_model, n_layers=args.n_layers,
        device=args.device,
    )
    train(data_path=args.data, config=config, checkpoint_dir=args.checkpoint_dir)
