#!/usr/bin/env python3
"""
Project PARALLAX — Pillar 3: Mamba Temporal State Model
=========================================================
Pure-PyTorch implementation of a Mamba-style SSM for temporal
state prediction in infrastructure cascades.

Learns to predict system state at t+1, t+2, ... t+k from
current state sequence. Domain-agnostic: operates on abstract
state vectors, not infrastructure-specific features.

Uses selective state space model (S6) mechanics without custom
CUDA kernels — trades some speed for CUDA version compatibility.
Can swap in mamba-ssm optimized kernels later.

Usage:
    python sable_mamba.py                                 # Train with defaults
    python sable_mamba.py --data temporal_data.pt --epochs 100
    python sable_mamba.py --eval checkpoint.pt            # Evaluate only

Requires: ml-env with torch (CUDA)
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Terminal Colors ────────────────────────────────────────────────────────

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


# ── Selective SSM (Pure PyTorch) ──────────────────────────────────────────


class SelectiveSSM(nn.Module):
    """Selective State Space Model — the core of Mamba.

    Pure PyTorch implementation of the S6 selective scan.
    Input-dependent A, B, C matrices (selectivity).
    Linear complexity in sequence length.

    For reference: Mamba paper (Gu & Dao, 2023), Section 3.2.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: str = "auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        if dt_rank == "auto":
            self.dt_rank = max(1, d_model // 16)
        else:
            self.dt_rank = int(dt_rank)

        # Input projection: d_model → 2 * d_inner (split into x and z)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal 1D convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # SSM parameters — input-dependent (selective)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Fixed A parameter (discretized from continuous)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, h_init: torch.Tensor | None = None,
                return_state: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, L, d_model)
            h_init: Optional initial hidden state (B, d_inner, d_state).
                    If None, starts from zeros (stateless). If provided,
                    continues from prior state (temporal chaining).
            return_state: If True, returns (output, h_final) tuple.

        Returns:
            (B, L, d_model) if return_state=False
            ((B, L, d_model), (B, d_inner, d_state)) if return_state=True
        """
        B, L, _ = x.shape

        # Input projection and split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal convolution
        x_conv = x_inner.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # Trim to causal
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # SSM parameters from input (selectivity)
        x_ssm = self.x_proj(x_conv)  # (B, L, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = torch.split(
            x_ssm, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # dt projection and softplus
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative

        # Selective scan — now with optional state persistence
        y, h_final = self._selective_scan(x_conv, dt, A, B_ssm, C_ssm, h_init=h_init)

        # Skip connection
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        if return_state:
            return output, h_final
        return output

    def _selective_scan(
        self, u: torch.Tensor, dt: torch.Tensor,
        A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
        h_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Selective scan — the core SSM recurrence.

        Implements: h_t = A_bar * h_{t-1} + B_bar * u_t
                    y_t = C_t * h_t

        Pure PyTorch, sequential over time dimension.

        Args:
            h_init: Optional (B, d_inner, d_state) to continue from prior state.

        Returns:
            (output, h_final) — output: (B, L, d_inner), h_final: (B, d_inner, d_state)
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        # Initialize or continue hidden state
        if h_init is not None:
            h = h_init
        else:
            h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)

        outputs = []
        for t in range(seq_len):
            # Discretize: A_bar = exp(dt * A)
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            A_bar = torch.exp(dt_t * A.unsqueeze(0))  # (B, d_inner, d_state)

            # B_bar = dt * B
            B_t = B[:, t, :].unsqueeze(1)  # (B, 1, d_state)
            B_bar = dt_t * B_t  # (B, d_inner, d_state)

            # u_t
            u_t = u[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)

            # State update
            h = A_bar * h + B_bar * u_t  # (B, d_inner, d_state)

            # Output
            C_t = C[:, t, :].unsqueeze(1)  # (B, 1, d_state)
            y_t = (h * C_t).sum(dim=-1)  # (B, d_inner)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1), h  # (B, L, d_inner), (B, d_inner, d_state)


# ── Mamba Block ───────────────────────────────────────────────────────────


class MambaBlock(nn.Module):
    """A single Mamba block: LayerNorm → SSM → Residual."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h_init: torch.Tensor | None = None,
                return_state: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        if return_state:
            x, h_final = self.ssm(x, h_init=h_init, return_state=True)
            x = self.dropout(x)
            return x + residual, h_final
        else:
            x = self.ssm(x, h_init=h_init)
            x = self.dropout(x)
            return x + residual


# ── SableMamba Model ──────────────────────────────────────────────────────


class SableMamba(nn.Module):
    """Mamba-based temporal state predictor for SABLE Pillar 3.

    Given a sequence of system states [s_0, s_1, ..., s_t],
    predicts the next state s_{t+1}.

    Domain-agnostic: operates on abstract state vectors.
    """

    def __init__(
        self,
        state_dim: int = 1000,     # Input state vector dimension
        d_model: int = 256,        # Internal model dimension
        n_layers: int = 4,         # Number of Mamba blocks
        d_state: int = 16,         # SSM state dimension
        d_conv: int = 4,           # Convolution kernel size
        expand: int = 2,           # Expansion factor
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(state_dim, d_model)

        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        # Output projection — predict next state
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, state_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, state_dim) — sequence of system states

        Returns:
            (B, T, state_dim) — predicted next state at each position
        """
        # Project to model dim
        h = self.input_proj(x)  # (B, T, d_model)

        # Mamba blocks
        for layer in self.layers:
            h = layer(h)

        # Project to output
        h = self.output_norm(h)
        return self.output_proj(h)  # (B, T, state_dim)

    def predict_horizon(self, x: torch.Tensor, horizon: int = 5) -> list[torch.Tensor]:
        """Autoregressive multi-step prediction.

        Args:
            x: (B, T, state_dim) — initial sequence
            horizon: number of future steps to predict

        Returns:
            list of (B, state_dim) predictions for t+1, t+2, ... t+horizon
        """
        predictions = []
        current = x

        for _ in range(horizon):
            out = self.forward(current)
            # Take last position's prediction
            next_state = out[:, -1:, :]  # (B, 1, state_dim)
            predictions.append(next_state.squeeze(1))
            # Append to sequence for next step
            current = torch.cat([current, next_state], dim=1)

        return predictions


# ── Training ──────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 15
    d_model: int = 256
    n_layers: int = 4
    d_state: int = 16
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train(data_path: str = "temporal_data.pt", config: TrainConfig = None,
          checkpoint_dir: str = "checkpoints"):
    """Train SableMamba on temporal state sequences."""
    if config is None:
        config = TrainConfig()

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Pillar 3: Mamba Training{C_RESET}")
    print(f"  {C_DIM}{'═' * 50}{C_RESET}\n")

    # Load data
    data = torch.load(data_path, weights_only=False)
    X = data["X"]
    lengths = data["lengths"]
    state_dim = data["state_dim"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    print(f"  {C_TEXT}sequences:  {C_BRIGHT}{X.size(0)}{C_RESET}")
    print(f"  {C_TEXT}max ticks:  {C_BRIGHT}{X.size(1)}{C_RESET}")
    print(f"  {C_TEXT}state dim:  {C_BRIGHT}{state_dim}{C_RESET}")
    print(f"  {C_TEXT}device:     {C_BRIGHT}{config.device}{C_RESET}")

    # Build input/target pairs: input = states[:-1], target = states[1:]
    X_input = X[:, :-1, :]   # (N, T-1, state_dim)
    X_target = X[:, 1:, :]   # (N, T-1, state_dim)

    # Build model
    model = SableMamba(
        state_dim=state_dim,
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
    train_dataset = TensorDataset(X_input[train_idx], X_target[train_idx], lengths[train_idx])
    val_dataset = TensorDataset(X_input[val_idx], X_target[val_idx], lengths[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Training loop
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n  {C_DIM}{'epoch':>5s}  {'train_loss':>10s}  {'val_loss':>10s}  {'val_mse':>10s}  {'lr':>10s}{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")

    for epoch in range(1, config.epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_x, batch_y, batch_len in train_loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)
            batch_len = batch_len.to(config.device)

            pred = model(batch_x)

            # Masked loss — only count valid ticks
            mask = torch.zeros_like(pred[:, :, 0])
            for i, l in enumerate(batch_len):
                valid = min(l.item() - 1, pred.size(1))
                if valid > 0:
                    mask[i, :valid] = 1.0

            loss = ((pred - batch_y) ** 2 * mask.unsqueeze(-1)).sum()
            loss = loss / (mask.sum() * state_dim + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * mask.sum().item()
            train_count += mask.sum().item()

        train_loss = train_loss_sum / (train_count + 1e-8)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_mse_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch_x, batch_y, batch_len in val_loader:
                batch_x = batch_x.to(config.device)
                batch_y = batch_y.to(config.device)
                batch_len = batch_len.to(config.device)

                pred = model(batch_x)

                mask = torch.zeros_like(pred[:, :, 0])
                for i, l in enumerate(batch_len):
                    valid = min(l.item() - 1, pred.size(1))
                    if valid > 0:
                        mask[i, :valid] = 1.0

                loss = ((pred - batch_y) ** 2 * mask.unsqueeze(-1)).sum()
                loss = loss / (mask.sum() * state_dim + 1e-8)
                val_loss_sum += loss.item() * mask.sum().item()

                # MSE per valid position
                mse = ((pred - batch_y) ** 2).mean(dim=-1)  # (B, T)
                val_mse_sum += (mse * mask).sum().item()
                val_count += mask.sum().item()

        val_loss = val_loss_sum / (val_count + 1e-8)
        val_mse = val_mse_sum / (val_count + 1e-8)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Log
        improved = val_loss < best_val_loss - 1e-5
        marker = f"{C_SUCCESS}*{C_RESET}" if improved else " "

        if epoch <= 5 or epoch % 5 == 0 or improved or epoch == config.epochs:
            print(
                f"  {C_TEXT}{epoch:5d}{C_RESET}  "
                f"{train_loss:10.6f}  "
                f"{val_loss:10.6f}  "
                f"{val_mse:10.6f}  "
                f"{current_lr:10.6f} {marker}"
            )

        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_mse": val_mse,
                "config": {
                    "state_dim": state_dim,
                    "d_model": config.d_model,
                    "n_layers": config.n_layers,
                    "d_state": config.d_state,
                    "dropout": config.dropout,
                },
                "n_params": n_params,
            }, checkpoint_path / "best_mamba.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\n  {C_DIM}Early stopping at epoch {epoch}{C_RESET}")
                break

    # ── Evaluation ──
    print(f"\n  {C_GOLD}{C_BOLD}  Test Set Evaluation{C_RESET}")
    print(f"  {C_DIM}{'─' * 50}{C_RESET}")

    ckpt = torch.load(checkpoint_path / "best_mamba.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_dataset = TensorDataset(X_input[test_idx], X_target[test_idx], lengths[test_idx])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    test_mse_per_step = {}  # tick → list of MSEs
    with torch.no_grad():
        for batch_x, batch_y, batch_len in test_loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)

            pred = model(batch_x)
            for i in range(batch_x.size(0)):
                valid = min(batch_len[i].item() - 1, pred.size(1))
                for t in range(valid):
                    mse = ((pred[i, t] - batch_y[i, t]) ** 2).mean().item()
                    test_mse_per_step.setdefault(t + 1, []).append(mse)

    print(f"\n  {C_INFO}MSE by prediction horizon:{C_RESET}")
    for step in sorted(test_mse_per_step.keys()):
        mses = test_mse_per_step[step]
        mean_mse = np.mean(mses)
        bar = "█" * min(int(mean_mse * 500), 40)
        print(f"    {C_DIM}t+{step}:{C_RESET} {C_TEXT}MSE={mean_mse:.6f}{C_RESET} (n={len(mses):4d}) {C_GOLD}{bar}{C_RESET}")

    # Baseline: last-state predictor (predict s_{t+1} = s_t)
    print(f"\n  {C_INFO}Baseline comparison (last-state predictor):{C_RESET}")
    baseline_mse_per_step = {}
    with torch.no_grad():
        for batch_x, batch_y, batch_len in test_loader:
            for i in range(batch_x.size(0)):
                valid = min(batch_len[i].item() - 1, batch_x.size(1))
                for t in range(valid):
                    mse = ((batch_x[i, t] - batch_y[i, t]) ** 2).mean().item()
                    baseline_mse_per_step.setdefault(t + 1, []).append(mse)

    print(f"\n  {C_DIM}{'step':>6s}  {'Mamba MSE':>12s}  {'Baseline MSE':>12s}  {'Improvement':>12s}{C_RESET}")
    print(f"  {C_DIM}{'─' * 50}{C_RESET}")
    for step in sorted(test_mse_per_step.keys()):
        mamba_mse = np.mean(test_mse_per_step[step])
        baseline_mse = np.mean(baseline_mse_per_step.get(step, [0]))
        if baseline_mse > 0:
            improvement = (baseline_mse - mamba_mse) / baseline_mse * 100
        else:
            improvement = 0
        c = C_SUCCESS if improvement > 0 else C_DANGER
        print(
            f"  {C_TEXT}t+{step:3d}{C_RESET}  "
            f"{mamba_mse:12.6f}  "
            f"{baseline_mse:12.6f}  "
            f"{c}{improvement:+11.1f}%{C_RESET}"
        )

    print(f"\n  {C_SUCCESS}{C_BOLD}Best checkpoint:{C_RESET} {C_TEXT}{checkpoint_path / 'best_mamba.pt'}{C_RESET}")
    print(f"  {C_TEXT}Epoch: {ckpt['epoch']}  Val MSE: {ckpt['val_mse']:.6f}{C_RESET}")
    print(f"  {C_TEXT}Parameters: {C_BRIGHT}{n_params:,}{C_RESET}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SableMamba")
    parser.add_argument("--data", type=str, default="temporal_data.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        device=args.device,
    )

    train(data_path=args.data, config=config, checkpoint_dir=args.checkpoint_dir)
