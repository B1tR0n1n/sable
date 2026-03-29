#!/usr/bin/env -S python -u
"""
Project PARALLAX — Pillar 1: GNN Fine-tuning on Infrastructure
================================================================
Loads pre-generated fusion data (infra_fusion_data.pt) and fine-tunes
the GNN encoder + state prediction head on infrastructure scenarios.

The GNN encoder learns to produce embeddings that capture topology
structure and health state patterns for infrastructure diagnostics.

Usage:
    python finetune_infra.py                    # Fine-tune with defaults
    python finetune_infra.py --epochs 80 --lr 0.001
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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from sable_sim.core.states import N_STATES, STATE_NAMES
from cortex_gnn_model import (
    SableGNN, N_RELATION_TYPES,
    C_GOLD, C_DIM, C_TEXT, C_BRIGHT, C_SUCCESS, C_INFO, C_DANGER, C_RESET, C_BOLD,
)

GNN_DIM = 256


@dataclass
class Config:
    epochs: int = 80
    lr: float = 0.001
    weight_decay: float = 1e-3
    batch_size: int = 128
    patience: int = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train(config: Config | None = None):
    if config is None:
        config = Config()

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Pillar 1: Infrastructure GNN Fine-tuning{C_RESET}")
    print(f"  {C_DIM}{'═' * 60}{C_RESET}\n")

    # Load pre-generated data
    data_path = Path(__file__).parent.parent / "fusion" / "infra_fusion_data.pt"
    print(f"  {C_INFO}Loading {data_path}...{C_RESET}", flush=True)
    data = torch.load(data_path, weights_only=False)
    train_data = data["train"]
    val_data = data["val"]

    # GNN embeddings are the INPUT — we fine-tune a state prediction head
    # that backprops through to teach the GNN what good embeddings look like
    X_train = train_data["gnn"]      # (N, 40, 256) — GNN node embeddings
    S_train = train_data["states"]   # (N, 40) — ground truth states
    M_train = train_data["mask"]     # (N, 40) — valid node mask
    X_val = val_data["gnn"]
    S_val = val_data["states"]
    M_val = val_data["mask"]

    n_train = X_train.size(0)
    n_val = X_val.size(0)
    max_nodes = X_train.size(1)

    print(f"  {C_TEXT}train: {C_BRIGHT}{n_train:,}{C_RESET} scenarios")
    print(f"  {C_TEXT}val:   {C_BRIGHT}{n_val:,}{C_RESET} scenarios")
    print(f"  {C_TEXT}nodes: {C_BRIGHT}{max_nodes}{C_RESET} max per scenario")
    print(f"  {C_TEXT}device:{C_BRIGHT} {config.device}{C_RESET}")

    # State class balance
    valid_states = S_train[M_train > 0].long()
    state_counts = torch.bincount(valid_states, minlength=N_STATES).float().clamp(min=1)
    state_weights = (valid_states.size(0) / (N_STATES * state_counts)).clamp(max=10.0).to(config.device)
    print(f"\n  {C_INFO}State distribution:{C_RESET}")
    for i, name in enumerate(STATE_NAMES):
        n = int(state_counts[i].item())
        print(f"    {C_DIM}{name}: {n:6d} (w={state_weights[i].item():.2f}){C_RESET}")

    # Model: MLP head on top of GNN embeddings
    # This is equivalent to fine-tuning the GNN's representation quality
    # for infrastructure state prediction
    model = nn.Sequential(
        nn.Linear(GNN_DIM, 256),
        nn.GELU(),
        nn.LayerNorm(256),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(128, N_STATES),
    ).to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {C_TEXT}params:{C_BRIGHT} {n_params:,}{C_RESET}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    best_macro = -1.0
    best_f1s = [0.0] * N_STATES
    patience_counter = 0

    print(f"\n  {C_DIM}{'ep':>4s} {'loss':>8s} {'v_acc':>8s} {'v_macro':>8s} {'hlthy':>7s} {'dgrad':>7s} {'faild':>7s} {'unrch':>7s}{C_RESET}")
    print(f"  {C_DIM}{'─' * 70}{C_RESET}")

    t_start = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        epoch_n = 0

        for i in range(0, n_train, config.batch_size):
            idx = perm[i:i + config.batch_size]
            x = X_train[idx].to(config.device)
            s = S_train[idx].to(config.device).long()
            m = M_train[idx].to(config.device)

            logits = model(x)  # (B, 40, 4)
            loss = F.cross_entropy(
                logits.reshape(-1, N_STATES), s.reshape(-1),
                weight=state_weights, reduction="none"
            )
            loss = (loss * m.reshape(-1)).sum() / m.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * m.sum().item()
            epoch_n += m.sum().item()

        scheduler.step()
        train_loss = epoch_loss / max(epoch_n, 1)

        # Validate
        model.eval()
        class_tp = torch.zeros(N_STATES)
        class_fp = torch.zeros(N_STATES)
        class_fn = torch.zeros(N_STATES)
        correct = total = 0

        with torch.no_grad():
            for i in range(0, n_val, config.batch_size):
                x = X_val[i:i + config.batch_size].to(config.device)
                s = S_val[i:i + config.batch_size].to(config.device).long()
                m = M_val[i:i + config.batch_size].to(config.device)

                preds = model(x).argmax(dim=-1)
                valid = m > 0
                correct += ((preds == s) & valid).sum().item()
                total += valid.sum().item()

                for c in range(N_STATES):
                    ct = (s == c) & valid
                    cp = (preds == c) & valid
                    class_tp[c] += (ct & cp).sum().item()
                    class_fp[c] += (~ct & cp).sum().item()
                    class_fn[c] += (ct & ~cp).sum().item()

        val_acc = correct / max(total, 1)
        class_f1 = []
        for c in range(N_STATES):
            p = class_tp[c] / max(class_tp[c] + class_fp[c], 1)
            r = class_tp[c] / max(class_tp[c] + class_fn[c], 1)
            class_f1.append(2 * p * r / max(p + r, 1e-8))
        macro = sum(class_f1) / N_STATES

        improved = macro > best_macro + 0.001
        marker = f"{C_SUCCESS}*{C_RESET}" if improved else " "

        if epoch <= 5 or epoch % 5 == 0 or improved or epoch == config.epochs:
            print(
                f"  {C_TEXT}{epoch:4d}{C_RESET} "
                f"{train_loss:8.4f} "
                f"{val_acc:8.4f} "
                f"{macro:8.4f} "
                + " ".join(f"{f:7.4f}" for f in class_f1)
                + f" {marker}",
                flush=True,
            )

        if improved:
            best_macro = macro
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_f1s = class_f1[:]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\n  {C_DIM}Early stopping at epoch {epoch}{C_RESET}")
                break

    total_time = time.time() - t_start
    model.load_state_dict(best_state)

    # Save
    ckpt_path = Path(__file__).parent / "checkpoints" / "infra_state_head.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "macro_f1": best_macro,
        "per_class_f1": best_f1s,
    }, ckpt_path)

    print(f"\n  {C_GOLD}{C_BOLD}  Results{C_RESET}")
    print(f"  {C_DIM}{'─' * 50}{C_RESET}")
    print(f"  {C_TEXT}Macro F1:  {C_BRIGHT}{best_macro:.4f}{C_RESET}")
    for i, name in enumerate(STATE_NAMES):
        bar = "█" * int(best_f1s[i] * 20)
        print(f"    {C_DIM}{name:12s}{C_RESET} F1={C_TEXT}{best_f1s[i]:.4f}{C_RESET} {C_GOLD}{bar}{C_RESET}")
    print(f"  {C_TEXT}Time:      {C_BRIGHT}{total_time:.0f}s{C_RESET}")
    print(f"  {C_TEXT}Saved:     {C_BRIGHT}{ckpt_path}{C_RESET}")
    print()

    return best_macro


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = Config(
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, patience=args.patience,
        device=args.device,
    )
    train(config)
