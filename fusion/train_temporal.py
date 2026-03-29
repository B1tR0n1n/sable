#!/usr/bin/env -S python -u
"""
Project PARALLAX — Train Temporal Chain
=========================================
Freezes the base SharpRoutedFusion, trains only ContextMixer + RevisionGate
on temporal sequences via truncated BPTT (4-step windows).

Usage:
    python train_temporal.py --epochs 80 --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar3"))

from sable_sim.core.states import N_STATES, STATE_NAMES
from temporal_chain import TemporalChainFusion, TemporalState
from staged_fusion_v3 import SharpRoutedFusion
from shared_latent_space import GNN_DIM, POMDP_DIM
from generate_temporal_data import NODE_FEAT_DIM

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_DANGER = "\033[38;2;201;74;58m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"



def train(epochs: int = 80, window: int = 4, lr: float = 5e-4, device: str = "cuda"):
    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Temporal Chain Training{C_RESET}")
    print(f"  {C_DIM}{'═' * 50}{C_RESET}\n")

    # Load temporal sequences
    data_path = Path(__file__).parent / "temporal_sequences.pt"
    print(f"  {C_INFO}Loading {data_path.name}...{C_RESET}", flush=True)
    ds = torch.load(data_path, weights_only=False)

    n_seq = ds["gnn"].size(0)
    max_ticks = ds["max_ticks"]
    max_nodes = ds["max_nodes"]
    seq_lens = ds["seq_len"]

    # Filter to sequences with enough ticks for windowed training
    valid_mask = seq_lens >= window
    valid_idx = torch.where(valid_mask)[0]
    print(f"  {C_TEXT}Sequences: {C_BRIGHT}{n_seq}{C_RESET} total, {C_BRIGHT}{len(valid_idx)}{C_RESET} with >= {window} ticks")

    # Split valid sequences
    perm = torch.randperm(len(valid_idx))
    n_train = int(len(valid_idx) * 0.8)
    train_idx = valid_idx[perm[:n_train]]
    val_idx = valid_idx[perm[n_train:]]
    print(f"  {C_TEXT}Train: {C_BRIGHT}{len(train_idx)}{C_RESET}  Val: {C_BRIGHT}{len(val_idx)}{C_RESET}")

    # Load base fusion (frozen)
    base_ckpt_path = Path(__file__).parent / "checkpoints" / "staged_fusion_v3.pt"
    base_fusion = SharpRoutedFusion(mamba_dim=NODE_FEAT_DIM)
    if base_ckpt_path.exists():
        ckpt = torch.load(base_ckpt_path, weights_only=False)
        base_fusion.load_state_dict(ckpt["model_state_dict"])
        print(f"  {C_TEXT}Base fusion loaded from checkpoint{C_RESET}", flush=True)

    # Freeze base fusion
    for p in base_fusion.parameters():
        p.requires_grad = False

    # Build temporal chain
    model = TemporalChainFusion(base_fusion).to(device)
    n_temporal = model.n_temporal_params()
    n_total = sum(p.numel() for p in model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  {C_TEXT}Temporal params: {C_BRIGHT}{n_temporal:,}{C_RESET} (of {n_total:,} total)")
    print(f"  {C_TEXT}Device: {C_BRIGHT}{device}{C_RESET}")

    # Class weights
    valid_states = ds["states"][ds["mask"] > 0].long()
    state_counts = torch.bincount(valid_states, minlength=N_STATES).float().clamp(min=1)
    state_weights = (valid_states.size(0) / (N_STATES * state_counts)).clamp(max=10.0).to(device)

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_macro = -1.0
    best_state = None
    best_f1s = [0.0] * N_STATES
    patience = 0

    print(f"\n  {C_DIM}{'ep':>4s} {'loss':>8s} {'v_macro':>8s} {'hlthy':>7s} {'dgrad':>7s} {'faild':>7s} {'unrch':>7s} {'t':>5s}{C_RESET}")
    print(f"  {C_DIM}{'─' * 65}{C_RESET}")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        epoch_n = 0

        perm_train = torch.randperm(len(train_idx))

        for bi in range(0, len(perm_train), 16):  # batch of 16 sequences
            batch_indices = train_idx[perm_train[bi:bi + 16]]

            losses = []
            total_nodes = 0

            for seq_idx in batch_indices:
                seq_len = seq_lens[seq_idx].item()
                n_nodes = ds["n_nodes"][seq_idx].item()

                max_start = seq_len - window
                start = 0 if max_start <= 0 else torch.randint(0, max_start + 1, (1,)).item()

                ts = TemporalState.cold_start(n_nodes, device=device)

                tick_losses = []
                for t in range(start, min(start + window, seq_len)):
                    gnn_t = ds["gnn"][seq_idx, t, :n_nodes].unsqueeze(0).to(device)
                    pomdp_t = ds["pomdp"][seq_idx, t, :n_nodes].unsqueeze(0).to(device).clone()
                    mamba_t = ds["mamba"][seq_idx, t, :n_nodes].unsqueeze(0).to(device)
                    gt_t = ds["states"][seq_idx, t, :n_nodes].unsqueeze(0).to(device)
                    mask_t = ds["mask"][seq_idx, t, :n_nodes].unsqueeze(0).to(device)

                    # Noise hardening: 35% of steps, scramble POMDP for degraded nodes
                    # Also corrupt 10% of random nodes
                    if torch.rand(1).item() < 0.35:
                        degraded_mask = (gt_t[0] == 1)  # degraded class
                        if degraded_mask.any():
                            noise = torch.rand(degraded_mask.sum().item(), 4, device=device)
                            noise = noise / noise.sum(dim=-1, keepdim=True)
                            pomdp_t[0, degraded_mask, :4] = noise
                        random_mask = torch.rand(n_nodes, device=device) < 0.10
                        if random_mask.any():
                            noise2 = torch.rand(random_mask.sum().item(), 4, device=device)
                            noise2 = noise2 / noise2.sum(dim=-1, keepdim=True)
                            pomdp_t[0, random_mask, :4] = noise2

                    out = model(gnn_t, pomdp_t, mamba_t, temporal_state=ts)

                    l_state = F.cross_entropy(
                        out["revised_logits"].reshape(-1, N_STATES),
                        gt_t.reshape(-1),
                        weight=state_weights, reduction="none"
                    )
                    l_state = (l_state * mask_t.reshape(-1)).sum() / mask_t.sum().clamp(min=1)
                    tick_losses.append(l_state)

                    if t > start:
                        prev_gt = ds["states"][seq_idx, t - 1, :n_nodes].to(device)
                        direction = (gt_t[0] - prev_gt).sign().long() + 1
                        direction = direction.clamp(0, 2)
                        l_trans = F.cross_entropy(
                            out["transition"].reshape(-1, 3),
                            direction.reshape(-1), reduction="none"
                        )
                        l_trans = (l_trans * mask_t.reshape(-1)).sum() / mask_t.sum().clamp(min=1)
                        tick_losses.append(0.3 * l_trans)

                    ts.update(
                        out["revised_logits"].detach(),
                        out["confidence"].detach(),
                        out.get("z_fused").detach() if out.get("z_fused") is not None else None,
                    )

                if tick_losses:
                    seq_loss = torch.stack(tick_losses).sum()
                    losses.append(seq_loss * n_nodes)
                    total_nodes += n_nodes * min(window, seq_len - start)

            if losses and total_nodes > 0:
                avg_loss = torch.stack(losses).sum() / total_nodes
                optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                epoch_loss += avg_loss.item() * total_nodes
                epoch_n += total_nodes

        scheduler.step()
        train_loss = epoch_loss / max(epoch_n, 1)
        elapsed = time.time() - t0

        # Validate
        if not (epoch <= 5 or epoch % 5 == 0 or epoch == epochs):
            continue

        model.eval()
        tp = torch.zeros(N_STATES); fp = torch.zeros(N_STATES); fn = torch.zeros(N_STATES)

        with torch.no_grad():
            for seq_idx in val_idx:
                seq_len = seq_lens[seq_idx].item()
                n_nodes = ds["n_nodes"][seq_idx].item()
                ts = TemporalState.cold_start(n_nodes, device=device)

                for t in range(seq_len):
                    gnn_t = ds["gnn"][seq_idx, t, :n_nodes].unsqueeze(0).to(device)
                    pomdp_t = ds["pomdp"][seq_idx, t, :n_nodes].unsqueeze(0).to(device)
                    mamba_t = ds["mamba"][seq_idx, t, :n_nodes].unsqueeze(0).to(device)
                    gt_t = ds["states"][seq_idx, t, :n_nodes].to(device)
                    mask_t = ds["mask"][seq_idx, t, :n_nodes].to(device)

                    out = model(gnn_t, pomdp_t, mamba_t, temporal_state=ts)
                    preds = out["revised_logits"][0].argmax(dim=-1)
                    valid = mask_t > 0

                    for c in range(N_STATES):
                        ct = (gt_t == c) & valid
                        cp = (preds == c) & valid
                        tp[c] += (ct & cp).sum().item()
                        fp[c] += (~ct & cp).sum().item()
                        fn[c] += (ct & ~cp).sum().item()

                    ts.update(out["revised_logits"].detach(), out["confidence"].detach(),
                              out.get("z_fused", ts.prev_z).detach() if out.get("z_fused") is not None else None)

        f1s = []
        for c in range(N_STATES):
            p = tp[c] / max(tp[c] + fp[c], 1)
            r = tp[c] / max(tp[c] + fn[c], 1)
            f1s.append(2 * p * r / max(p + r, 1e-8))
        macro = sum(f1s) / N_STATES

        improved = macro > best_macro + 0.001
        marker = f"{C_SUCCESS}*{C_RESET}" if improved else " "

        print(
            f"  {C_TEXT}{epoch:4d}{C_RESET} "
            f"{train_loss:8.4f} "
            f"{macro:8.4f} "
            + " ".join(f"{f:7.4f}" for f in f1s)
            + f" {elapsed:4.1f}s {marker}",
            flush=True,
        )

        if improved:
            best_macro = macro
            best_f1s = f1s[:]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                print(f"\n  {C_DIM}Early stopping at epoch {epoch}{C_RESET}")
                break

    total_time = time.time() - t_start

    if best_state:
        model.load_state_dict(best_state)

    # Save
    ckpt_path = Path(__file__).parent / "checkpoints" / "temporal_chain.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "temporal_params": {k: v for k, v in model.state_dict().items()
                           if "context_mixer" in k or "revision_gate" in k},
        "macro_f1": best_macro,
        "per_class_f1": best_f1s if best_state else [],
    }, ckpt_path)

    print(f"\n  {C_GOLD}{C_BOLD}  Results{C_RESET}")
    print(f"  {C_DIM}{'─' * 50}{C_RESET}")
    print(f"  {C_TEXT}Macro F1: {C_BRIGHT}{best_macro:.4f}{C_RESET}")
    if best_state:
        for i, name in enumerate(STATE_NAMES):
            bar = "█" * int(best_f1s[i] * 20)
            print(f"    {C_DIM}{name:12s}{C_RESET} F1={C_TEXT}{best_f1s[i]:.4f}{C_RESET} {C_GOLD}{bar}{C_RESET}")
    print(f"  {C_TEXT}Time: {C_BRIGHT}{total_time:.0f}s{C_RESET}")
    print(f"  {C_TEXT}Saved: {C_BRIGHT}{ckpt_path}{C_RESET}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(args.epochs, args.window, args.lr, args.device)
