#!/usr/bin/env -S python -u
"""
Project PARALLAX — Train GNN on Real Infrastructure Topologies
================================================================
Fine-tunes SableGNN on infrastructure graphs from Topology Zoo +
Microservice Dataset. Loads pre-built infra_graphs.pt.

Usage:
    python train_infra_gnn.py --epochs 100
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from sable_sim.core.states import N_STATES, STATE_NAMES
from cortex_gnn_model import SableGNN, compute_auc
from build_infra_dataset import N_NODE_TYPES, N_EDGE_TYPES, NODE_FEAT_DIM, EDGE_FEAT_DIM

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_DANGER = "\033[38;2;201;74;58m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"



def train(epochs=100, lr=0.001, device="cuda"):
    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — GNN on Real Infrastructure Topologies{C_RESET}")
    print(f"  {C_DIM}{'═' * 55}{C_RESET}\n")

    # Load pre-built dataset
    data_path = Path(__file__).parent / "data" / "infra_graphs.pt"
    print(f"  {C_INFO}Loading {data_path.name}...{C_RESET}", flush=True)
    ds = torch.load(data_path, weights_only=False)

    train_idx = ds["train_idx"]
    val_idx = ds["val_idx"]
    test_idx = ds["test_idx"]

    print(f"  {C_TEXT}train: {C_BRIGHT}{len(train_idx):,}{C_RESET}  val: {C_BRIGHT}{len(val_idx):,}{C_RESET}  test: {C_BRIGHT}{len(test_idx):,}{C_RESET}")
    print(f"  {C_TEXT}node_feat: {C_BRIGHT}{ds['node_feat_dim']}{C_RESET}  edge_feat: {C_BRIGHT}{ds['edge_feat_dim']}{C_RESET}")
    print(f"  {C_TEXT}max_nodes: {C_BRIGHT}{ds['max_nodes']}{C_RESET}  max_edges: {C_BRIGHT}{ds['max_edges']}{C_RESET}")

    # Model: SableGNN with infrastructure dimensions
    hidden_dim = 128
    model = SableGNN(
        in_dim=ds["node_feat_dim"],
        hidden_dim=hidden_dim,
        edge_dim=ds["edge_feat_dim"],
        num_layers=3,
        heads=4,
        n_relation_types=N_EDGE_TYPES,
        dropout=0.15,
    ).to(device)

    # State prediction head
    state_head = nn.Sequential(
        nn.Linear(hidden_dim, 128), nn.GELU(), nn.Dropout(0.15),
        nn.Linear(128, N_STATES),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in state_head.parameters())
    print(f"  {C_TEXT}params: {C_BRIGHT}{n_params:,}{C_RESET}  device: {C_BRIGHT}{device}{C_RESET}")

    # Class weights
    valid_gt = ds["GT"][ds["MASK"] > 0].long()
    state_counts = torch.bincount(valid_gt, minlength=N_STATES).float().clamp(min=1)
    state_weights = (valid_gt.size(0) / (N_STATES * state_counts)).clamp(max=10.0).to(device)

    # Edge type weights
    all_params = list(model.parameters()) + list(state_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_macro = -1.0
    patience = 0
    best_state = None

    print(f"\n  {C_DIM}{'ep':>4s} {'loss':>7s} {'v_acc':>7s} {'macro':>7s} {'hlthy':>7s} {'dgrad':>7s} {'faild':>7s} {'unrch':>7s} {'auc':>7s} {'t':>5s}{C_RESET}")
    print(f"  {C_DIM}{'─' * 78}{C_RESET}")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        state_head.train()
        t0 = time.time()

        perm = torch.randperm(len(train_idx))
        batch_size = 64
        epoch_loss = 0.0
        epoch_n = 0

        for b in range(0, len(perm), batch_size):
            batch_perm = perm[b:b + batch_size]
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            total_nodes = 0

            for pidx in batch_perm:
                idx = train_idx[pidx]
                nn_ = ds["N_NODES"][idx].item()
                ne_ = ds["N_EDGES"][idx].item()
                if ne_ < 2:
                    continue

                x = ds["X"][idx, :nn_].to(device)
                ei = ds["EI"][idx, :, :ne_].to(device)
                ea = ds["EA"][idx, :ne_].to(device)
                gt = ds["GT"][idx, :nn_].to(device)

                node_emb = model.encode(x, ei, ea)

                # State prediction loss
                state_logits = state_head(node_emb)
                loss_state = F.cross_entropy(state_logits, gt, weight=state_weights)

                # Link prediction loss (structural reasoning)
                neg_ei = negative_sampling(ei, num_nodes=nn_, num_neg_samples=min(ne_ * 2, 200))
                pos_logits = model.predict_link(node_emb, ei)
                neg_logits = model.predict_link(node_emb, neg_ei)
                link_logits = torch.cat([pos_logits, neg_logits])
                link_targets = torch.cat([
                    torch.ones(pos_logits.size(0), device=device),
                    torch.zeros(neg_logits.size(0), device=device),
                ])
                loss_link = F.binary_cross_entropy_with_logits(link_logits, link_targets)

                loss = loss_state + 0.3 * loss_link
                total_loss = total_loss + loss * nn_
                total_nodes += nn_

            if total_nodes > 0:
                avg_loss = total_loss / total_nodes
                optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
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
        state_head.eval()
        tp = torch.zeros(N_STATES); fp = torch.zeros(N_STATES); fn = torch.zeros(N_STATES)
        correct = total = 0
        auc_sum = auc_n = 0

        with torch.no_grad():
            for pidx in range(len(val_idx)):
                idx = val_idx[pidx]
                nn_ = ds["N_NODES"][idx].item()
                ne_ = ds["N_EDGES"][idx].item()
                if ne_ < 2:
                    continue

                x = ds["X"][idx, :nn_].to(device)
                ei = ds["EI"][idx, :, :ne_].to(device)
                ea = ds["EA"][idx, :ne_].to(device)
                gt = ds["GT"][idx, :nn_].to(device)

                node_emb = model.encode(x, ei, ea)
                preds = state_head(node_emb).argmax(dim=-1)

                correct += (preds == gt).sum().item()
                total += nn_

                for c in range(N_STATES):
                    ct = (gt == c); cp = (preds == c)
                    tp[c] += (ct & cp).sum().item()
                    fp[c] += (~ct & cp).sum().item()
                    fn[c] += (ct & ~cp).sum().item()

                # Link AUC (sample)
                if ne_ >= 4:
                    neg_ei = negative_sampling(ei, num_nodes=nn_, num_neg_samples=min(ne_, 100))
                    pos_l = model.predict_link(node_emb, ei)
                    neg_l = model.predict_link(node_emb, neg_ei)
                    all_l = torch.cat([pos_l, neg_l])
                    all_t = torch.cat([torch.ones(pos_l.size(0), device=device), torch.zeros(neg_l.size(0), device=device)])
                    auc_sum += compute_auc(torch.sigmoid(all_l), all_t) * ne_
                    auc_n += ne_

        val_acc = correct / max(total, 1)
        val_auc = auc_sum / max(auc_n, 1)
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
            f"{train_loss:7.4f} "
            f"{val_acc:7.4f} "
            f"{macro:7.4f} "
            + " ".join(f"{f:7.4f}" for f in f1s)
            + f" {val_auc:7.4f} "
            + f"{elapsed:4.1f}s {marker}",
            flush=True,
        )

        if improved:
            best_macro = macro
            best_f1s = f1s[:]
            best_auc = val_auc
            best_state = {
                "gnn": {k: v.clone() for k, v in model.state_dict().items()},
                "head": {k: v.clone() for k, v in state_head.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                print(f"\n  {C_DIM}Early stopping at epoch {epoch}{C_RESET}")
                break

    total_time = time.time() - t_start

    # Save
    if best_state:
        model.load_state_dict(best_state["gnn"])
        state_head.load_state_dict(best_state["head"])

    ckpt_path = Path(__file__).parent / "checkpoints" / "best_model.pt"
    torch.save({
        "model_state_dict": best_state["gnn"] if best_state else model.state_dict(),
        "state_head_state_dict": best_state["head"] if best_state else state_head.state_dict(),
        "config": {
            "in_dim": ds["node_feat_dim"],
            "hidden_dim": hidden_dim,
            "edge_dim": ds["edge_feat_dim"],
            "num_layers": 3,
            "heads": 4,
            "dropout": 0.15,
        },
        "macro_f1": best_macro,
        "link_auc": best_auc if best_state else 0,
        "per_class_f1": best_f1s if best_state else [],
    }, ckpt_path)

    print(f"\n  {C_GOLD}{C_BOLD}  Results{C_RESET}")
    print(f"  {C_DIM}{'─' * 50}{C_RESET}")
    print(f"  {C_TEXT}Macro F1:  {C_BRIGHT}{best_macro:.4f}{C_RESET}")
    print(f"  {C_TEXT}Link AUC:  {C_BRIGHT}{best_auc:.4f}{C_RESET}" if best_state else "")
    for i, name in enumerate(STATE_NAMES):
        bar = "█" * int(best_f1s[i] * 20) if best_state else ""
        print(f"    {C_DIM}{name:12s}{C_RESET} F1={C_TEXT}{best_f1s[i]:.4f}{C_RESET} {C_GOLD}{bar}{C_RESET}" if best_state else "")
    print(f"  {C_TEXT}Time: {C_BRIGHT}{total_time:.0f}s{C_RESET}")
    print(f"  {C_TEXT}Saved: {C_BRIGHT}{ckpt_path}{C_RESET}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args.epochs, args.lr, args.device)
