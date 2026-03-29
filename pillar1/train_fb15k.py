#!/usr/bin/env python3
"""
Project PARALLAX — Pillar 1: GNN Pre-training on FB15k-237
=============================================================
Real knowledge graph benchmark for multi-relational link prediction.
14,541 entities, 272,115 triples, 237 relation types.

Teaches SableGNN structural reasoning on real typed edges — the same
capability needed for infrastructure dependency graph reasoning.

Architecture: SableGNN backbone with learnable node/relation embeddings.
Same GATv2Conv encoder, same multi-task heads, real data.

Usage:
    python train_fb15k.py                            # Train with defaults
    python train_fb15k.py --epochs 300 --lr 0.0003   # Custom hyperparams
    python train_fb15k.py --eval checkpoints_fb15k/best_fb15k.pt

Requires: ml-env with torch, torch_geometric
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
from torch_geometric.datasets import FB15k_237
from torch_geometric.utils import negative_sampling

sys.path.insert(0, str(Path(__file__).parent))
from cortex_gnn_model import (
    SableGNN, compute_auc, focal_bce_loss,
    C_GOLD, C_DIM, C_TEXT, C_BRIGHT, C_SUCCESS, C_INFO, C_DANGER, C_RESET, C_BOLD,
)


# ── KG Wrapper ────────────────────────────────────────────────────────────


class SableGNN_KG(nn.Module):
    """SableGNN with learnable embeddings for knowledge graph pre-training.

    Wraps the domain-agnostic SableGNN backbone with:
      - Learnable node embeddings (replacing pre-computed features)
      - Learnable relation embeddings (replacing one-hot edge features)

    The backbone learns structural reasoning patterns that transfer to
    infrastructure topology — multi-relational link prediction, typed edge
    classification, and structural anomaly detection.
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        embed_dim: int = 256,
        edge_embed_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations

        # Learnable feature embeddings
        self.node_emb = nn.Embedding(num_nodes, embed_dim)
        self.rel_emb = nn.Embedding(num_relations, edge_embed_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # SableGNN backbone — same architecture, different input dims
        self.gnn = SableGNN(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_embed_dim,
            num_layers=num_layers,
            heads=heads,
            n_relation_types=num_relations,
            dropout=dropout,
        )

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """Encode all nodes through the GNN backbone."""
        x = self.node_emb.weight  # [N, embed_dim]
        edge_attr = self.rel_emb(edge_type)  # [E, edge_embed_dim]
        return self.gnn.encode(x, edge_index, edge_attr)

    def predict_link_type(self, node_emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gnn.predict_link_type(node_emb, edge_index)

    def predict_link(self, node_emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gnn.predict_link(node_emb, edge_index)


# ── Configuration ─────────────────────────────────────────────────────────


@dataclass
class KGConfig:
    epochs: int = 200
    lr: float = 0.0005
    weight_decay: float = 1e-4
    embed_dim: int = 256
    edge_embed_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.1
    neg_ratio: int = 5
    link_type_weight: float = 1.0
    link_pred_weight: float = 0.5
    patience: int = 30
    eval_every: int = 10
    mrr_every: int = 50
    mrr_sample: int = 500  # Corruptions per triple for MRR
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ── MRR / Hits@K Evaluation ──────────────────────────────────────────────


@torch.no_grad()
def compute_mrr_hits(
    model: SableGNN_KG,
    node_emb: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    n_eval: int = 2000,
    n_corrupt: int = 500,
) -> dict:
    """Compute MRR and Hits@K via sampled tail corruption."""
    device = node_emb.device
    n_edges = edge_index.size(1)
    eval_idx = torch.randperm(n_edges)[:min(n_eval, n_edges)]
    eval_ei = edge_index[:, eval_idx]

    ranks = []
    batch_size = 256

    for i in range(0, eval_ei.size(1), batch_size):
        batch_ei = eval_ei[:, i:i + batch_size]
        B = batch_ei.size(1)
        heads = batch_ei[0]
        tails = batch_ei[1]

        # Score true triples
        true_scores = model.gnn.predict_link(node_emb, batch_ei)  # [B]

        # Corrupt tails
        corrupt_tails = torch.randint(0, num_nodes, (B, n_corrupt), device=device)

        for b in range(B):
            corrupt_ei = torch.stack([
                heads[b].expand(n_corrupt),
                corrupt_tails[b],
            ])
            corrupt_scores = model.gnn.predict_link(node_emb, corrupt_ei)
            rank = (corrupt_scores > true_scores[b]).sum().item() + 1
            ranks.append(rank)

    ranks = np.array(ranks, dtype=np.float64)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits@1": float(np.mean(ranks <= 1)),
        "hits@3": float(np.mean(ranks <= 3)),
        "hits@10": float(np.mean(ranks <= 10)),
    }


# ── Training ──────────────────────────────────────────────────────────────


def train(config: KGConfig | None = None, data_root: str | None = None):
    if config is None:
        config = KGConfig()

    if data_root is None:
        data_root = str(Path(__file__).parent / "data")

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Pillar 1: FB15k-237 Pre-training{C_RESET}")
    print(f"  {C_DIM}{'═' * 55}{C_RESET}\n")

    # ── Load data ──
    print(f"  {C_INFO}Loading FB15k-237...{C_RESET}")
    train_data = FB15k_237(root=data_root, split="train")[0]
    val_data = FB15k_237(root=data_root, split="val")[0]
    test_data = FB15k_237(root=data_root, split="test")[0]

    num_nodes = train_data.num_nodes
    num_relations = int(train_data.edge_type.max()) + 1

    print(f"  {C_TEXT}nodes:     {C_BRIGHT}{num_nodes:,}{C_RESET}")
    print(f"  {C_TEXT}relations: {C_BRIGHT}{num_relations}{C_RESET}")
    print(f"  {C_TEXT}train:     {C_BRIGHT}{train_data.edge_index.size(1):,} triples{C_RESET}")
    print(f"  {C_TEXT}val:       {C_BRIGHT}{val_data.edge_index.size(1):,} triples{C_RESET}")
    print(f"  {C_TEXT}test:      {C_BRIGHT}{test_data.edge_index.size(1):,} triples{C_RESET}")
    print(f"  {C_TEXT}device:    {C_BRIGHT}{config.device}{C_RESET}")

    # ── Build model ──
    model = SableGNN_KG(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embed_dim=config.embed_dim,
        edge_embed_dim=config.edge_embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        heads=config.heads,
        dropout=config.dropout,
    ).to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    n_backbone = sum(p.numel() for p in model.gnn.parameters())
    print(f"  {C_TEXT}total params:    {C_BRIGHT}{n_params:,}{C_RESET}")
    print(f"  {C_TEXT}backbone params: {C_BRIGHT}{n_backbone:,}{C_RESET}")
    print(f"  {C_TEXT}embedding params:{C_BRIGHT} {n_params - n_backbone:,}{C_RESET}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    # Move data to device
    train_ei = train_data.edge_index.to(config.device)
    train_et = train_data.edge_type.to(config.device)
    val_ei = val_data.edge_index.to(config.device)
    val_et = val_data.edge_type.to(config.device)
    test_ei = test_data.edge_index.to(config.device)
    test_et = test_data.edge_type.to(config.device)

    # Class weights for link type prediction (inverse frequency, capped)
    class_counts = torch.bincount(train_et.cpu(), minlength=num_relations).float().clamp(min=1)
    class_weights = (train_et.size(0) / (num_relations * class_counts)).to(config.device)
    class_weights = class_weights.clamp(max=20.0)

    # Relation type distribution
    print(f"\n  {C_INFO}Relation type distribution (top 10 / {num_relations}):{C_RESET}")
    counts_sorted, idx_sorted = class_counts.sort(descending=True)
    for i in range(min(10, num_relations)):
        bar = "█" * min(int(counts_sorted[i].item() / 200), 40)
        print(f"    {C_DIM}rel {idx_sorted[i].item():3d}:{C_RESET} {C_TEXT}{int(counts_sorted[i].item()):5d}{C_RESET} {C_GOLD}{bar}{C_RESET}")

    # ── Training loop ──
    checkpoint_path = Path(__file__).parent / "checkpoints_fb15k"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_val_metric = -1.0
    patience_counter = 0
    best_metrics = {}

    print(f"\n  {C_DIM}{'ep':>4s} {'loss':>8s} {'type_l':>8s} {'link_l':>8s} {'v_type':>8s} {'v_auc':>8s} {'v_mrr':>8s} {'h@10':>8s} {'time':>6s}{C_RESET}")
    print(f"  {C_DIM}{'─' * 80}{C_RESET}")

    t_start = time.time()

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        model.train()

        # Encode all nodes using training graph
        node_emb = model.encode(train_ei, train_et)

        # Task 1: Link type prediction (237-class)
        type_logits = model.predict_link_type(node_emb, train_ei)
        loss_type = F.cross_entropy(type_logits, train_et.long(), weight=class_weights)

        # Task 2: Link prediction (positive + negative)
        neg_ei = negative_sampling(
            train_ei, num_nodes=num_nodes,
            num_neg_samples=train_ei.size(1) * config.neg_ratio,
        )
        pos_logits = model.predict_link(node_emb, train_ei)
        neg_logits = model.predict_link(node_emb, neg_ei)
        link_logits = torch.cat([pos_logits, neg_logits])
        link_targets = torch.cat([
            torch.ones(pos_logits.size(0), device=config.device),
            torch.zeros(neg_logits.size(0), device=config.device),
        ])
        loss_link = F.binary_cross_entropy_with_logits(link_logits, link_targets)

        # Combined loss
        loss = config.link_type_weight * loss_type + config.link_pred_weight * loss_link

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t0

        # ── Evaluate ──
        should_eval = epoch <= 5 or epoch % config.eval_every == 0 or epoch == config.epochs
        if not should_eval:
            continue

        model.eval()
        with torch.no_grad():
            node_emb = model.encode(train_ei, train_et)

            # Val link type accuracy
            val_type_logits = model.predict_link_type(node_emb, val_ei)
            val_type_acc = (val_type_logits.argmax(-1) == val_et).float().mean().item()

            # Val link prediction AUC
            val_neg_ei = negative_sampling(
                val_ei, num_nodes=num_nodes,
                num_neg_samples=val_ei.size(1) * 5,
            )
            val_pos_logits = model.predict_link(node_emb, val_ei)
            val_neg_logits = model.predict_link(node_emb, val_neg_ei)
            val_all_logits = torch.cat([val_pos_logits, val_neg_logits])
            val_all_targets = torch.cat([
                torch.ones(val_pos_logits.size(0), device=config.device),
                torch.zeros(val_neg_logits.size(0), device=config.device),
            ])
            val_link_auc = compute_auc(torch.sigmoid(val_all_logits), val_all_targets)

        # MRR (expensive — less frequently)
        mrr_metrics = {"mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0}
        if epoch % config.mrr_every == 0 or epoch == config.epochs:
            mrr_metrics = compute_mrr_hits(
                model, node_emb, val_ei, num_nodes,
                n_eval=2000, n_corrupt=config.mrr_sample,
            )

        val_metric = val_link_auc
        improved = val_metric > best_val_metric + 0.001
        marker = f"{C_SUCCESS}*{C_RESET}" if improved else " "

        print(
            f"  {C_TEXT}{epoch:4d}{C_RESET} "
            f"{loss.item():8.4f} "
            f"{loss_type.item():8.4f} "
            f"{loss_link.item():8.4f} "
            f"{val_type_acc:8.4f} "
            f"{val_link_auc:8.4f} "
            f"{mrr_metrics['mrr']:8.4f} "
            f"{mrr_metrics['hits@10']:8.4f} "
            f"{elapsed:5.1f}s {marker}"
        )

        if improved:
            best_val_metric = val_metric
            best_metrics = {
                "val_link_auc": val_link_auc,
                "val_type_acc": val_type_acc,
                **mrr_metrics,
            }
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "gnn_state_dict": model.gnn.state_dict(),
                "val_metrics": best_metrics,
                "config": {
                    "num_nodes": num_nodes,
                    "num_relations": num_relations,
                    "embed_dim": config.embed_dim,
                    "edge_embed_dim": config.edge_embed_dim,
                    "hidden_dim": config.hidden_dim,
                    "num_layers": config.num_layers,
                    "heads": config.heads,
                    "dropout": config.dropout,
                },
                "n_params": n_params,
                "n_backbone": n_backbone,
            }, checkpoint_path / "best_fb15k.pt")
        else:
            patience_counter += config.eval_every
            if patience_counter >= config.patience * config.eval_every:
                print(f"\n  {C_DIM}Early stopping at epoch {epoch}{C_RESET}")
                break

    total_time = time.time() - t_start

    # ── Test evaluation ──
    print(f"\n{C_GOLD}{C_BOLD}  Test Set Evaluation{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")

    ckpt = torch.load(checkpoint_path / "best_fb15k.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        # Use train + val edges for message passing at test time (standard transductive)
        all_ei = torch.cat([train_ei, val_ei], dim=1)
        all_et = torch.cat([train_et, val_et])
        node_emb = model.encode(all_ei, all_et)

        # Test link type accuracy
        test_type_logits = model.predict_link_type(node_emb, test_ei)
        test_type_acc = (test_type_logits.argmax(-1) == test_et).float().mean().item()

        # Per-class metrics for link type
        pred_classes = test_type_logits.argmax(-1).cpu()
        true_classes = test_et.cpu()

        # Test link prediction AUC
        test_neg_ei = negative_sampling(
            test_ei, num_nodes=num_nodes,
            num_neg_samples=test_ei.size(1) * 5,
        )
        test_pos_logits = model.predict_link(node_emb, test_ei)
        test_neg_logits = model.predict_link(node_emb, test_neg_ei)
        test_all_logits = torch.cat([test_pos_logits, test_neg_logits])
        test_all_targets = torch.cat([
            torch.ones(test_pos_logits.size(0), device=config.device),
            torch.zeros(test_neg_logits.size(0), device=config.device),
        ])
        test_link_auc = compute_auc(torch.sigmoid(test_all_logits), test_all_targets)

        # Link prediction accuracy
        test_link_preds = (test_all_logits > 0).float()
        test_link_acc = (test_link_preds == test_all_targets).float().mean().item()

    # Test MRR
    test_mrr = compute_mrr_hits(
        model, node_emb, test_ei, num_nodes,
        n_eval=3000, n_corrupt=config.mrr_sample,
    )

    print(f"\n  {C_INFO}Link Prediction:{C_RESET}")
    print(f"    {C_TEXT}AUC-ROC:  {C_BRIGHT}{test_link_auc:.4f}{C_RESET}")
    print(f"    {C_TEXT}Accuracy: {C_BRIGHT}{test_link_acc:.4f}{C_RESET}")

    print(f"\n  {C_INFO}Knowledge Graph Ranking:{C_RESET}")
    print(f"    {C_TEXT}MRR:      {C_BRIGHT}{test_mrr['mrr']:.4f}{C_RESET}")
    print(f"    {C_TEXT}Hits@1:   {C_BRIGHT}{test_mrr['hits@1']:.4f}{C_RESET}")
    print(f"    {C_TEXT}Hits@3:   {C_BRIGHT}{test_mrr['hits@3']:.4f}{C_RESET}")
    print(f"    {C_TEXT}Hits@10:  {C_BRIGHT}{test_mrr['hits@10']:.4f}{C_RESET}")

    print(f"\n  {C_INFO}Link Type Classification ({num_relations}-class):{C_RESET}")
    print(f"    {C_TEXT}Accuracy: {C_BRIGHT}{test_type_acc:.4f}{C_RESET}")

    # Top-10 and bottom-10 relation types by accuracy
    per_rel_correct = torch.zeros(num_relations)
    per_rel_total = torch.zeros(num_relations)
    for r in range(num_relations):
        mask = true_classes == r
        if mask.sum() > 0:
            per_rel_total[r] = mask.sum().float()
            per_rel_correct[r] = ((pred_classes == r) & mask).sum().float()

    per_rel_acc = per_rel_correct / per_rel_total.clamp(min=1)
    valid_rels = per_rel_total > 0
    valid_accs = per_rel_acc[valid_rels]
    valid_indices = torch.arange(num_relations)[valid_rels]

    sorted_accs, sort_idx = valid_accs.sort(descending=True)
    print(f"\n    {C_DIM}Top 5 relation types:{C_RESET}")
    for i in range(min(5, len(sorted_accs))):
        rel_id = valid_indices[sort_idx[i]].item()
        acc = sorted_accs[i].item()
        n = int(per_rel_total[rel_id].item())
        bar = "█" * int(acc * 20)
        print(f"      {C_DIM}rel {rel_id:3d}{C_RESET} acc={C_TEXT}{acc:.3f}{C_RESET} (n={n:4d}) {C_GOLD}{bar}{C_RESET}")

    print(f"\n    {C_DIM}Bottom 5 relation types:{C_RESET}")
    for i in range(max(0, len(sorted_accs) - 5), len(sorted_accs)):
        rel_id = valid_indices[sort_idx[i]].item()
        acc = sorted_accs[i].item()
        n = int(per_rel_total[rel_id].item())
        bar = "█" * int(acc * 20)
        print(f"      {C_DIM}rel {rel_id:3d}{C_RESET} acc={C_TEXT}{acc:.3f}{C_RESET} (n={n:4d}) {C_GOLD}{bar}{C_RESET}")

    # Summary
    print(f"\n  {C_GOLD}{C_BOLD}  Summary{C_RESET}")
    print(f"  {C_DIM}{'─' * 45}{C_RESET}")
    print(f"  {C_TEXT}Best epoch:   {C_BRIGHT}{ckpt['epoch']}{C_RESET}")
    print(f"  {C_TEXT}Total time:   {C_BRIGHT}{total_time:.0f}s ({total_time/60:.1f}m){C_RESET}")
    print(f"  {C_TEXT}Parameters:   {C_BRIGHT}{n_params:,}{C_RESET}")
    print(f"  {C_TEXT}Backbone:     {C_BRIGHT}{n_backbone:,}{C_RESET}")
    print(f"  {C_TEXT}Checkpoint:   {C_BRIGHT}{checkpoint_path / 'best_fb15k.pt'}{C_RESET}")

    if test_link_auc > 0.85:
        print(f"\n  {C_SUCCESS}{C_BOLD}STRUCTURAL REASONING VALIDATED ON REAL DATA{C_RESET}")
        print(f"  {C_TEXT}GATv2Conv learns multi-relational link prediction.{C_RESET}")
        print(f"  {C_TEXT}Ready to scale to ogbl-biokg for primary pre-training.{C_RESET}")
    else:
        print(f"\n  {C_DANGER}Link AUC below threshold — architecture may need tuning{C_RESET}")

    print()
    return {
        "test_link_auc": test_link_auc,
        "test_type_acc": test_type_acc,
        **test_mrr,
    }


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train SableGNN on FB15k-237 knowledge graph"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--edge-embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--neg-ratio", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--eval", type=str, default=None, help="Evaluate checkpoint only")
    args = parser.parse_args()

    config = KGConfig(
        epochs=args.epochs,
        lr=args.lr,
        embed_dim=args.embed_dim,
        edge_embed_dim=args.edge_embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        neg_ratio=args.neg_ratio,
        patience=args.patience,
        device=args.device,
    )

    if args.eval:
        # Eval-only mode
        print(f"\n{C_GOLD}{C_BOLD}  Evaluating checkpoint: {args.eval}{C_RESET}\n")
        ckpt = torch.load(args.eval, weights_only=False)
        cfg = ckpt["config"]
        data_root = args.data_root or str(Path(__file__).parent / "data")
        test_data = FB15k_237(root=data_root, split="test")[0]
        train_data = FB15k_237(root=data_root, split="train")[0]
        val_data = FB15k_237(root=data_root, split="val")[0]

        model = SableGNN_KG(
            num_nodes=cfg["num_nodes"],
            num_relations=cfg["num_relations"],
            embed_dim=cfg["embed_dim"],
            edge_embed_dim=cfg["edge_embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            heads=cfg["heads"],
            dropout=cfg["dropout"],
        ).to(config.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        all_ei = torch.cat([
            train_data.edge_index, val_data.edge_index
        ], dim=1).to(config.device)
        all_et = torch.cat([
            train_data.edge_type, val_data.edge_type
        ]).to(config.device)

        with torch.no_grad():
            node_emb = model.encode(all_ei, all_et)
        test_ei = test_data.edge_index.to(config.device)

        mrr = compute_mrr_hits(model, node_emb, test_ei, cfg["num_nodes"])
        print(f"  MRR: {mrr['mrr']:.4f}  Hits@1: {mrr['hits@1']:.4f}  "
              f"Hits@3: {mrr['hits@3']:.4f}  Hits@10: {mrr['hits@10']:.4f}\n")
    else:
        train(config=config, data_root=args.data_root)


if __name__ == "__main__":
    main()
