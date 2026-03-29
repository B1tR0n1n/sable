#!/usr/bin/env python3
"""
Project PARALLAX — Pillar 1: GNN Primary Pre-training on ogbl-biokg
=====================================================================
93,773 entities (5 types), 5,088,434 triples, 51 relation types.
Heterogeneous knowledge graph — structural analog to infrastructure:
  Entity types → component types (server, switch, router, storage, app)
  Relation types → dependency types (hosts, connects_to, depends_on, ...)

Uses DropEdge-style message passing subsampling to fit on single GPU.
Edge scoring uses mini-batched positive + negative sampling.

Usage:
    python train_biokg.py                              # Train with defaults
    python train_biokg.py --epochs 100 --msg-ratio 0.1 # Custom
    python train_biokg.py --eval checkpoints_biokg/best_biokg.pt

Requires: ml-env with torch, torch_geometric, ogb
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
from torch_geometric.utils import negative_sampling

# Patch torch.load for OGB compatibility (PyTorch 2.10 changed weights_only default)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

sys.path.insert(0, str(Path(__file__).parent))
from cortex_gnn_model import (
    SableGNN, compute_auc,
    C_GOLD, C_DIM, C_TEXT, C_BRIGHT, C_SUCCESS, C_INFO, C_DANGER, C_RESET, C_BOLD,
)


# ── Data Loading ──────────────────────────────────────────────────────────


def load_biokg(data_root: str) -> dict:
    """Load ogbl-biokg and convert to global indexing."""
    from ogb.linkproppred import LinkPropPredDataset

    print(f"  {C_INFO}Loading ogbl-biokg...{C_RESET}")
    dataset = LinkPropPredDataset(name="ogbl-biokg", root=data_root)
    split = dataset.get_edge_split()
    node_counts = dataset.graph["num_nodes_dict"]

    # Entity type ordering (fixed)
    entity_types = sorted(node_counts.keys())
    type_to_idx = {t: i for i, t in enumerate(entity_types)}
    n_types = len(entity_types)

    # Global index offsets: each type gets a contiguous range
    offsets = {}
    running = 0
    for t in entity_types:
        offsets[t] = running
        running += node_counts[t]
    total_nodes = running

    print(f"  {C_TEXT}Entity types:{C_RESET}")
    for t in entity_types:
        print(f"    {C_DIM}{t:20s}: {node_counts[t]:6d} (offset {offsets[t]}){C_RESET}")
    print(f"  {C_TEXT}Total nodes: {C_BRIGHT}{total_nodes:,}{C_RESET}")

    # Build entity type array for all nodes
    node_types = torch.zeros(total_nodes, dtype=torch.long)
    for t in entity_types:
        start = offsets[t]
        end = start + node_counts[t]
        node_types[start:end] = type_to_idx[t]

    def convert_split(s, type_key_prefix=""):
        """Convert a split's local indices to global."""
        head_type = s["head_type"]
        tail_type = s["tail_type"]
        head_local = s["head"]
        tail_local = s["tail"]
        relations = s["relation"]

        # Vectorized conversion: build offset lookup tensor
        head_offsets = torch.tensor([offsets[ht] for ht in head_type], dtype=torch.long)
        tail_offsets = torch.tensor([offsets[tt] for tt in tail_type], dtype=torch.long)
        head_global = torch.tensor(head_local, dtype=torch.long) + head_offsets
        tail_global = torch.tensor(tail_local, dtype=torch.long) + tail_offsets

        edge_index = torch.stack([head_global, tail_global])
        edge_type = torch.tensor(relations, dtype=torch.long)
        return edge_index, edge_type

    print(f"\n  {C_INFO}Converting to global indices...{C_RESET}")
    t0 = time.time()
    train_ei, train_et = convert_split(split["train"])
    val_ei, val_et = convert_split(split["valid"])
    test_ei, test_et = convert_split(split["test"])
    print(f"  {C_TEXT}Conversion: {time.time() - t0:.1f}s{C_RESET}")

    num_relations = int(train_et.max()) + 1

    print(f"  {C_TEXT}Train edges: {C_BRIGHT}{train_ei.size(1):,}{C_RESET}")
    print(f"  {C_TEXT}Val edges:   {C_BRIGHT}{val_ei.size(1):,}{C_RESET}")
    print(f"  {C_TEXT}Test edges:  {C_BRIGHT}{test_ei.size(1):,}{C_RESET}")
    print(f"  {C_TEXT}Relations:   {C_BRIGHT}{num_relations}{C_RESET}")

    return {
        "total_nodes": total_nodes,
        "num_relations": num_relations,
        "n_types": n_types,
        "node_types": node_types,
        "train_ei": train_ei,
        "train_et": train_et,
        "val_ei": val_ei,
        "val_et": val_et,
        "test_ei": test_ei,
        "test_et": test_et,
        "entity_types": entity_types,
        "offsets": offsets,
        "node_counts": node_counts,
    }


# ── KG Model (Heterogeneous) ─────────────────────────────────────────────


class SableGnnBioKg(nn.Module):
    """SableGNN with type-aware node embeddings for heterogeneous KG.

    Each entity gets: learnable embedding + entity type embedding.
    This teaches the GNN that different node types connect via
    different relation types — the structural analog to infrastructure
    component types connected by dependency types.
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        n_types: int = 5,
        embed_dim: int = 200,
        type_embed_dim: int = 56,
        edge_embed_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        total_node_dim = embed_dim + type_embed_dim

        # Learnable embeddings
        self.node_emb = nn.Embedding(num_nodes, embed_dim)
        self.type_emb = nn.Embedding(n_types, type_embed_dim)
        self.rel_emb = nn.Embedding(num_relations, edge_embed_dim)

        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.type_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # GNN backbone
        self.gnn = SableGNN(
            in_dim=total_node_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_embed_dim,
            num_layers=num_layers,
            heads=heads,
            n_relation_types=num_relations,
            dropout=dropout,
        )

    def get_node_features(self, node_types: torch.Tensor) -> torch.Tensor:
        """Build node features: entity embedding + type embedding."""
        entity_feat = self.node_emb.weight  # [N, embed_dim]
        type_feat = self.type_emb(node_types)  # [N, type_embed_dim]
        return torch.cat([entity_feat, type_feat], dim=-1)

    def encode(
        self,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Encode all nodes via GNN with message passing."""
        x = self.get_node_features(node_types)
        edge_attr = self.rel_emb(edge_type)
        return self.gnn.encode(x, edge_index, edge_attr)

    def predict_link_type(self, node_emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gnn.predict_link_type(node_emb, edge_index)

    def predict_link(self, node_emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gnn.predict_link(node_emb, edge_index)


# ── Configuration ─────────────────────────────────────────────────────────


@dataclass
class BioKGConfig:
    epochs: int = 100
    lr: float = 0.0005
    weight_decay: float = 1e-4
    embed_dim: int = 200
    type_embed_dim: int = 56
    edge_embed_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.1
    msg_ratio: float = 0.02  # Fraction of edges for message passing (DropEdge)
    score_batch: int = 100000  # Edges to score per training step
    neg_ratio: int = 5
    link_type_weight: float = 1.0
    link_pred_weight: float = 0.5
    patience: int = 20
    eval_every: int = 5
    mrr_sample: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ── MRR ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_mrr(model, node_emb, edge_index, num_nodes, n_eval=2000, n_corrupt=500):
    """Sampled MRR computation."""
    device = node_emb.device
    n_edges = edge_index.size(1)
    eval_idx = torch.randperm(n_edges, device=device)[:min(n_eval, n_edges)]
    eval_ei = edge_index[:, eval_idx]

    ranks = []
    for i in range(0, eval_ei.size(1), 256):
        batch_ei = eval_ei[:, i:i + 256]
        B = batch_ei.size(1)
        true_scores = model.predict_link(node_emb, batch_ei)

        for b in range(B):
            corrupt_tails = torch.randint(0, num_nodes, (n_corrupt,), device=device)
            corrupt_ei = torch.stack([batch_ei[0, b].expand(n_corrupt), corrupt_tails])
            corrupt_scores = model.predict_link(node_emb, corrupt_ei)
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


def train(data: dict, config: BioKGConfig | None = None):
    if config is None:
        config = BioKGConfig()

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Pillar 1: ogbl-biokg Pre-training{C_RESET}")
    print(f"  {C_DIM}{'═' * 55}{C_RESET}\n")

    total_nodes = data["total_nodes"]
    num_relations = data["num_relations"]
    n_types = data["n_types"]
    node_types = data["node_types"].to(config.device)
    train_ei = data["train_ei"].to(config.device)
    train_et = data["train_et"].to(config.device)
    val_ei = data["val_ei"].to(config.device)
    val_et = data["val_et"].to(config.device)

    n_train = train_ei.size(1)
    msg_edges = int(n_train * config.msg_ratio)

    print(f"  {C_TEXT}nodes:       {C_BRIGHT}{total_nodes:,}{C_RESET}")
    print(f"  {C_TEXT}relations:   {C_BRIGHT}{num_relations}{C_RESET}")
    print(f"  {C_TEXT}train edges: {C_BRIGHT}{n_train:,}{C_RESET}")
    print(f"  {C_TEXT}msg edges:   {C_BRIGHT}{msg_edges:,} ({config.msg_ratio:.0%} DropEdge){C_RESET}")
    print(f"  {C_TEXT}score batch: {C_BRIGHT}{config.score_batch:,}{C_RESET}")
    print(f"  {C_TEXT}device:      {C_BRIGHT}{config.device}{C_RESET}")

    # Build model
    model = SableGnnBioKg(
        num_nodes=total_nodes,
        num_relations=num_relations,
        n_types=n_types,
        embed_dim=config.embed_dim,
        type_embed_dim=config.type_embed_dim,
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    # Class weights for link type (51 relations)
    class_counts = torch.bincount(train_et.cpu(), minlength=num_relations).float().clamp(min=1)
    class_weights = (n_train / (num_relations * class_counts)).to(config.device)
    class_weights = class_weights.clamp(max=20.0)

    checkpoint_path = Path(__file__).parent / "checkpoints_biokg"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_val_metric = -1.0
    patience_counter = 0

    print(f"\n  {C_DIM}{'ep':>4s} {'loss':>8s} {'type_l':>8s} {'link_l':>8s} {'v_type':>8s} {'v_auc':>8s} {'v_mrr':>8s} {'h@10':>8s} {'time':>6s}{C_RESET}")
    print(f"  {C_DIM}{'─' * 80}{C_RESET}")

    t_start = time.time()

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        model.train()

        # DropEdge: sample subset for message passing
        msg_mask = torch.randperm(n_train, device=config.device)[:msg_edges]
        msg_ei = train_ei[:, msg_mask]
        msg_et = train_et[msg_mask]

        # Encode with subsampled message-passing edges
        node_emb = model.encode(node_types, msg_ei, msg_et)

        # Sample edges for scoring
        score_mask = torch.randperm(n_train, device=config.device)[:config.score_batch]
        score_ei = train_ei[:, score_mask]
        score_et = train_et[score_mask]

        # Task 1: Link type prediction
        type_logits = model.predict_link_type(node_emb, score_ei)
        loss_type = F.cross_entropy(type_logits, score_et, weight=class_weights)

        # Task 2: Link prediction
        neg_ei = negative_sampling(
            train_ei, num_nodes=total_nodes,
            num_neg_samples=config.score_batch * config.neg_ratio,
        )
        pos_logits = model.predict_link(node_emb, score_ei)
        neg_logits = model.predict_link(node_emb, neg_ei)
        link_logits = torch.cat([pos_logits, neg_logits])
        link_targets = torch.cat([
            torch.ones(pos_logits.size(0), device=config.device),
            torch.zeros(neg_logits.size(0), device=config.device),
        ])
        loss_link = F.binary_cross_entropy_with_logits(link_logits, link_targets)

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
            # Use more edges for eval message passing
            eval_msg = int(n_train * min(config.msg_ratio * 3, 0.2))
            eval_mask = torch.randperm(n_train, device=config.device)[:eval_msg]
            node_emb = model.encode(node_types, train_ei[:, eval_mask], train_et[eval_mask])

            # Val link type
            val_sample = torch.randperm(val_ei.size(1), device=config.device)[:50000]
            val_type_logits = model.predict_link_type(node_emb, val_ei[:, val_sample])
            val_type_acc = (val_type_logits.argmax(-1) == val_et[val_sample]).float().mean().item()

            # Val link AUC
            val_neg_ei = negative_sampling(val_ei, num_nodes=total_nodes, num_neg_samples=50000)
            val_pos_logits = model.predict_link(node_emb, val_ei[:, val_sample])
            val_neg_logits = model.predict_link(node_emb, val_neg_ei[:, :50000])
            val_all = torch.cat([val_pos_logits, val_neg_logits])
            val_targets = torch.cat([
                torch.ones(val_pos_logits.size(0), device=config.device),
                torch.zeros(val_neg_logits.size(0), device=config.device),
            ])
            val_auc = compute_auc(torch.sigmoid(val_all), val_targets)

        # MRR (expensive)
        mrr_metrics = {"mrr": 0.0, "hits@10": 0.0}
        if epoch % 25 == 0 or epoch == config.epochs:
            mrr_metrics = compute_mrr(
                model, node_emb, val_ei, total_nodes,
                n_eval=2000, n_corrupt=config.mrr_sample,
            )

        val_metric = val_auc
        improved = val_metric > best_val_metric + 0.001
        marker = f"{C_SUCCESS}*{C_RESET}" if improved else " "

        print(
            f"  {C_TEXT}{epoch:4d}{C_RESET} "
            f"{loss.item():8.4f} "
            f"{loss_type.item():8.4f} "
            f"{loss_link.item():8.4f} "
            f"{val_type_acc:8.4f} "
            f"{val_auc:8.4f} "
            f"{mrr_metrics['mrr']:8.4f} "
            f"{mrr_metrics.get('hits@10', 0):8.4f} "
            f"{elapsed:5.1f}s {marker}"
        )

        if improved:
            best_val_metric = val_metric
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "gnn_state_dict": model.gnn.state_dict(),
                "val_auc": val_auc,
                "val_type_acc": val_type_acc,
                **mrr_metrics,
                "config": {
                    "num_nodes": total_nodes,
                    "num_relations": num_relations,
                    "n_types": n_types,
                    "embed_dim": config.embed_dim,
                    "type_embed_dim": config.type_embed_dim,
                    "edge_embed_dim": config.edge_embed_dim,
                    "hidden_dim": config.hidden_dim,
                    "num_layers": config.num_layers,
                    "heads": config.heads,
                    "dropout": config.dropout,
                },
                "n_params": n_params,
                "n_backbone": n_backbone,
            }, checkpoint_path / "best_biokg.pt")
        else:
            patience_counter += config.eval_every
            if patience_counter >= config.patience * config.eval_every:
                print(f"\n  {C_DIM}Early stopping at epoch {epoch}{C_RESET}")
                break

    total_time = time.time() - t_start

    # ── Test ──
    print(f"\n{C_GOLD}{C_BOLD}  Test Set Evaluation{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")

    ckpt = torch.load(checkpoint_path / "best_biokg.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ei = data["test_ei"].to(config.device)
    test_et = data["test_et"].to(config.device)

    with torch.no_grad():
        # Use more edges for test message passing
        all_train_val_ei = torch.cat([train_ei, val_ei], dim=1)
        all_train_val_et = torch.cat([train_et, val_et])
        n_all = all_train_val_ei.size(1)
        test_msg = int(n_all * min(config.msg_ratio * 5, 0.3))
        test_mask = torch.randperm(n_all, device=config.device)[:test_msg]
        node_emb = model.encode(node_types, all_train_val_ei[:, test_mask], all_train_val_et[test_mask])

        # Test link type
        test_sample = torch.randperm(test_ei.size(1), device=config.device)[:50000]
        test_type_logits = model.predict_link_type(node_emb, test_ei[:, test_sample])
        test_type_acc = (test_type_logits.argmax(-1) == test_et[test_sample]).float().mean().item()

        # Test link AUC
        test_neg = negative_sampling(test_ei, num_nodes=total_nodes, num_neg_samples=50000)
        test_pos_logits = model.predict_link(node_emb, test_ei[:, test_sample])
        test_neg_logits = model.predict_link(node_emb, test_neg[:, :50000])
        test_all = torch.cat([test_pos_logits, test_neg_logits])
        test_targets = torch.cat([
            torch.ones(test_pos_logits.size(0), device=config.device),
            torch.zeros(test_neg_logits.size(0), device=config.device),
        ])
        test_auc = compute_auc(torch.sigmoid(test_all), test_targets)

    test_mrr = compute_mrr(model, node_emb, test_ei, total_nodes, n_eval=3000, n_corrupt=config.mrr_sample)

    print(f"\n  {C_INFO}Link Prediction:{C_RESET}")
    print(f"    {C_TEXT}AUC-ROC:  {C_BRIGHT}{test_auc:.4f}{C_RESET}")
    print(f"\n  {C_INFO}Knowledge Graph Ranking:{C_RESET}")
    print(f"    {C_TEXT}MRR:      {C_BRIGHT}{test_mrr['mrr']:.4f}{C_RESET}")
    print(f"    {C_TEXT}Hits@1:   {C_BRIGHT}{test_mrr['hits@1']:.4f}{C_RESET}")
    print(f"    {C_TEXT}Hits@3:   {C_BRIGHT}{test_mrr['hits@3']:.4f}{C_RESET}")
    print(f"    {C_TEXT}Hits@10:  {C_BRIGHT}{test_mrr['hits@10']:.4f}{C_RESET}")
    print(f"\n  {C_INFO}Link Type Classification ({num_relations}-class):{C_RESET}")
    print(f"    {C_TEXT}Accuracy: {C_BRIGHT}{test_type_acc:.4f}{C_RESET}")

    print(f"\n  {C_GOLD}{C_BOLD}  Summary{C_RESET}")
    print(f"  {C_DIM}{'─' * 45}{C_RESET}")
    print(f"  {C_TEXT}Best epoch:   {C_BRIGHT}{ckpt['epoch']}{C_RESET}")
    print(f"  {C_TEXT}Total time:   {C_BRIGHT}{total_time:.0f}s ({total_time/60:.1f}m){C_RESET}")
    print(f"  {C_TEXT}Parameters:   {C_BRIGHT}{n_params:,}{C_RESET}")
    print(f"  {C_TEXT}Backbone:     {C_BRIGHT}{n_backbone:,}{C_RESET}")
    print(f"  {C_TEXT}Checkpoint:   {C_BRIGHT}{checkpoint_path / 'best_biokg.pt'}{C_RESET}")

    if test_auc > 0.85:
        print(f"\n  {C_SUCCESS}{C_BOLD}HETEROGENEOUS STRUCTURAL REASONING VALIDATED{C_RESET}")
        print(f"  {C_TEXT}Type-aware GNN learns multi-relational reasoning at scale.{C_RESET}")
        print(f"  {C_TEXT}Ready for infrastructure domain fine-tuning.{C_RESET}")

    print()


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Pre-train SableGNN on ogbl-biokg")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--embed-dim", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--msg-ratio", type=float, default=0.05)
    parser.add_argument("--score-batch", type=int, default=100000)
    parser.add_argument("--neg-ratio", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--eval", type=str, default=None)
    args = parser.parse_args()

    config = BioKGConfig(
        epochs=args.epochs,
        lr=args.lr,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        msg_ratio=args.msg_ratio,
        score_batch=args.score_batch,
        neg_ratio=args.neg_ratio,
        patience=args.patience,
        device=args.device,
    )

    data = load_biokg(args.data_root)
    train(data, config)


if __name__ == "__main__":
    main()
