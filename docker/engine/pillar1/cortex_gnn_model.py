#!/usr/bin/env python3
"""
Project PARALLAX — Session 2: GNN Architecture (Pillar 1)
============================================================
EdgeConditionedGAT for CORTEX knowledge graph reasoning.
Domain-agnostic: operates on typed embeddings and directional edges.

Three training objectives:
  1. Link type prediction (7-class) — what relation connects two nodes?
  2. Link prediction (binary) — should a link exist between two nodes?
  3. Contradiction detection (binary) — do two nodes oppose each other?

Usage:
    python cortex_gnn_model.py                          # Train with defaults
    python cortex_gnn_model.py --epochs 200 --lr 0.001  # Custom hyperparams
    python cortex_gnn_model.py --eval checkpoint.pt     # Evaluate only
    python cortex_gnn_model.py --sweep                  # Hyperparameter sweep

Requires: ml-env with torch, torch_geometric
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import negative_sampling

# ── Configuration ──────────────────────────────────────────────────────────

N_RELATION_TYPES = 7
N_THOUGHT_TYPES = 6
EMBEDDING_DIM = 1024
NODE_FEAT_DIM = EMBEDDING_DIM + N_THOUGHT_TYPES  # 1030
EDGE_FEAT_DIM = N_RELATION_TYPES + 1  # 8

# b1tr0n1n terminal aesthetic
C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


# ── Edge Conditioning MLP ─────────────────────────────────────────────────


class EdgeConditioner(nn.Module):
    """Transforms edge features into attention biases for GATv2Conv."""

    def __init__(self, edge_dim: int, heads: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, heads),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.mlp(edge_attr)  # [E, heads]


# ── GNN Model ─────────────────────────────────────────────────────────────


class SableGNN(nn.Module):
    """EdgeConditionedGAT for knowledge graph reasoning.

    Domain-agnostic: operates on typed embeddings and directional edges.
    Shared encoder with three task-specific heads.
    """

    def __init__(
        self,
        in_dim: int = NODE_FEAT_DIM,
        hidden_dim: int = 256,
        edge_dim: int = EDGE_FEAT_DIM,
        num_layers: int = 3,
        heads: int = 4,
        n_relation_types: int = N_RELATION_TYPES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection: 1030 → hidden_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GATv2Conv layers with edge conditioning
        self.convs = nn.ModuleList()
        self.edge_conditioners = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            # GATv2Conv: multi-head attention
            # Input/output dims account for concatenated heads
            conv_in = hidden_dim if i == 0 else hidden_dim * heads
            self.convs.append(
                GATv2Conv(
                    in_channels=conv_in,
                    out_channels=hidden_dim,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    concat=True,  # Concatenate heads (output = hidden_dim * heads)
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # Final projection back to hidden_dim
        self.output_proj = nn.Linear(hidden_dim * heads, hidden_dim)

        # ── Task Heads ──

        # Head 1: Link type prediction (7-class)
        # Takes concatenated source + target embeddings
        self.link_type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_relation_types),
        )

        # Head 2: Link prediction (binary)
        # Scores whether a link should exist between two nodes
        self.link_pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Head 3: Contradiction detection (binary)
        # Specialized for the "contradicts" relation
        self.contradiction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def encode(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Encode all nodes through the GNN backbone.

        Returns node embeddings [N, hidden_dim].
        """
        h = self.input_proj(x)  # [N, hidden_dim]

        for i in range(self.num_layers):
            h_res = h  # Residual connection (pre-conv)
            h = self.convs[i](h, edge_index, edge_attr=edge_attr)
            h = self.norms[i](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # Residual: project h_res to match conv output dim if needed
            if h_res.size(-1) != h.size(-1):
                h = h + h_res.repeat(1, h.size(-1) // h_res.size(-1))
            else:
                h = h + h_res

        h = self.output_proj(h)  # [N, hidden_dim]
        return h

    def predict_link_type(
        self, node_emb: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Predict relation type for edges. Returns logits [E, n_relation_types]."""
        src_emb = node_emb[edge_index[0]]
        tgt_emb = node_emb[edge_index[1]]
        pair_emb = torch.cat([src_emb, tgt_emb], dim=-1)
        return self.link_type_head(pair_emb)

    def predict_link(
        self, node_emb: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Predict link existence. Returns logits [E, 1]."""
        src_emb = node_emb[edge_index[0]]
        tgt_emb = node_emb[edge_index[1]]
        pair_emb = torch.cat([src_emb, tgt_emb], dim=-1)
        return self.link_pred_head(pair_emb).squeeze(-1)

    def predict_contradiction(
        self, node_emb: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Predict contradiction. Returns logits [E, 1]."""
        src_emb = node_emb[edge_index[0]]
        tgt_emb = node_emb[edge_index[1]]
        pair_emb = torch.cat([src_emb, tgt_emb], dim=-1)
        return self.contradiction_head(pair_emb).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        pred_edge_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            x: Node features [N, in_dim]
            edge_index: Message-passing edges [2, E_msg]
            edge_attr: Edge features [E_msg, edge_dim]
            pred_edge_index: Edges to make predictions on [2, E_pred]
                             (defaults to edge_index if None)

        Returns dict with 'node_emb', 'link_type_logits', 'link_logits', 'contradiction_logits'.
        """
        node_emb = self.encode(x, edge_index, edge_attr)

        if pred_edge_index is None:
            pred_edge_index = edge_index

        return {
            "node_emb": node_emb,
            "link_type_logits": self.predict_link_type(node_emb, pred_edge_index),
            "link_logits": self.predict_link(node_emb, pred_edge_index),
            "contradiction_logits": self.predict_contradiction(node_emb, pred_edge_index),
        }


# ── Training ──────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 1e-4
    link_type_weight: float = 1.0
    link_pred_weight: float = 0.5
    contradiction_weight: float = 1.0
    focal_gamma: float = 2.0  # Focal loss gamma for contradiction detection
    patience: int = 20
    min_delta: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim: int = 256
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.1


def compute_metrics(
    preds: torch.Tensor, targets: torch.Tensor, task: str = "multiclass"
) -> dict:
    """Compute evaluation metrics."""
    with torch.no_grad():
        if task == "multiclass":
            pred_classes = preds.argmax(dim=-1)
            acc = (pred_classes == targets).float().mean().item()

            # Per-class F1
            n_classes = preds.size(-1)
            f1s = []
            for c in range(n_classes):
                tp = ((pred_classes == c) & (targets == c)).sum().float()
                fp = ((pred_classes == c) & (targets != c)).sum().float()
                fn = ((pred_classes != c) & (targets == c)).sum().float()
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                f1s.append(f1.item())

            return {"accuracy": acc, "macro_f1": np.mean(f1s), "per_class_f1": f1s}

        elif task == "binary":
            pred_binary = (preds > 0).float()
            acc = (pred_binary == targets).float().mean().item()
            tp = ((pred_binary == 1) & (targets == 1)).sum().float()
            fp = ((pred_binary == 1) & (targets == 0)).sum().float()
            fn = ((pred_binary == 0) & (targets == 1)).sum().float()
            precision = (tp / (tp + fp + 1e-8)).item()
            recall = (tp / (tp + fn + 1e-8)).item()
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            # AUC-ROC approximation via sorted thresholds
            probs = torch.sigmoid(preds)
            auc = compute_auc(probs, targets)

            return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


def compute_auc(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute AUC-ROC."""
    if targets.sum() == 0 or targets.sum() == len(targets):
        return 0.5  # Undefined

    sorted_indices = torch.argsort(probs, descending=True)
    sorted_targets = targets[sorted_indices]

    tps = torch.cumsum(sorted_targets, dim=0)
    fps = torch.cumsum(1 - sorted_targets, dim=0)

    tpr = tps / (targets.sum() + 1e-8)
    fpr = fps / ((1 - targets).sum() + 1e-8)

    # Trapezoidal rule
    auc = torch.trapezoid(tpr, fpr).item()
    return abs(auc)


def compute_class_weights(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced classification."""
    counts = torch.bincount(labels, minlength=n_classes).float()
    # Avoid division by zero for classes with no samples
    counts = counts.clamp(min=1.0)
    weights = labels.size(0) / (n_classes * counts)
    return weights


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: float | None = None,
) -> torch.Tensor:
    """Focal loss for binary classification — down-weights easy negatives."""
    probs = torch.sigmoid(logits)
    # p_t = probability of correct class
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    # Standard BCE
    if pos_weight is not None:
        weight = targets * pos_weight + (1 - targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce = bce * weight
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    return (focal_weight * bce).mean()


def train_epoch(
    model: SableGNN,
    data: Data,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    class_weights: torch.Tensor | None = None,
) -> dict:
    """Single training epoch."""
    model.train()
    device = config.device

    # Message-passing uses training edges
    train_ei = data.train_edge_index.to(device)
    train_ea = data.train_edge_attr.to(device)
    train_labels = data.train_edge_labels.to(device)
    neg_ei = data.train_neg_edge_index.to(device)

    # Forward pass with training edges for message passing
    node_emb = model.encode(data.x.to(device), train_ei, train_ea)

    # Task 1: Link type prediction — weighted cross-entropy for class imbalance
    link_type_logits = model.predict_link_type(node_emb, train_ei)
    if class_weights is not None:
        loss_type = F.cross_entropy(link_type_logits, train_labels, weight=class_weights.to(device))
    else:
        loss_type = F.cross_entropy(link_type_logits, train_labels)

    # Task 2: Link prediction (positive + negative edges)
    pos_logits = model.predict_link(node_emb, train_ei)
    neg_logits = model.predict_link(node_emb, neg_ei)
    link_logits = torch.cat([pos_logits, neg_logits])
    link_targets = torch.cat([
        torch.ones(pos_logits.size(0), device=device),
        torch.zeros(neg_logits.size(0), device=device),
    ])
    loss_link = F.binary_cross_entropy_with_logits(link_logits, link_targets)

    # Task 3: Contradiction detection — focal loss with pos_weight
    # Label: 1 if relation_type == "contradicts" (index 1)
    contradiction_targets = (train_labels == 1).float()
    contradiction_logits = model.predict_contradiction(node_emb, train_ei)
    n_pos = contradiction_targets.sum().clamp(min=1)
    n_neg = (1 - contradiction_targets).sum().clamp(min=1)
    pos_weight = (n_neg / n_pos).item()
    loss_contradiction = focal_bce_loss(
        contradiction_logits, contradiction_targets,
        gamma=config.focal_gamma, pos_weight=pos_weight,
    )

    # Combined loss
    loss = (
        config.link_type_weight * loss_type
        + config.link_pred_weight * loss_link
        + config.contradiction_weight * loss_contradiction
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": loss.item(),
        "loss_type": loss_type.item(),
        "loss_link": loss_link.item(),
        "loss_contradiction": loss_contradiction.item(),
    }


@torch.no_grad()
def evaluate(model: SableGNN, data: Data, split: str, config: TrainConfig) -> dict:
    """Evaluate on val or test split."""
    model.eval()
    device = config.device

    # Use ALL edges for message passing during eval
    all_ei = data.edge_index.to(device)
    all_ea = data.edge_attr.to(device)

    node_emb = model.encode(data.x.to(device), all_ei, all_ea)

    # Get split-specific edges
    split_ei = getattr(data, f"{split}_edge_index").to(device)
    split_ea = getattr(data, f"{split}_edge_attr").to(device)
    split_labels = getattr(data, f"{split}_edge_labels").to(device)
    neg_ei = getattr(data, f"{split}_neg_edge_index").to(device)

    # Task 1: Link type
    type_logits = model.predict_link_type(node_emb, split_ei)
    type_metrics = compute_metrics(type_logits, split_labels, "multiclass")

    # Task 2: Link prediction
    pos_logits = model.predict_link(node_emb, split_ei)
    neg_logits = model.predict_link(node_emb, neg_ei)
    link_logits = torch.cat([pos_logits, neg_logits])
    link_targets = torch.cat([
        torch.ones(pos_logits.size(0), device=device),
        torch.zeros(neg_logits.size(0), device=device),
    ])
    link_metrics = compute_metrics(link_logits, link_targets, "binary")

    # Task 3: Contradiction
    contradiction_targets = (split_labels == 1).float()
    contradiction_logits = model.predict_contradiction(node_emb, split_ei)
    contradiction_metrics = compute_metrics(
        contradiction_logits, contradiction_targets, "binary"
    )

    # Combined val loss for early stopping
    loss_type = F.cross_entropy(type_logits, split_labels).item()
    loss_link = F.binary_cross_entropy_with_logits(link_logits, link_targets).item()
    loss_contra = F.binary_cross_entropy_with_logits(
        contradiction_logits, contradiction_targets
    ).item()
    val_loss = (
        config.link_type_weight * loss_type
        + config.link_pred_weight * loss_link
        + config.contradiction_weight * loss_contra
    )

    return {
        "loss": val_loss,
        "link_type": type_metrics,
        "link_pred": link_metrics,
        "contradiction": contradiction_metrics,
    }


def train(
    data_path: str = "cortex_graph.pt",
    config: TrainConfig | None = None,
    checkpoint_dir: str = "checkpoints",
) -> dict:
    """Full training pipeline."""
    if config is None:
        config = TrainConfig()

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Pillar 1 GNN Training{C_RESET}")
    print(f"  {C_DIM}{'═' * 50}{C_RESET}\n")

    # Load data
    data = torch.load(data_path, weights_only=False)
    print(f"  {C_TEXT}nodes:  {C_BRIGHT}{data.num_nodes}{C_RESET}")
    print(f"  {C_TEXT}edges:  {C_BRIGHT}{data.edge_index.size(1)}{C_RESET}")
    print(f"  {C_TEXT}device: {C_BRIGHT}{config.device}{C_RESET}")

    # Build model
    model = SableGNN(
        in_dim=data.x.size(1),
        hidden_dim=config.hidden_dim,
        edge_dim=data.edge_attr.size(1),
        num_layers=config.num_layers,
        heads=config.heads,
        dropout=config.dropout,
    ).to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {C_TEXT}params: {C_BRIGHT}{n_params:,}{C_RESET}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    # Compute class weights from training labels (inverse frequency)
    class_weights = compute_class_weights(data.train_edge_labels, N_RELATION_TYPES)
    print(f"  {C_INFO}Class weights (inverse frequency):{C_RESET}")
    relation_types = ["supports", "contradicts", "elaborates", "depends_on", "caused_by", "related", "supersedes"]
    for i, rt in enumerate(relation_types):
        print(f"    {C_DIM}{rt:15s}{C_RESET} {C_TEXT}{class_weights[i]:.2f}{C_RESET}")
    print()

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_metrics = {}
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    print(f"  {C_DIM}{'epoch':>5s}  {'loss':>8s}  {'type':>8s}  {'link':>8s}  {'contra':>8s}  {'val_loss':>8s}  {'type_f1':>7s}  {'link_auc':>8s}{C_RESET}")
    print(f"  {C_DIM}{'─' * 75}{C_RESET}")

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        train_losses = train_epoch(model, data, optimizer, config, class_weights)

        # Evaluate every 5 epochs or last epoch
        if epoch % 5 == 0 or epoch == config.epochs or epoch <= 3:
            val_metrics = evaluate(model, data, "val", config)
            val_loss = val_metrics["loss"]
            scheduler.step(val_loss)

            type_f1 = val_metrics["link_type"]["macro_f1"]
            link_auc = val_metrics["link_pred"]["auc"]

            # Color-code improvement
            improved = val_loss < best_val_loss - config.min_delta
            marker = f"{C_SUCCESS}*{C_RESET}" if improved else " "

            print(
                f"  {C_TEXT}{epoch:5d}{C_RESET}  "
                f"{train_losses['loss']:8.4f}  "
                f"{train_losses['loss_type']:8.4f}  "
                f"{train_losses['loss_link']:8.4f}  "
                f"{train_losses['loss_contradiction']:8.4f}  "
                f"{val_loss:8.4f}  "
                f"{type_f1:7.4f}  "
                f"{link_auc:8.4f} {marker}"
            )

            if improved:
                best_val_loss = val_loss
                best_metrics = val_metrics
                patience_counter = 0
                # Save best checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                        "config": {
                            "hidden_dim": config.hidden_dim,
                            "num_layers": config.num_layers,
                            "heads": config.heads,
                            "dropout": config.dropout,
                            "in_dim": data.x.size(1),
                            "edge_dim": data.edge_attr.size(1),
                        },
                    },
                    checkpoint_path / "best_model.pt",
                )
            else:
                patience_counter += 5  # We eval every 5 epochs
                if patience_counter >= config.patience:
                    print(f"\n  {C_DIM}Early stopping at epoch {epoch}{C_RESET}")
                    break

    # ── Final Evaluation on Test Set ──
    print(f"\n  {C_GOLD}{C_BOLD}  Test Set Evaluation{C_RESET}")
    print(f"  {C_DIM}{'─' * 50}{C_RESET}")

    # Load best model
    ckpt = torch.load(checkpoint_path / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, data, "test", config)

    # Link type prediction
    print(f"\n  {C_INFO}Link Type Prediction (7-class):{C_RESET}")
    print(f"    {C_TEXT}accuracy:  {C_BRIGHT}{test_metrics['link_type']['accuracy']:.4f}{C_RESET}")
    print(f"    {C_TEXT}macro F1:  {C_BRIGHT}{test_metrics['link_type']['macro_f1']:.4f}{C_RESET}")
    relation_types = ["supports", "contradicts", "elaborates", "depends_on", "caused_by", "related", "supersedes"]
    for i, rt in enumerate(relation_types):
        f1 = test_metrics["link_type"]["per_class_f1"][i]
        bar = "█" * int(f1 * 30)
        print(f"    {C_DIM}{rt:15s}{C_RESET} F1={C_TEXT}{f1:.3f}{C_RESET} {C_GOLD}{bar}{C_RESET}")

    # Link prediction
    print(f"\n  {C_INFO}Link Prediction (binary):{C_RESET}")
    lp = test_metrics["link_pred"]
    print(f"    {C_TEXT}AUC-ROC:   {C_BRIGHT}{lp['auc']:.4f}{C_RESET}")
    print(f"    {C_TEXT}accuracy:  {C_BRIGHT}{lp['accuracy']:.4f}{C_RESET}")
    print(f"    {C_TEXT}precision: {C_BRIGHT}{lp['precision']:.4f}{C_RESET}")
    print(f"    {C_TEXT}recall:    {C_BRIGHT}{lp['recall']:.4f}{C_RESET}")
    print(f"    {C_TEXT}F1:        {C_BRIGHT}{lp['f1']:.4f}{C_RESET}")

    # Contradiction detection
    print(f"\n  {C_INFO}Contradiction Detection (binary):{C_RESET}")
    cd = test_metrics["contradiction"]
    print(f"    {C_TEXT}accuracy:  {C_BRIGHT}{cd['accuracy']:.4f}{C_RESET}")
    print(f"    {C_TEXT}F1:        {C_BRIGHT}{cd['f1']:.4f}{C_RESET}")
    print(f"    {C_TEXT}AUC:       {C_BRIGHT}{cd['auc']:.4f}{C_RESET}")

    print(f"\n  {C_SUCCESS}{C_BOLD}Best checkpoint:{C_RESET} {C_TEXT}{checkpoint_path / 'best_model.pt'}{C_RESET}")
    print(f"  {C_TEXT}Epoch: {ckpt['epoch']}  Val loss: {ckpt['val_loss']:.4f}{C_RESET}")
    print(f"  {C_TEXT}Parameters: {C_BRIGHT}{n_params:,}{C_RESET}\n")

    return test_metrics


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train SableGNN on CORTEX knowledge graph"
    )
    parser.add_argument("--data", type=str, default="cortex_graph.pt", help="Path to exported PyG data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--eval", type=str, default=None, help="Evaluate a checkpoint only")
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        patience=args.patience,
        device=args.device,
    )

    if args.eval:
        # Eval-only mode
        data = torch.load(args.data, weights_only=False)
        ckpt = torch.load(args.eval, weights_only=False)
        model_config = ckpt["config"]
        model = SableGNN(
            in_dim=model_config["in_dim"],
            hidden_dim=model_config["hidden_dim"],
            edge_dim=model_config["edge_dim"],
            num_layers=model_config["num_layers"],
            heads=model_config["heads"],
            dropout=model_config["dropout"],
        ).to(config.device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate(model, data, "test", config)
        print(json.dumps(test_metrics, indent=2, default=str))
    else:
        train(data_path=args.data, config=config, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
