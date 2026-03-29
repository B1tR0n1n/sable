#!/usr/bin/env python3
"""
Project PARALLAX — Create Infrastructure-Adapted GNN
======================================================
Takes the pretrained FB15k-237 GATv2Conv backbone and adapts it for
infrastructure feature dimensions (1044 node features, 8 edge features).

1044 = 1024 embedding + 20 native infrastructure type one-hot.

Transfer: GATv2Conv attention weights, layer norms, output projection.
Reinitialize: input_proj (1044→256), edge conditioning (8→heads).

Saves as checkpoints/best_model.pt for compatibility with fusion pipeline.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from cortex_gnn_model import SableGNN

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


def create_infra_gnn():
    """Create infrastructure GNN with pretrained structural reasoning weights."""

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Infrastructure GNN Adaptation (1044){C_RESET}")
    print(f"  {C_DIM}{'═' * 55}{C_RESET}\n")

    # Target dimensions (infrastructure domain via InfrastructureAdapter)
    INFRA_NODE_DIM = 1044  # 1024 embedding + 20 native infrastructure type one-hot
    INFRA_EDGE_DIM = 8     # 7 relation one-hot + confidence
    INFRA_N_RELATIONS = 7

    # Source: FB15k-237 pretrained checkpoint
    fb15k_path = Path(__file__).parent / "checkpoints_fb15k" / "best_fb15k.pt"
    if not fb15k_path.exists():
        print(f"  {C_TEXT}FB15k checkpoint not found, using biokg...{C_RESET}")
        fb15k_path = Path(__file__).parent / "checkpoints_biokg" / "best_biokg.pt"

    ckpt = torch.load(fb15k_path, weights_only=False, map_location="cpu")
    src_config = ckpt["config"]
    print(f"  {C_INFO}Source checkpoint:{C_RESET} {fb15k_path}")
    print(f"  {C_TEXT}  hidden_dim: {C_BRIGHT}{src_config['hidden_dim']}{C_RESET}")
    print(f"  {C_TEXT}  num_layers: {C_BRIGHT}{src_config['num_layers']}{C_RESET}")
    print(f"  {C_TEXT}  heads:      {C_BRIGHT}{src_config['heads']}{C_RESET}")

    # Create infrastructure-dimensioned model with SAME backbone architecture
    infra_model = SableGNN(
        in_dim=INFRA_NODE_DIM,
        hidden_dim=src_config["hidden_dim"],
        edge_dim=INFRA_EDGE_DIM,
        num_layers=src_config["num_layers"],
        heads=src_config["heads"],
        n_relation_types=INFRA_N_RELATIONS,
        dropout=src_config["dropout"],
    )

    # Load source GNN state dict
    src_state = ckpt["gnn_state_dict"]

    # Transfer weights where shapes match
    infra_state = infra_model.state_dict()
    transferred = 0
    reinitialized = 0

    print(f"\n  {C_INFO}Weight transfer:{C_RESET}")
    for key in infra_state:
        if key in src_state and infra_state[key].shape == src_state[key].shape:
            infra_state[key] = src_state[key]
            transferred += 1
        elif key in src_state:
            print(f"    {C_DIM}SKIP {key}: {src_state[key].shape} → {infra_state[key].shape}{C_RESET}")
            reinitialized += 1
        else:
            reinitialized += 1

    infra_model.load_state_dict(infra_state)

    n_params = sum(p.numel() for p in infra_model.parameters())
    print(f"\n  {C_TEXT}Transferred: {C_BRIGHT}{transferred}{C_RESET} parameter tensors")
    print(f"  {C_TEXT}Reinitialized: {C_BRIGHT}{reinitialized}{C_RESET} parameter tensors")
    print(f"  {C_TEXT}Total params: {C_BRIGHT}{n_params:,}{C_RESET}")

    # Save in format compatible with fusion pipeline
    out_path = Path(__file__).parent / "checkpoints" / "best_model.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": 0,
        "model_state_dict": infra_model.state_dict(),
        "config": {
            "in_dim": INFRA_NODE_DIM,
            "hidden_dim": src_config["hidden_dim"],
            "edge_dim": INFRA_EDGE_DIM,
            "num_layers": src_config["num_layers"],
            "heads": src_config["heads"],
            "dropout": src_config["dropout"],
        },
        "source": str(fb15k_path),
        "transfer_info": f"{transferred} transferred, {reinitialized} reinitialized",
    }, out_path)

    print(f"\n  {C_SUCCESS}{C_BOLD}Saved:{C_RESET} {C_TEXT}{out_path}{C_RESET}")
    print(f"  {C_TEXT}Compatible with fusion pipeline (generate_fusion_data){C_RESET}")
    print()


if __name__ == "__main__":
    create_infra_gnn()
