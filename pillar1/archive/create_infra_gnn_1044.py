#!/usr/bin/env python3
"""
Project PARALLAX — Infrastructure GNN (1044-dim Native Types)
================================================================
Future upgrade path: replaces the 6 CORTEX node types with 20 native
infrastructure component types, eliminating the lossy type compression.

1030-dim (current):  1024 embedding + 6 CORTEX type one-hot
1044-dim (this file): 1024 embedding + 20 infrastructure type one-hot

To activate:
  1. Run this script to create the 1044-dim checkpoint
  2. Update InfrastructureAdapter to use INFRA_NODE_TYPES instead of GNN_NODE_TYPES
  3. Retrain on infrastructure data (the attention weights transfer, only input_proj resets)
  4. Update fusion pipeline references from 1030 → 1044

Requires: pretrained FB15k or infrastructure checkpoint for weight transfer.
"""

import sys
from pathlib import Path

import numpy as np
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

# ── Native Infrastructure Types (20) ─────────────────────────────────────

INFRA_NODE_TYPES = [
    "CORE_SWITCH", "ACCESS_SWITCH", "FIREWALL", "ROUTER", "LOAD_BALANCER",
    "SERVER_PHYSICAL", "SERVER_VIRTUAL", "HYPERVISOR", "STORAGE_ARRAY",
    "STORAGE_TARGET", "VDI_BROKER", "VDI_HOST", "DNS_SERVER", "DHCP_SERVER",
    "DOMAIN_CONTROLLER", "CERTIFICATE_AUTHORITY", "MONITORING_SERVER",
    "WAN_LINK", "INTERNET_GATEWAY", "APPLICATION_SERVICE",
]

INFRA_EMBEDDING_DIM = 1024
INFRA_NODE_FEAT_DIM = INFRA_EMBEDDING_DIM + len(INFRA_NODE_TYPES)  # 1044
INFRA_EDGE_FEAT_DIM = 8  # 7 relation one-hot + confidence (unchanged)
INFRA_N_RELATIONS = 7


class InfrastructureAdapter1044:
    """
    Direct infrastructure → GNN adapter with native 20-type encoding.

    No more CORTEX type compression. Each of the 20 infrastructure component
    types gets its own one-hot dimension. The GNN sees the full type resolution.

    Node features: 1024 embedding + 20 type one-hot = 1044
    Edge features: 7 relation one-hot + confidence = 8 (unchanged from 1030)
    """

    # Functional groups for embedding generation — types within a group
    # share similar embedding directions (semantically clustered)
    FUNCTIONAL_GROUPS = {
        "networking":  ["CORE_SWITCH", "ACCESS_SWITCH", "FIREWALL", "ROUTER", "LOAD_BALANCER"],
        "compute":     ["SERVER_PHYSICAL", "SERVER_VIRTUAL", "HYPERVISOR", "VDI_HOST", "APPLICATION_SERVICE"],
        "storage":     ["STORAGE_ARRAY", "STORAGE_TARGET"],
        "services":    ["VDI_BROKER", "DNS_SERVER", "DHCP_SERVER", "DOMAIN_CONTROLLER"],
        "management":  ["CERTIFICATE_AUTHORITY", "MONITORING_SERVER"],
        "external":    ["WAN_LINK", "INTERNET_GATEWAY"],
    }

    # Edge mapping (unchanged from 1030 adapter)
    EDGE_TYPE_MAP = {
        "NETWORK_PATH": 5,           # related
        "POWER_DEPENDENCY": 3,       # depends_on
        "SERVICE_DEPENDENCY": 3,     # depends_on
        "STORAGE_DEPENDENCY": 3,     # depends_on
        "AUTHENTICATION_DEPENDENCY": 3,  # depends_on
        "DNS_DEPENDENCY": 0,         # supports
        "HOSTING_DEPENDENCY": 2,     # elaborates
        "REPLICATION_DEPENDENCY": 0, # supports
        "MONITORING_DEPENDENCY": 5,  # related
    }

    GNN_RELATION_TYPES = ["supports", "contradicts", "elaborates", "depends_on",
                          "caused_by", "related", "supersedes"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.type_to_idx = {t: i for i, t in enumerate(INFRA_NODE_TYPES)}
        self.type_embeddings = self._generate_type_embeddings()

    def _generate_type_embeddings(self) -> dict[str, np.ndarray]:
        """Generate pseudo-embeddings for infrastructure component types."""
        embeddings = {}

        group_centroids = {
            name: self.rng.randn(INFRA_EMBEDDING_DIM) * 0.3
            for name in self.FUNCTIONAL_GROUPS
        }

        type_to_group = {}
        for group, types in self.FUNCTIONAL_GROUPS.items():
            for t in types:
                type_to_group[t] = group

        for comp_type in INFRA_NODE_TYPES:
            group = type_to_group[comp_type]
            base = group_centroids[group].copy()
            noise = self.rng.randn(INFRA_EMBEDDING_DIM) * 0.15
            emb = base + noise
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings[comp_type] = emb

        return embeddings

    def adapt_node(self, comp_type: str, health: float = 1.0) -> np.ndarray:
        """Convert an infrastructure node to 1044-dim GNN feature vector.

        Returns: 1044-dim vector (1024 embedding + 20 type one-hot)
        """
        emb = self.type_embeddings.get(comp_type, np.zeros(INFRA_EMBEDDING_DIM))

        if health < 1.0:
            perturbation = self.rng.randn(INFRA_EMBEDDING_DIM) * (1.0 - health) * 0.2
            emb = emb + perturbation
            emb = emb / (np.linalg.norm(emb) + 1e-8)

        # Native 20-type one-hot — no compression
        type_one_hot = np.zeros(len(INFRA_NODE_TYPES), dtype=np.float32)
        idx = self.type_to_idx.get(comp_type, 0)
        type_one_hot[idx] = 1.0

        return np.concatenate([emb, type_one_hot]).astype(np.float32)

    def adapt_edge(self, dep_type: str, criticality: str = "SOFT",
                   confidence: float = 0.8) -> tuple[int, np.ndarray]:
        """Convert an infrastructure edge to GNN edge features.

        Returns: (relation_type_index, 8-dim edge feature vector)
        Edge format is unchanged from 1030 adapter.
        """
        rel_idx = self.EDGE_TYPE_MAP.get(dep_type, 5)

        crit_map = {"HARD": 1.0, "SOFT": 0.7, "REDUNDANT": 0.3}
        conf = crit_map.get(criticality, confidence)

        rel_one_hot = [0.0] * len(self.GNN_RELATION_TYPES)
        rel_one_hot[rel_idx] = 1.0
        edge_feat = rel_one_hot + [conf]

        return rel_idx, np.array(edge_feat, dtype=np.float32)


def create_infra_gnn_1044():
    """Create 1044-dim infrastructure GNN with pretrained weight transfer."""

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Infrastructure GNN 1044 (Native Types){C_RESET}")
    print(f"  {C_DIM}{'═' * 55}{C_RESET}\n")

    # Try to load existing infrastructure checkpoint (1030) for maximum transfer
    ckpt_path = Path(__file__).parent / "checkpoints" / "best_model.pt"
    fb15k_path = Path(__file__).parent / "checkpoints_fb15k" / "best_fb15k.pt"

    if ckpt_path.exists():
        src_path = ckpt_path
        print(f"  {C_INFO}Source: infrastructure checkpoint (1030-dim){C_RESET}")
    elif fb15k_path.exists():
        src_path = fb15k_path
        print(f"  {C_INFO}Source: FB15k checkpoint{C_RESET}")
    else:
        print(f"  {C_TEXT}No source checkpoint found. Creating from scratch.{C_RESET}")
        src_path = None

    if src_path:
        ckpt = torch.load(src_path, weights_only=False, map_location="cpu")
        src_config = ckpt["config"]
    else:
        src_config = {
            "hidden_dim": 256, "num_layers": 3, "heads": 4, "dropout": 0.1,
        }

    print(f"  {C_TEXT}  hidden_dim: {C_BRIGHT}{src_config['hidden_dim']}{C_RESET}")
    print(f"  {C_TEXT}  num_layers: {C_BRIGHT}{src_config['num_layers']}{C_RESET}")
    print(f"  {C_TEXT}  heads:      {C_BRIGHT}{src_config['heads']}{C_RESET}")
    print(f"  {C_TEXT}  in_dim:     {C_BRIGHT}1030 → 1044{C_RESET}")

    # Create 1044-dim model
    model = SableGNN(
        in_dim=INFRA_NODE_FEAT_DIM,
        hidden_dim=src_config["hidden_dim"],
        edge_dim=INFRA_EDGE_FEAT_DIM,
        num_layers=src_config["num_layers"],
        heads=src_config["heads"],
        n_relation_types=INFRA_N_RELATIONS,
        dropout=src_config["dropout"],
    )

    if src_path:
        src_key = "model_state_dict" if "model_state_dict" in ckpt else "gnn_state_dict"
        src_state = ckpt[src_key]
        new_state = model.state_dict()

        transferred = 0
        reinitialized = 0

        print(f"\n  {C_INFO}Weight transfer:{C_RESET}")
        for key in new_state:
            if key in src_state and new_state[key].shape == src_state[key].shape:
                new_state[key] = src_state[key]
                transferred += 1
            elif key in src_state:
                # input_proj weight/bias will mismatch (1030 vs 1044 input)
                # Transfer what we can: the first 1024 dims of input_proj.weight
                if "input_proj" in key and "weight" in key:
                    src_w = src_state[key]  # (hidden, 1030)
                    new_w = new_state[key]  # (hidden, 1044)
                    # Copy embedding columns (0:1024) directly
                    new_w[:, :INFRA_EMBEDDING_DIM] = src_w[:, :INFRA_EMBEDDING_DIM]
                    # Type columns (1024:1044) initialized fresh — the whole point
                    nn.init.xavier_uniform_(new_w[:, INFRA_EMBEDDING_DIM:].unsqueeze(0))
                    new_state[key] = new_w
                    print(f"    {C_TEXT}PARTIAL {key}: transferred embedding columns, "
                          f"reinitialized type columns (6→20){C_RESET}")
                else:
                    print(f"    {C_DIM}SKIP {key}: {src_state[key].shape} → {new_state[key].shape}{C_RESET}")
                reinitialized += 1
            else:
                reinitialized += 1

        model.load_state_dict(new_state)
        print(f"\n  {C_TEXT}Transferred: {C_BRIGHT}{transferred}{C_RESET}")
        print(f"  {C_TEXT}Reinitialized: {C_BRIGHT}{reinitialized}{C_RESET}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {C_TEXT}Total params: {C_BRIGHT}{n_params:,}{C_RESET}")

    # Save
    out_path = Path(__file__).parent / "checkpoints_1044" / "best_model_1044.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "config": {
            "in_dim": INFRA_NODE_FEAT_DIM,
            "hidden_dim": src_config["hidden_dim"],
            "edge_dim": INFRA_EDGE_FEAT_DIM,
            "num_layers": src_config["num_layers"],
            "heads": src_config["heads"],
            "dropout": src_config["dropout"],
        },
        "source": str(src_path) if src_path else "scratch",
        "upgrade_notes": "1030→1044: 6 CORTEX types replaced with 20 native infrastructure types",
    }, out_path)

    print(f"\n  {C_SUCCESS}{C_BOLD}Saved:{C_RESET} {C_TEXT}{out_path}{C_RESET}")
    print(f"  {C_DIM}Pull this when ready to retrain with full type resolution.{C_RESET}")
    print()


if __name__ == "__main__":
    create_infra_gnn_1044()
