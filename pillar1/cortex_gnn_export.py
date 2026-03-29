#!/usr/bin/env python3
"""
Project PARALLAX — Session 1: CORTEX → PyG Data Export
========================================================
Exports the CORTEX knowledge graph as a PyTorch Geometric dataset
for GNN training (Pillar 1: Causal Graph Reasoning).

Domain-agnostic: nodes are typed embeddings, edges are typed directional
links with confidence. No infrastructure assumptions.

Usage:
    python cortex_gnn_export.py                      # Export to default path
    python cortex_gnn_export.py --output graph.pt    # Custom output
    python cortex_gnn_export.py --stats              # Print graph stats only
    python cortex_gnn_export.py --split 0.7 0.15 0.15  # Custom train/val/test split

Requires: ml-env with torch, torch_geometric, httpx, numpy
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

# ── Configuration ──────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lqpvskwevanpgywdksqu.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Thought types in CORTEX — domain-agnostic type system
THOUGHT_TYPES = ["observation", "task", "idea", "reference", "person_note", "instruction"]
N_THOUGHT_TYPES = len(THOUGHT_TYPES)
THOUGHT_TYPE_INDEX = {t: i for i, t in enumerate(THOUGHT_TYPES)}

# Relation types in CORTEX — domain-agnostic edge semantics
RELATION_TYPES = [
    "supports", "contradicts", "elaborates",
    "depends_on", "caused_by", "related", "supersedes",
]
N_RELATION_TYPES = len(RELATION_TYPES)
RELATION_TYPE_INDEX = {t: i for i, t in enumerate(RELATION_TYPES)}

EMBEDDING_DIM = 1024

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


# ── Supabase Client (reused from retroactive_linker.py) ───────────────────


class SupabaseClient:
    """Direct Supabase REST API client for bulk operations."""

    def __init__(self, url: str = SUPABASE_URL, key: str = SUPABASE_KEY):
        self.url = url.rstrip("/")
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        self.client = httpx.Client(timeout=60)

    def fetch_all_thoughts(self) -> list[dict]:
        """Fetch all thoughts with embeddings from Supabase."""
        thoughts = []
        offset = 0
        batch_size = 1000

        while True:
            resp = self.client.get(
                f"{self.url}/rest/v1/thoughts",
                headers=self.headers,
                params={
                    "select": "id,content,embedding,metadata,created_at",
                    "order": "created_at.asc",
                    "offset": offset,
                    "limit": batch_size,
                },
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            thoughts.extend(batch)
            offset += len(batch)
            if len(batch) < batch_size:
                break

        return thoughts

    def fetch_all_links(self) -> list[dict]:
        """Fetch all thought_links with full metadata."""
        links = []
        offset = 0
        batch_size = 1000

        while True:
            resp = self.client.get(
                f"{self.url}/rest/v1/thought_links",
                headers=self.headers,
                params={
                    "select": "source_id,target_id,relation_type,confidence",
                    "offset": offset,
                    "limit": batch_size,
                },
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            links.extend(batch)
            offset += len(batch)
            if len(batch) < batch_size:
                break

        return links

    def close(self):
        self.client.close()


# ── Feature Encoding ──────────────────────────────────────────────────────


def encode_node_features(thoughts: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Encode thought nodes as feature vectors.

    Returns (feature_matrix [N x (EMBEDDING_DIM + N_THOUGHT_TYPES)], node_ids).
    Domain-agnostic: embedding + type one-hot. No infrastructure assumptions.
    """
    node_ids = []
    features = []
    skipped = 0

    for t in thoughts:
        raw_embedding = t.get("embedding")
        if raw_embedding is None:
            skipped += 1
            continue

        # pgvector returns embeddings as strings — parse to list of floats
        if isinstance(raw_embedding, str):
            try:
                cleaned = raw_embedding.strip()
                if cleaned.startswith("["):
                    embedding = json.loads(cleaned)
                else:
                    embedding = [float(x) for x in cleaned.split(",")]
            except (ValueError, json.JSONDecodeError):
                skipped += 1
                continue
        elif isinstance(raw_embedding, list):
            embedding = raw_embedding
        else:
            skipped += 1
            continue

        if len(embedding) != EMBEDDING_DIM:
            skipped += 1
            continue

        # Extract thought type from metadata
        metadata = t.get("metadata", {}) or {}
        thought_type = metadata.get("type", "observation")
        type_idx = THOUGHT_TYPE_INDEX.get(thought_type, 0)

        # One-hot type encoding
        type_one_hot = [0.0] * N_THOUGHT_TYPES
        type_one_hot[type_idx] = 1.0

        # Feature vector: embedding + type one-hot
        feat = list(embedding) + type_one_hot
        features.append(feat)
        node_ids.append(t["id"])

    if skipped:
        print(f"  {C_DIM}skipped {skipped} thoughts without valid embeddings{C_RESET}")

    return np.array(features, dtype=np.float32), node_ids


def encode_edges(
    links: list[dict], node_id_to_idx: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Encode thought_links as edge_index + edge_attr.

    Returns (edge_index [2 x E], edge_features [E x (N_RELATION_TYPES + 1)], edge_labels).
    edge_labels = relation_type index for each edge (training target).
    """
    sources = []
    targets = []
    edge_feats = []
    edge_labels = []
    skipped = 0

    for link in links:
        src_idx = node_id_to_idx.get(link["source_id"])
        tgt_idx = node_id_to_idx.get(link["target_id"])
        if src_idx is None or tgt_idx is None:
            skipped += 1
            continue

        rel_type = link.get("relation_type", "related")
        rel_idx = RELATION_TYPE_INDEX.get(rel_type, RELATION_TYPE_INDEX["related"])
        confidence = float(link.get("confidence", 1.0))

        # One-hot relation type + confidence
        rel_one_hot = [0.0] * N_RELATION_TYPES
        rel_one_hot[rel_idx] = 1.0
        feat = rel_one_hot + [confidence]

        sources.append(src_idx)
        targets.append(tgt_idx)
        edge_feats.append(feat)
        edge_labels.append(rel_idx)

    if skipped:
        print(f"  {C_DIM}skipped {skipped} links with missing node references{C_RESET}")

    edge_index = np.array([sources, targets], dtype=np.int64)
    edge_attr = np.array(edge_feats, dtype=np.float32) if edge_feats else np.zeros((0, N_RELATION_TYPES + 1), dtype=np.float32)

    return edge_index, edge_attr, edge_labels


# ── Edge Splitting ────────────────────────────────────────────────────────


def split_edges(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_labels: torch.Tensor,
    num_nodes: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    neg_ratio: int = 5,
    seed: int = 42,
) -> dict:
    """Split edges into train/val/test with negative sampling.

    Returns dict with train/val/test edge_index, edge_attr, edge_labels,
    plus negative edges for link prediction training.
    """
    rng = torch.Generator().manual_seed(seed)
    num_edges = edge_index.size(1)

    # Shuffle edge indices
    perm = torch.randperm(num_edges, generator=rng)
    n_train = int(num_edges * train_ratio)
    n_val = int(num_edges * val_ratio)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    # Generate negative samples for link prediction
    num_neg = num_edges * neg_ratio
    neg_edge_index = negative_sampling(
        edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
    )

    # Split negatives proportionally
    n_neg = neg_edge_index.size(1)
    neg_perm = torch.randperm(n_neg, generator=rng)
    n_neg_train = int(n_neg * train_ratio)
    n_neg_val = int(n_neg * val_ratio)

    return {
        "train": {
            "edge_index": edge_index[:, train_idx],
            "edge_attr": edge_attr[train_idx],
            "edge_labels": edge_labels[train_idx],
            "neg_edge_index": neg_edge_index[:, neg_perm[:n_neg_train]],
        },
        "val": {
            "edge_index": edge_index[:, val_idx],
            "edge_attr": edge_attr[val_idx],
            "edge_labels": edge_labels[val_idx],
            "neg_edge_index": neg_edge_index[:, neg_perm[n_neg_train:n_neg_train + n_neg_val]],
        },
        "test": {
            "edge_index": edge_index[:, test_idx],
            "edge_attr": edge_attr[test_idx],
            "edge_labels": edge_labels[test_idx],
            "neg_edge_index": neg_edge_index[:, neg_perm[n_neg_train + n_neg_val:]],
        },
    }


# ── Main Pipeline ─────────────────────────────────────────────────────────


def export_cortex_graph(
    output_path: str = "cortex_graph.pt",
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    stats_only: bool = False,
) -> dict:
    """Full pipeline: fetch CORTEX → encode → split → save PyG Data."""

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Pillar 1 Data Export{C_RESET}")
    print(f"  {C_DIM}{'═' * 50}{C_RESET}\n")

    # ── Fetch ──
    print(f"  {C_INFO}Fetching CORTEX graph...{C_RESET}")
    client = SupabaseClient()
    try:
        thoughts = client.fetch_all_thoughts()
        links = client.fetch_all_links()
    finally:
        client.close()

    print(f"  {C_TEXT}thoughts: {C_BRIGHT}{len(thoughts)}{C_RESET}")
    print(f"  {C_TEXT}links:    {C_BRIGHT}{len(links)}{C_RESET}")

    # ── Encode nodes ──
    print(f"\n  {C_INFO}Encoding node features...{C_RESET}")
    node_features, node_ids = encode_node_features(thoughts)
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)
    feat_dim = node_features.shape[1] if num_nodes > 0 else 0

    print(f"  {C_TEXT}nodes with embeddings: {C_BRIGHT}{num_nodes}{C_RESET}")
    print(f"  {C_TEXT}feature dim:          {C_BRIGHT}{feat_dim}{C_RESET} ({EMBEDDING_DIM} embedding + {N_THOUGHT_TYPES} type one-hot)")

    # ── Encode edges ──
    print(f"\n  {C_INFO}Encoding edge features...{C_RESET}")
    edge_index, edge_attr, edge_labels = encode_edges(links, node_id_to_idx)
    num_edges = edge_index.shape[1] if edge_index.size > 0 else 0
    edge_dim = edge_attr.shape[1] if num_edges > 0 else 0

    print(f"  {C_TEXT}edges encoded:        {C_BRIGHT}{num_edges}{C_RESET}")
    print(f"  {C_TEXT}edge feature dim:     {C_BRIGHT}{edge_dim}{C_RESET} ({N_RELATION_TYPES} relation one-hot + 1 confidence)")

    # ── Type distribution ──
    print(f"\n  {C_INFO}Thought type distribution:{C_RESET}")
    type_counts = {}
    for t in thoughts:
        tt = (t.get("metadata") or {}).get("type", "observation")
        type_counts[tt] = type_counts.get(tt, 0) + 1
    for tt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(count // 5, 40)
        print(f"    {C_DIM}{tt:15s}{C_RESET} {C_TEXT}{count:4d}{C_RESET} {C_GOLD}{bar}{C_RESET}")

    print(f"\n  {C_INFO}Relation type distribution:{C_RESET}")
    rel_counts = {}
    for link in links:
        rt = link.get("relation_type", "related")
        rel_counts[rt] = rel_counts.get(rt, 0) + 1
    for rt, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(count // 10, 40)
        print(f"    {C_DIM}{rt:15s}{C_RESET} {C_TEXT}{count:4d}{C_RESET} {C_GOLD}{bar}{C_RESET}")

    # ── Graph density ──
    density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
    print(f"\n  {C_TEXT}graph density:        {C_BRIGHT}{density:.6f}{C_RESET}")
    print(f"  {C_TEXT}avg degree:           {C_BRIGHT}{avg_degree:.1f}{C_RESET}")

    if stats_only:
        print(f"\n  {C_DIM}stats-only mode, skipping export{C_RESET}\n")
        return {"num_nodes": num_nodes, "num_edges": num_edges}

    # ── Convert to tensors ──
    x = torch.tensor(node_features, dtype=torch.float32)
    ei = torch.tensor(edge_index, dtype=torch.long)
    ea = torch.tensor(edge_attr, dtype=torch.float32)
    el = torch.tensor(edge_labels, dtype=torch.long)

    # ── Split edges ──
    print(f"\n  {C_INFO}Splitting edges ({split_ratios[0]:.0%}/{split_ratios[1]:.0%}/{split_ratios[2]:.0%})...{C_RESET}")
    splits = split_edges(ei, ea, el, num_nodes, split_ratios[0], split_ratios[1])

    for name, split in splits.items():
        n_pos = split["edge_index"].size(1)
        n_neg = split["neg_edge_index"].size(1)
        print(f"    {C_DIM}{name:5s}{C_RESET}  {C_TEXT}pos: {n_pos:5d}  neg: {n_neg:5d}{C_RESET}")

    # ── Build PyG Data ──
    data = Data(
        x=x,
        edge_index=ei,
        edge_attr=ea,
        edge_labels=el,
        num_nodes=num_nodes,
    )

    # Store splits and metadata as attributes
    data.train_edge_index = splits["train"]["edge_index"]
    data.train_edge_attr = splits["train"]["edge_attr"]
    data.train_edge_labels = splits["train"]["edge_labels"]
    data.train_neg_edge_index = splits["train"]["neg_edge_index"]

    data.val_edge_index = splits["val"]["edge_index"]
    data.val_edge_attr = splits["val"]["edge_attr"]
    data.val_edge_labels = splits["val"]["edge_labels"]
    data.val_neg_edge_index = splits["val"]["neg_edge_index"]

    data.test_edge_index = splits["test"]["edge_index"]
    data.test_edge_attr = splits["test"]["edge_attr"]
    data.test_edge_labels = splits["test"]["edge_labels"]
    data.test_neg_edge_index = splits["test"]["neg_edge_index"]

    # Store node ID mapping for inference
    data.node_ids = node_ids
    data.thought_types = THOUGHT_TYPES
    data.relation_types = RELATION_TYPES

    # ── Save ──
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(output))

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"\n  {C_SUCCESS}{C_BOLD}Exported:{C_RESET} {C_TEXT}{output}{C_RESET} ({size_mb:.1f} MB)")

    # ── Summary ──
    print(f"\n  {C_GOLD}{C_BOLD}  Dataset Summary{C_RESET}")
    print(f"  {C_DIM}{'─' * 40}{C_RESET}")
    print(f"  {C_TEXT}nodes:          {C_BRIGHT}{num_nodes}{C_RESET}")
    print(f"  {C_TEXT}edges:          {C_BRIGHT}{num_edges}{C_RESET}")
    print(f"  {C_TEXT}node feat dim:  {C_BRIGHT}{feat_dim}{C_RESET}")
    print(f"  {C_TEXT}edge feat dim:  {C_BRIGHT}{edge_dim}{C_RESET}")
    print(f"  {C_TEXT}thought types:  {C_BRIGHT}{N_THOUGHT_TYPES}{C_RESET}")
    print(f"  {C_TEXT}relation types: {C_BRIGHT}{N_RELATION_TYPES}{C_RESET}")
    print(f"  {C_TEXT}density:        {C_BRIGHT}{density:.6f}{C_RESET}")
    print(f"  {C_TEXT}avg degree:     {C_BRIGHT}{avg_degree:.1f}{C_RESET}")
    print()

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "feat_dim": feat_dim,
        "edge_dim": edge_dim,
        "output_path": str(output),
    }


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Export CORTEX knowledge graph to PyTorch Geometric format"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="cortex_graph.pt",
        help="Output .pt file path (default: cortex_graph.pt)",
    )
    parser.add_argument(
        "--split", nargs=3, type=float, default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios (default: 0.7 0.15 0.15)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print graph statistics only, don't export",
    )
    args = parser.parse_args()

    # Validate split ratios
    if abs(sum(args.split) - 1.0) > 0.01:
        print(f"{C_DANGER}Error: split ratios must sum to 1.0{C_RESET}")
        sys.exit(1)

    export_cortex_graph(
        output_path=args.output,
        split_ratios=tuple(args.split),
        stats_only=args.stats,
    )


if __name__ == "__main__":
    main()
