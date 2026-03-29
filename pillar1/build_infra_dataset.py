#!/usr/bin/env -S python -u
"""
Project PARALLAX — Build Infrastructure Topology Dataset
==========================================================
Converts real infrastructure topologies (Topology Zoo + Microservice graphs)
into PyG-compatible training data for GNN fine-tuning.

For each real topology:
  - Assign infrastructure node types (router, switch, server, etc.)
  - Map edge types from link labels
  - Simulate multiple failure scenarios on the topology
  - Generate per-node state labels from cascade propagation

Output: infra_graphs.pt — ready for fast training.

Usage:
    python build_infra_dataset.py
"""

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar3"))

from sable_sim.core.states import N_STATES, STATE_NAMES
from sable_sim.utils.random import SeededRandom

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"

# Node type encoding (8 types covering infrastructure roles)
NODE_TYPES = {
    "router": 0, "switch": 1, "server": 2, "storage": 3,
    "service": 4, "gateway": 5, "endpoint": 6, "unknown": 7,
}
N_NODE_TYPES = len(NODE_TYPES)

# Edge type encoding (6 dependency types)
EDGE_TYPES = {
    "backbone": 0, "access": 1, "service_dep": 2,
    "storage_dep": 3, "management": 4, "unknown": 5,
}
N_EDGE_TYPES = len(EDGE_TYPES)

# Node feature dim: type one-hot (8) + degree_norm (1) + health (1) = 10
NODE_FEAT_DIM = N_NODE_TYPES + 2
# Edge feature dim: type one-hot (6) + weight (1) = 7
EDGE_FEAT_DIM = N_EDGE_TYPES + 1


def classify_zoo_node(node_data: dict, degree: int, max_degree: int) -> int:
    """Classify a Topology Zoo node into infrastructure type by degree/role."""
    internal = node_data.get("Internal", 1)

    if not internal:
        return NODE_TYPES["gateway"]

    # High-degree nodes are core routers/switches
    if max_degree > 0 and degree >= max_degree * 0.7:
        return NODE_TYPES["router"]
    elif degree >= 3:
        return NODE_TYPES["switch"]
    elif degree == 1:
        return NODE_TYPES["endpoint"]
    else:
        return NODE_TYPES["server"]


def classify_zoo_edge(edge_data: dict) -> int:
    """Classify a Topology Zoo edge into infrastructure dependency type."""
    link_type = str(edge_data.get("LinkType", "")).lower()
    link_label = str(edge_data.get("LinkLabel", "")).lower()
    combined = link_type + " " + link_label

    if any(k in combined for k in ["oc-", "stm-", "dwdm", "wavelength", "10g", "100g", "backbone"]):
        return EDGE_TYPES["backbone"]
    elif any(k in combined for k in ["ethernet", "gigabit", "fast", "access"]):
        return EDGE_TYPES["access"]
    elif any(k in combined for k in ["ds-", "t1", "t3", "e1", "e3", "leased"]):
        return EDGE_TYPES["backbone"]
    else:
        return EDGE_TYPES["unknown"]


def classify_ms_node(node_id: str, node_data: dict) -> int:
    """Classify a microservice graph node."""
    label = str(node_data.get("label", node_id)).lower()

    if any(k in label for k in ["db", "database", "sql", "mongo", "redis", "postgres", "mysql"]):
        return NODE_TYPES["storage"]
    elif any(k in label for k in ["gateway", "api-gateway", "proxy", "nginx", "envoy", "load"]):
        return NODE_TYPES["gateway"]
    elif any(k in label for k in ["queue", "rabbit", "kafka", "bus", "message"]):
        return NODE_TYPES["switch"]  # message routing
    elif any(k in label for k in ["monitor", "log", "trace", "metrics"]):
        return NODE_TYPES["service"]
    else:
        return NODE_TYPES["server"]  # generic microservice


def _build_edge_features(G, node_idx, edge_types):
    """Build bidirectional edge index and edge feature arrays."""
    edges = list(G.edges())
    if not edges:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, EDGE_FEAT_DIM), dtype=np.float32)

    edge_index = np.array([[node_idx[u], node_idx[v]] for u, v in edges], dtype=np.int64).T
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)

    edge_feats = np.zeros((len(edges) * 2, EDGE_FEAT_DIM), dtype=np.float32)
    for j, (u, v) in enumerate(edges):
        etype = edge_types[j] if j < len(edge_types) else EDGE_TYPES["unknown"]
        for offset in [0, len(edges)]:
            edge_feats[j + offset, etype] = 1.0
            edge_feats[j + offset, N_EDGE_TYPES] = 1.0
    return edge_index, edge_feats


def _build_node_features(n, node_types, degrees, max_degree, health_fn):
    """Build node feature array with type one-hot, degree norm, and health."""
    node_feats = np.zeros((n, NODE_FEAT_DIM), dtype=np.float32)
    for i in range(n):
        node_feats[i, node_types[i]] = 1.0
        node_feats[i, N_NODE_TYPES] = degrees[i] / max(max_degree, 1)
        node_feats[i, N_NODE_TYPES + 1] = health_fn(i)
    return node_feats


def _cascade_root_failures(health, states, fail_indices):
    """Stage 1: Set root failure nodes to failed state."""
    for fi in fail_indices:
        health[fi] = 0.0
        states[fi] = 2  # failed


def _cascade_neighbor_degradation(nodes, adj, node_idx, health, states, fail_indices, rng):
    """Stage 2: Direct neighbors of failed nodes degrade probabilistically."""
    for fi in fail_indices:
        node = nodes[fi]
        for neighbor in adj[node]:
            ni = node_idx[neighbor]
            if states[ni] == 0 and rng.random() < 0.7:
                health[ni] = rng.uniform(0.2, 0.6)
                states[ni] = 1  # degraded


def _cascade_2hop(n, nodes, adj, node_idx, health, states, rng):
    """Stage 3: 2-hop cascade — degraded nodes may cause further degradation."""
    for i in range(n):
        if states[i] == 1:
            node = nodes[i]
            for neighbor in adj[node]:
                ni = node_idx[neighbor]
                if states[ni] == 0 and rng.random() < 0.3:
                    health[ni] = rng.uniform(0.4, 0.8)
                    states[ni] = 1


def _detect_unreachable(n, nodes, adj, node_idx, health, states, degrees):
    """Stage 4: Nodes cut off from all healthy high-degree nodes become unreachable."""
    healthy_hubs = set(i for i in range(n) if states[i] == 0 and degrees[i] >= 3)
    if not healthy_hubs:
        return
    for i in range(n):
        if states[i] == 0:
            node = nodes[i]
            has_healthy_path = any(
                node_idx[nb] in healthy_hubs or states[node_idx[nb]] == 0
                for nb in adj[node]
            )
            if not has_healthy_path and degrees[i] <= 1:
                states[i] = 3
                health[i] = 0.0


def _inject_oscillating(n, health, states, rng):
    """Stage 5: Mark 2-4 nodes as oscillating in 12% of scenarios."""
    if rng.random() >= 0.12:
        return
    n_osc = rng.choice([2, 2, 3, 4])
    osc_candidates = [i for i in range(n) if states[i] in (0, 1)]
    if len(osc_candidates) >= n_osc:
        osc_idx = np.random.default_rng(rng.randint(0, 2**31)).choice(
            osc_candidates, size=n_osc, replace=False)
        for oi in osc_idx:
            states[oi] = 4
            health[oi] = rng.uniform(0.3, 0.7)


def simulate_cascades(G: nx.Graph, node_types: list[int], edge_types: list[int],
                      n_scenarios: int, rng: SeededRandom) -> list[dict]:
    """Simulate failure cascades on a topology and generate labeled samples.

    For each scenario:
    - Pick 1-3 random nodes to fail
    - Propagate: neighbors of failed nodes degrade, 2-hop neighbors may degrade
    - Label all nodes: 0=healthy, 1=degraded, 2=failed, 3=unreachable
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n < 4:
        return []

    adj = {node: list(G.neighbors(node)) for node in nodes}
    node_idx = {node: i for i, node in enumerate(nodes)}
    degrees = np.array([G.degree(node) for node in nodes], dtype=np.float32)
    max_degree = degrees.max()

    samples = []

    for _ in range(n_scenarios):
        health = np.ones(n, dtype=np.float32)
        states = np.zeros(n, dtype=np.int64)

        # 12% of scenarios: no injection
        if rng.random() < 0.12:
            node_feats = _build_node_features(n, node_types, degrees, max_degree, lambda i: 1.0)
            edge_index, edge_feats = _build_edge_features(G, node_idx, edge_types)
            samples.append({
                "x": node_feats, "edge_index": edge_index, "edge_attr": edge_feats,
                "states": states, "health": health, "n_nodes": n,
                "n_edges": edge_index.shape[1] if edge_index.size else 0,
            })
            continue

        # Pick failure roots
        n_failures = rng.choice([1, 1, 1, 2, 2, 3])
        weights = degrees / degrees.sum()
        fail_indices = np.random.default_rng(rng.randint(0, 2**31)).choice(
            n, size=min(n_failures, n), replace=False, p=weights)

        _cascade_root_failures(health, states, fail_indices)
        _cascade_neighbor_degradation(nodes, adj, node_idx, health, states, fail_indices, rng)
        _cascade_2hop(n, nodes, adj, node_idx, health, states, rng)
        _detect_unreachable(n, nodes, adj, node_idx, health, states, degrees)
        _inject_oscillating(n, health, states, rng)

        fail_set = set(fail_indices)
        node_feats = _build_node_features(
            n, node_types, degrees, max_degree,
            lambda i: 1.0 if i not in fail_set else 0.0)
        edge_index, edge_feats = _build_edge_features(G, node_idx, edge_types)

        samples.append({
            "x": node_feats, "edge_index": edge_index, "edge_attr": edge_feats,
            "states": states, "health": health, "n_nodes": n,
            "n_edges": edge_index.shape[1],
        })

    return samples


def load_all_topologies(data_dir: str) -> list[tuple[nx.Graph, list[int], list[int], str]]:
    """Load all topologies and classify nodes/edges."""
    topos = []
    zoo_dir = Path(data_dir) / "topology_zoo"
    ms_dir = Path(data_dir) / "microservices" / "MicroDepGraphDataset"

    # Topology Zoo
    if zoo_dir.exists():
        for f in sorted(zoo_dir.glob("*.gml")):
            try:
                G = nx.read_gml(str(f), label="id")
                # Convert multigraph to simple graph
                if isinstance(G, nx.MultiGraph):
                    G = nx.Graph(G)
                if G.number_of_nodes() < 5 or G.number_of_edges() < 4:
                    continue

                degrees = dict(G.degree())
                max_deg = max(degrees.values()) if degrees else 1
                node_types = [classify_zoo_node(G.nodes[n], degrees[n], max_deg) for n in G.nodes()]
                edge_types = [classify_zoo_edge(G.edges[u, v]) for u, v in G.edges()]
                topos.append((G, node_types, edge_types, f.stem))
            except Exception:
                continue

    # Microservices
    if ms_dir.exists():
        for f in sorted(ms_dir.glob("*.graphml")):
            try:
                G = nx.read_graphml(str(f))
                if isinstance(G, nx.MultiDiGraph):
                    G = nx.Graph(G)
                elif isinstance(G, nx.DiGraph):
                    G = nx.Graph(G)
                if G.number_of_nodes() < 3:
                    continue

                node_types = [classify_ms_node(n, G.nodes[n]) for n in G.nodes()]
                edge_types = [EDGE_TYPES["service_dep"]] * G.number_of_edges()
                topos.append((G, node_types, edge_types, f.stem))
            except Exception:
                continue

    return topos


def build_dataset(data_dir: str, scenarios_per_topo: int = 30, seed: int = 42) -> dict:
    """Build complete training dataset from real topologies."""
    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Infrastructure Topology Dataset{C_RESET}")
    print(f"  {C_DIM}{'═' * 55}{C_RESET}\n")

    print(f"  {C_INFO}Loading topologies...{C_RESET}", flush=True)
    t0 = time.time()
    topos = load_all_topologies(data_dir)
    print(f"  {C_TEXT}Loaded {C_BRIGHT}{len(topos)}{C_RESET} topologies in {time.time()-t0:.1f}s", flush=True)

    # Stats
    sizes = [(G.number_of_nodes(), G.number_of_edges()) for G, _, _, _ in topos]
    print(f"  {C_TEXT}Nodes: {C_BRIGHT}{min(s[0] for s in sizes)}-{max(s[0] for s in sizes)}{C_RESET}")
    print(f"  {C_TEXT}Edges: {C_BRIGHT}{min(s[1] for s in sizes)}-{max(s[1] for s in sizes)}{C_RESET}")

    # Generate cascade scenarios
    print(f"\n  {C_INFO}Generating {scenarios_per_topo} scenarios per topology...{C_RESET}", flush=True)
    rng = SeededRandom(seed)
    all_samples = []
    t0 = time.time()

    for i, (G, nt, et, name) in enumerate(topos):
        samples = simulate_cascades(G, nt, et, scenarios_per_topo, rng)
        all_samples.extend(samples)
        if (i + 1) % 50 == 0:
            print(f"    {C_DIM}{i+1}/{len(topos)} topos, {len(all_samples)} samples ({len(all_samples)/(time.time()-t0):.0f}/s){C_RESET}", flush=True)

    elapsed = time.time() - t0
    print(f"  {C_TEXT}Generated {C_BRIGHT}{len(all_samples):,}{C_RESET} samples in {elapsed:.1f}s", flush=True)

    # Pack into tensors
    max_nodes = max(s["n_nodes"] for s in all_samples)
    max_edges = max(s["n_edges"] for s in all_samples)
    n = len(all_samples)

    X = torch.zeros(n, max_nodes, NODE_FEAT_DIM)
    EI = torch.zeros(n, 2, max_edges, dtype=torch.long)
    EA = torch.zeros(n, max_edges, EDGE_FEAT_DIM)
    GT = torch.zeros(n, max_nodes, dtype=torch.long)
    HEALTH = torch.zeros(n, max_nodes)
    MASK = torch.zeros(n, max_nodes)
    N_NODES = torch.zeros(n, dtype=torch.long)
    N_EDGES = torch.zeros(n, dtype=torch.long)

    for i, s in enumerate(all_samples):
        nn_ = s["n_nodes"]
        ne_ = s["n_edges"]
        X[i, :nn_] = torch.tensor(s["x"])
        EI[i, :, :ne_] = torch.tensor(s["edge_index"])
        EA[i, :ne_] = torch.tensor(s["edge_attr"])
        GT[i, :nn_] = torch.tensor(s["states"])
        HEALTH[i, :nn_] = torch.tensor(s["health"])
        MASK[i, :nn_] = 1.0
        N_NODES[i] = nn_
        N_EDGES[i] = ne_

    # Split
    perm = torch.randperm(n)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    dataset = {
        "X": X, "EI": EI, "EA": EA, "GT": GT, "HEALTH": HEALTH,
        "MASK": MASK, "N_NODES": N_NODES, "N_EDGES": N_EDGES,
        "train_idx": perm[:n_train],
        "val_idx": perm[n_train:n_train + n_val],
        "test_idx": perm[n_train + n_val:],
        "node_feat_dim": NODE_FEAT_DIM,
        "edge_feat_dim": EDGE_FEAT_DIM,
        "n_node_types": N_NODE_TYPES,
        "n_edge_types": N_EDGE_TYPES,
        "max_nodes": max_nodes,
        "max_edges": max_edges,
    }

    # Stats
    valid_gt = GT[MASK > 0].long()
    print(f"\n  {C_INFO}State distribution:{C_RESET}")
    for s in range(N_STATES):
        count = (valid_gt == s).sum().item()
        print(f"    {C_DIM}{STATE_NAMES[s]:12s}: {count:8d} ({count/valid_gt.size(0)*100:.1f}%){C_RESET}")

    # Save
    out_path = Path(__file__).parent / "data" / "infra_graphs.pt"
    torch.save(dataset, str(out_path))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n  {C_SUCCESS}{C_BOLD}Saved:{C_RESET} {C_TEXT}{out_path} ({size_mb:.1f} MB){C_RESET}")
    print(f"  {C_TEXT}Samples: {n:,} | Max nodes: {max_nodes} | Max edges: {max_edges}{C_RESET}")
    print()

    return dataset


if __name__ == "__main__":
    data_dir = str(Path(__file__).parent / "data" / "infra_topo")
    build_dataset(data_dir, scenarios_per_topo=30)
