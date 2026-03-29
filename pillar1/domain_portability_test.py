#!/usr/bin/env python3
"""
Project PARALLAX — Domain Portability Test
=============================================
The definitive test: can a GNN trained on Keith's cognitive graph
reason about infrastructure topology it's never seen?

Generates an infrastructure graph from the SABLE simulator,
maps it through a domain adapter into the GNN's abstract feature space,
and runs inference cold — zero retraining.

If this works, SABLE's domain-agnostic claim is proven.
If it fails, the architecture hardcoded CORTEX assumptions.

Usage:
    python domain_portability_test.py
    python domain_portability_test.py --nodes 200 --device cuda

Requires: ml-env with torch, torch_geometric, numpy, networkx
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from cortex_gnn_model import SableGNN

# ── Terminal Colors ────────────────────────────────────────────────────────

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"

# ── GNN Constants (must match trained model) ───────────────────────────────

# Legacy CORTEX types (preserved for backward compat with CORTEX-domain inference)
GNN_NODE_TYPES_CORTEX = ["observation", "task", "idea", "reference", "person_note", "instruction"]

# Native infrastructure types — 20 component types, no lossy compression
GNN_NODE_TYPES = [
    "CORE_SWITCH", "ACCESS_SWITCH", "FIREWALL", "ROUTER", "LOAD_BALANCER",
    "SERVER_PHYSICAL", "SERVER_VIRTUAL", "HYPERVISOR", "STORAGE_ARRAY",
    "STORAGE_TARGET", "VDI_BROKER", "VDI_HOST", "DNS_SERVER", "DHCP_SERVER",
    "DOMAIN_CONTROLLER", "CERTIFICATE_AUTHORITY", "MONITORING_SERVER",
    "WAN_LINK", "INTERNET_GATEWAY", "APPLICATION_SERVICE",
]

GNN_RELATION_TYPES = ["supports", "contradicts", "elaborates", "depends_on", "caused_by", "related", "supersedes"]
GNN_EMBEDDING_DIM = 1024
GNN_NODE_FEAT_DIM = GNN_EMBEDDING_DIM + len(GNN_NODE_TYPES)  # 1044
GNN_EDGE_FEAT_DIM = len(GNN_RELATION_TYPES) + 1  # 8

# ── Infrastructure Domain ─────────────────────────────────────────────────

COMPONENT_TYPES = [
    "CORE_SWITCH", "ACCESS_SWITCH", "FIREWALL", "ROUTER", "LOAD_BALANCER",
    "SERVER_PHYSICAL", "SERVER_VIRTUAL", "HYPERVISOR", "STORAGE_ARRAY",
    "STORAGE_TARGET", "VDI_BROKER", "VDI_HOST", "DNS_SERVER", "DHCP_SERVER",
    "DOMAIN_CONTROLLER", "CERTIFICATE_AUTHORITY", "MONITORING_SERVER",
    "WAN_LINK", "INTERNET_GATEWAY", "APPLICATION_SERVICE",
]

DEPENDENCY_TYPES = [
    "NETWORK_PATH", "POWER_DEPENDENCY", "SERVICE_DEPENDENCY",
    "STORAGE_DEPENDENCY", "AUTHENTICATION_DEPENDENCY", "DNS_DEPENDENCY",
    "HOSTING_DEPENDENCY", "REPLICATION_DEPENDENCY", "MONITORING_DEPENDENCY",
]

# ── Domain Adapter ─────────────────────────────────────────────────────────
# This is the key piece: maps infrastructure concepts to abstract GNN space.
# The GNN never sees "FIREWALL" or "NETWORK_PATH" — it sees embeddings
# and relation type indices.


class InfrastructureAdapter:
    """
    Maps infrastructure domain → GNN feature space with native 20-type encoding.

    Node mapping:
        20 component types → 1024-dim learned embeddings + 20-dim type one-hot (1044 total)
        Each infrastructure type gets its own one-hot dimension — no lossy compression
        through CORTEX abstract types.

    Edge mapping:
        9 dependency types → 7 GNN relation types:
          - NETWORK_PATH → "related" (general connectivity)
          - POWER_DEPENDENCY → "depends_on" (hard dependency)
          - SERVICE_DEPENDENCY → "depends_on"
          - STORAGE_DEPENDENCY → "depends_on"
          - AUTHENTICATION_DEPENDENCY → "depends_on"
          - DNS_DEPENDENCY → "supports" (enabling service)
          - HOSTING_DEPENDENCY → "elaborates" (host elaborates on VM's existence)
          - REPLICATION_DEPENDENCY → "supports" (redundancy support)
          - MONITORING_DEPENDENCY → "related" (observational, not causal)
    """

    # Direct component type → index (native 20-type one-hot, no compression)
    NODE_TYPE_MAP = {t: i for i, t in enumerate(GNN_NODE_TYPES)}

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

    # Dependency type → GNN relation type index
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

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.type_embeddings = self._generate_type_embeddings()

    def _generate_type_embeddings(self) -> dict[str, np.ndarray]:
        """Generate pseudo-embeddings for infrastructure component types.

        Uses a deterministic scheme: each type gets a base direction in
        1024-dim space, plus noise. Types in the same functional group
        (networking, compute, storage) share similar directions.
        """
        embeddings = {}

        group_centroids = {
            name: self.rng.standard_normal(GNN_EMBEDDING_DIM) * 0.3
            for name in self.FUNCTIONAL_GROUPS
        }

        type_to_group = {}
        for group, types in self.FUNCTIONAL_GROUPS.items():
            for t in types:
                type_to_group[t] = group

        for comp_type in COMPONENT_TYPES:
            group = type_to_group[comp_type]
            base = group_centroids[group].copy()
            noise = self.rng.standard_normal(GNN_EMBEDDING_DIM) * 0.15
            emb = base + noise
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings[comp_type] = emb

        return embeddings

    def adapt_node(self, comp_type: str, health: float = 1.0,
                   properties: dict = None) -> np.ndarray:
        """Convert an infrastructure node to GNN feature vector.

        Returns: 1044-dim vector (1024 embedding + 20 type one-hot)
        """
        emb = self.type_embeddings.get(comp_type, np.zeros(GNN_EMBEDDING_DIM))

        # Modulate embedding by health — degraded nodes have shifted embeddings
        if health < 1.0:
            perturbation = self.rng.standard_normal(GNN_EMBEDDING_DIM) * (1.0 - health) * 0.2
            emb = emb + perturbation
            emb = emb / (np.linalg.norm(emb) + 1e-8)

        # Native 20-type one-hot — full type resolution
        type_one_hot = np.zeros(len(GNN_NODE_TYPES), dtype=np.float32)
        type_idx = self.NODE_TYPE_MAP.get(comp_type, 0)
        type_one_hot[type_idx] = 1.0

        return np.concatenate([emb, type_one_hot]).astype(np.float32)

    def adapt_edge(self, dep_type: str, criticality: str = "SOFT",
                   confidence: float = 0.8) -> tuple[int, np.ndarray]:
        """Convert an infrastructure edge to GNN edge features.

        Returns: (relation_type_index, 8-dim edge feature vector)
        """
        rel_idx = self.EDGE_TYPE_MAP.get(dep_type, 5)  # default: related

        crit_map = {"HARD": 1.0, "SOFT": 0.7, "REDUNDANT": 0.3}
        conf = crit_map.get(criticality, confidence)

        rel_one_hot = [0.0] * len(GNN_RELATION_TYPES)
        rel_one_hot[rel_idx] = 1.0
        edge_feat = rel_one_hot + [conf]

        return rel_idx, np.array(edge_feat, dtype=np.float32)


# ── Infrastructure Graph Generator ────────────────────────────────────────


def generate_infrastructure_graph(
    num_nodes: int = 100,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Generate a realistic infrastructure topology.

    Returns (nodes, edges) where each node/edge is a dict with
    type, health, properties, etc.
    """
    rng = np.random.default_rng(seed)

    # Type distribution matching a real enterprise environment
    type_weights = {
        "CORE_SWITCH": 0.02, "ACCESS_SWITCH": 0.08, "FIREWALL": 0.03,
        "ROUTER": 0.03, "LOAD_BALANCER": 0.02, "SERVER_PHYSICAL": 0.10,
        "SERVER_VIRTUAL": 0.18, "HYPERVISOR": 0.05, "STORAGE_ARRAY": 0.03,
        "STORAGE_TARGET": 0.04, "VDI_BROKER": 0.02, "VDI_HOST": 0.05,
        "DNS_SERVER": 0.03, "DHCP_SERVER": 0.02, "DOMAIN_CONTROLLER": 0.03,
        "CERTIFICATE_AUTHORITY": 0.02, "MONITORING_SERVER": 0.02,
        "WAN_LINK": 0.03, "INTERNET_GATEWAY": 0.02,
        "APPLICATION_SERVICE": 0.18,
    }
    types = list(type_weights.keys())
    probs = np.array(list(type_weights.values()))
    probs = probs / probs.sum()

    # Generate nodes
    nodes = []
    for i in range(num_nodes):
        comp_type = rng.choice(types, p=probs)
        health = 1.0 if rng.random() > 0.15 else rng.uniform(0.2, 0.9)
        nodes.append({
            "id": i,
            "type": comp_type,
            "health": health,
        })

    # Generate edges based on realistic dependency patterns
    edges = []
    # Dependency patterns: which types connect to which, and how
    patterns = [
        # (source_types, target_types, dep_type, criticality, probability)
        (["SERVER_VIRTUAL", "APPLICATION_SERVICE"], ["HYPERVISOR", "SERVER_PHYSICAL"], "HOSTING_DEPENDENCY", "HARD", 0.6),
        (["SERVER_VIRTUAL", "SERVER_PHYSICAL", "APPLICATION_SERVICE"], ["STORAGE_ARRAY", "STORAGE_TARGET"], "STORAGE_DEPENDENCY", "HARD", 0.3),
        (["SERVER_VIRTUAL", "SERVER_PHYSICAL", "APPLICATION_SERVICE"], ["DNS_SERVER"], "DNS_DEPENDENCY", "SOFT", 0.4),
        (["SERVER_VIRTUAL", "SERVER_PHYSICAL", "APPLICATION_SERVICE"], ["DOMAIN_CONTROLLER"], "AUTHENTICATION_DEPENDENCY", "HARD", 0.3),
        (["ACCESS_SWITCH"], ["CORE_SWITCH"], "NETWORK_PATH", "HARD", 0.7),
        (["SERVER_PHYSICAL", "HYPERVISOR"], ["ACCESS_SWITCH", "CORE_SWITCH"], "NETWORK_PATH", "HARD", 0.5),
        (["CORE_SWITCH"], ["FIREWALL", "ROUTER"], "NETWORK_PATH", "HARD", 0.5),
        (["FIREWALL", "ROUTER"], ["INTERNET_GATEWAY"], "NETWORK_PATH", "HARD", 0.4),
        (["INTERNET_GATEWAY"], ["WAN_LINK"], "NETWORK_PATH", "HARD", 0.6),
        (["VDI_HOST"], ["VDI_BROKER"], "SERVICE_DEPENDENCY", "HARD", 0.5),
        (["DOMAIN_CONTROLLER"], ["DOMAIN_CONTROLLER"], "REPLICATION_DEPENDENCY", "SOFT", 0.3),
        (["STORAGE_TARGET"], ["STORAGE_ARRAY"], "STORAGE_DEPENDENCY", "HARD", 0.6),
        (["MONITORING_SERVER"], ["SERVER_PHYSICAL", "CORE_SWITCH", "FIREWALL"], "MONITORING_DEPENDENCY", "REDUNDANT", 0.2),
        (["APPLICATION_SERVICE"], ["LOAD_BALANCER"], "SERVICE_DEPENDENCY", "SOFT", 0.3),
        (["APPLICATION_SERVICE"], ["APPLICATION_SERVICE"], "SERVICE_DEPENDENCY", "SOFT", 0.1),
    ]

    edge_set = set()
    for src_types, tgt_types, dep_type, crit, prob in patterns:
        for i, ni in enumerate(nodes):
            if ni["type"] not in src_types:
                continue
            for j, nj in enumerate(nodes):
                if i == j:
                    continue
                if nj["type"] not in tgt_types:
                    continue
                if (i, j) in edge_set:
                    continue
                if rng.random() < prob:
                    edges.append({
                        "source": i,
                        "target": j,
                        "dep_type": dep_type,
                        "criticality": crit,
                    })
                    edge_set.add((i, j))

    return nodes, edges


# ── Test Runner ────────────────────────────────────────────────────────────


def run_portability_test(
    num_nodes: int = 100,
    checkpoint_path: str = "checkpoints/best_model.pt",
    device: str = "cuda",
    seed: int = 42,
):
    """The test. Train on thoughts, infer on infrastructure."""

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Domain Portability Test{C_RESET}")
    print(f"  {C_DIM}{'═' * 50}{C_RESET}")
    print(f"  {C_TEXT}Trained on: CORTEX knowledge graph (thoughts, beliefs, decisions){C_RESET}")
    print(f"  {C_TEXT}Testing on: Infrastructure topology (servers, switches, dependencies){C_RESET}")
    print(f"  {C_TEXT}Retraining: None. Cold inference.{C_RESET}\n")

    # 1. Load the CORTEX-trained model
    print(f"  {C_INFO}Loading CORTEX-trained model...{C_RESET}")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    mc = ckpt["config"]
    model = SableGNN(
        in_dim=mc["in_dim"], hidden_dim=mc["hidden_dim"],
        edge_dim=mc["edge_dim"], num_layers=mc["num_layers"],
        heads=mc["heads"], dropout=mc["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  {C_DIM}Epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}, "
          f"params {sum(p.numel() for p in model.parameters()):,}{C_RESET}")

    # 2. Generate infrastructure graph
    print(f"\n  {C_INFO}Generating infrastructure topology ({num_nodes} nodes)...{C_RESET}")
    nodes, edges = generate_infrastructure_graph(num_nodes, seed)
    print(f"  {C_TEXT}nodes: {C_BRIGHT}{len(nodes)}{C_RESET}")
    print(f"  {C_TEXT}edges: {C_BRIGHT}{len(edges)}{C_RESET}")

    # Type distribution
    type_counts = {}
    for n in nodes:
        type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1
    dep_counts = {}
    for e in edges:
        dep_counts[e["dep_type"]] = dep_counts.get(e["dep_type"], 0) + 1

    print(f"\n  {C_INFO}Component types:{C_RESET}")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1])[:8]:
        print(f"    {C_DIM}{t:<25s}{C_RESET} {C_TEXT}{c}{C_RESET}")

    print(f"\n  {C_INFO}Dependency types:{C_RESET}")
    for t, c in sorted(dep_counts.items(), key=lambda x: -x[1]):
        print(f"    {C_DIM}{t:<30s}{C_RESET} {C_TEXT}{c}{C_RESET}")

    # 3. Apply domain adapter
    print(f"\n  {C_INFO}Applying domain adapter...{C_RESET}")
    adapter = InfrastructureAdapter(seed=seed)

    # Encode nodes
    node_features = []
    for n in nodes:
        feat = adapter.adapt_node(n["type"], n["health"])
        node_features.append(feat)
    x = torch.tensor(np.array(node_features), dtype=torch.float32)
    print(f"  {C_TEXT}node features: {C_BRIGHT}{x.shape}{C_RESET}")

    # Encode edges — hold out 20% for testing
    rng = np.random.default_rng(seed)
    indices = np.arange(len(edges))
    rng.shuffle(indices)
    split = int(len(edges) * 0.8)
    train_idx = indices[:split]
    test_idx = indices[split:]

    def encode_edges(edge_list, idx_set):
        sources, targets, feats, labels = [], [], [], []
        for i in idx_set:
            e = edge_list[i]
            rel_idx, feat = adapter.adapt_edge(e["dep_type"], e["criticality"])
            sources.append(e["source"])
            targets.append(e["target"])
            feats.append(feat)
            labels.append(rel_idx)
        ei = torch.tensor([sources, targets], dtype=torch.long)
        ea = torch.tensor(np.array(feats), dtype=torch.float32)
        el = torch.tensor(labels, dtype=torch.long)
        return ei, ea, el

    train_ei, train_ea, _ = encode_edges(edges, train_idx)
    test_ei, _, test_labels = encode_edges(edges, test_idx)

    print(f"  {C_TEXT}train edges: {C_BRIGHT}{train_ei.size(1)}{C_RESET}")
    print(f"  {C_TEXT}test edges:  {C_BRIGHT}{test_ei.size(1)}{C_RESET}")

    # Generate negative samples
    neg_ei = negative_sampling(
        torch.cat([train_ei, test_ei], dim=1),
        num_nodes=len(nodes),
        num_neg_samples=test_ei.size(1) * 5,
    )
    print(f"  {C_TEXT}negatives:   {C_BRIGHT}{neg_ei.size(1)}{C_RESET}")

    # 4. Run GNN inference — cold, no retraining
    print(f"\n  {C_INFO}Running GNN inference (cold, zero retraining)...{C_RESET}")
    t0 = time.time()

    with torch.no_grad():
        x_d = x.to(device)
        train_ei_d = train_ei.to(device)
        train_ea_d = train_ea.to(device)
        test_ei_d = test_ei.to(device)
        neg_ei_d = neg_ei.to(device)

        # Encode nodes using training edges for message passing
        node_emb = model.encode(x_d, train_ei_d, train_ea_d)

        # Link prediction
        pos_logits = model.predict_link(node_emb, test_ei_d).cpu()
        neg_logits = model.predict_link(node_emb, neg_ei_d).cpu()
        pos_probs = torch.sigmoid(pos_logits).numpy()
        neg_probs = torch.sigmoid(neg_logits).numpy()

        all_link_scores = np.concatenate([pos_probs, neg_probs])
        all_link_labels = np.concatenate([
            np.ones(len(pos_probs)),
            np.zeros(len(neg_probs)),
        ])

        # Link type prediction
        type_logits = model.predict_link_type(node_emb, test_ei_d)
        type_preds = type_logits.argmax(dim=-1).cpu().numpy()

        # Contradiction detection on all edges
        all_ei_d = torch.cat([train_ei, test_ei], dim=1).to(device)
        contra_logits = model.predict_contradiction(node_emb, all_ei_d)
        contra_probs = torch.sigmoid(contra_logits).cpu().numpy()

    elapsed_ms = (time.time() - t0) * 1000
    print(f"  {C_TEXT}inference time: {C_BRIGHT}{elapsed_ms:.1f}ms{C_RESET}")

    # 5. Compute metrics
    print(f"\n{C_GOLD}{C_BOLD}  Results — Infrastructure Domain (Cold Inference){C_RESET}")
    print(f"  {C_DIM}{'═' * 55}{C_RESET}")

    # Link prediction AUC
    from benchmark import compute_auc, compute_f1_binary, compute_mrr_hits, compute_macro_f1, compute_per_class_f1

    link_auc = compute_auc(all_link_scores, all_link_labels)
    link_f1 = compute_f1_binary(all_link_scores, all_link_labels)
    rr = compute_mrr_hits(pos_probs, neg_probs)

    print(f"\n  {C_INFO}Link Prediction:{C_RESET}")
    print(f"    {C_TEXT}AUC-ROC:  {C_BRIGHT}{link_auc:.4f}{C_RESET}")
    print(f"    {C_TEXT}F1:       {C_BRIGHT}{link_f1:.4f}{C_RESET}")
    print(f"    {C_TEXT}MRR:      {C_BRIGHT}{rr['mrr']:.4f}{C_RESET}")
    print(f"    {C_TEXT}Hits@1:   {C_BRIGHT}{rr['hits_1']:.4f}{C_RESET}")
    print(f"    {C_TEXT}Hits@5:   {C_BRIGHT}{rr['hits_5']:.4f}{C_RESET}")
    print(f"    {C_TEXT}Hits@10:  {C_BRIGHT}{rr['hits_10']:.4f}{C_RESET}")

    # Link type prediction
    test_labels_np = test_labels.numpy()
    type_macro_f1 = compute_macro_f1(type_preds, test_labels_np, len(GNN_RELATION_TYPES))
    type_per_class = compute_per_class_f1(type_preds, test_labels_np, len(GNN_RELATION_TYPES))

    print(f"\n  {C_INFO}Link Type Prediction:{C_RESET}")
    print(f"    {C_TEXT}Macro F1: {C_BRIGHT}{type_macro_f1:.4f}{C_RESET}")
    for i, rt in enumerate(GNN_RELATION_TYPES):
        count = (test_labels_np == i).sum()
        if count > 0:
            bar = "█" * int(type_per_class[i] * 20)
            print(f"    {C_DIM}{rt:15s}{C_RESET} F1={C_TEXT}{type_per_class[i]:.3f}{C_RESET} "
                  f"(n={count:3d}) {C_GOLD}{bar}{C_RESET}")

    # Contradiction: infrastructure graphs shouldn't have many
    # but the model should give low scores to everything
    mean_contra = contra_probs.mean()
    max_contra = contra_probs.max()
    print(f"\n  {C_INFO}Contradiction Scores (should be low — no contradictions in infra):{C_RESET}")
    print(f"    {C_TEXT}mean: {C_BRIGHT}{mean_contra:.4f}{C_RESET}")
    print(f"    {C_TEXT}max:  {C_BRIGHT}{max_contra:.4f}{C_RESET}")

    # 6. Baselines for comparison
    print(f"\n  {C_INFO}Baselines on same infrastructure graph:{C_RESET}")

    # Random baseline
    rng2 = np.random.default_rng(99)
    random_link = rng2.random(len(all_link_scores))
    random_auc = compute_auc(random_link, all_link_labels)
    random_type = rng2.integers(0, len(GNN_RELATION_TYPES), size=len(test_labels_np))
    random_type_f1 = compute_macro_f1(random_type, test_labels_np, len(GNN_RELATION_TYPES))

    # Cosine baseline
    emb_np = x[:, :1024].numpy()
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    emb_norm = emb_np / np.maximum(norms, 1e-8)

    cos_link = []
    test_ei_np = test_ei.numpy()
    for k in range(test_ei_np.shape[1]):
        cos_link.append(float(np.dot(emb_norm[test_ei_np[0, k]], emb_norm[test_ei_np[1, k]])))
    neg_ei_np = neg_ei.numpy()
    for k in range(neg_ei_np.shape[1]):
        cos_link.append(float(np.dot(emb_norm[neg_ei_np[0, k]], emb_norm[neg_ei_np[1, k]])))
    cos_link = np.array(cos_link)
    cos_auc = compute_auc(cos_link, all_link_labels)

    print(f"    {C_DIM}{'Method':<22s} {'Link AUC':>10s} {'Type F1':>10s}{C_RESET}")
    print(f"    {C_DIM}{'─' * 45}{C_RESET}")
    print(f"    {C_TEXT}{'random':<22s} {random_auc:10.4f} {random_type_f1:10.4f}{C_RESET}")
    print(f"    {C_TEXT}{'cosine_similarity':<22s} {cos_auc:10.4f} {'—':>10s}{C_RESET}")
    print(f"    {C_SUCCESS}{C_BOLD}{'sable_gnn (cold)':<22s} {link_auc:10.4f} {type_macro_f1:10.4f}{C_RESET}")

    # 7. Verdict
    print(f"\n  {C_GOLD}{C_BOLD}Verdict:{C_RESET}")

    beats_random_link = link_auc > random_auc + 0.05
    beats_cosine_link = link_auc > cos_auc + 0.03
    beats_random_type = type_macro_f1 > random_type_f1 + 0.03
    low_contradiction = mean_contra < 0.3

    checks = [
        ("Link AUC > random + 0.05", beats_random_link),
        ("Link AUC > cosine + 0.03", beats_cosine_link),
        ("Type F1 > random + 0.03", beats_random_type),
        ("Contradiction scores low (no false alarms)", low_contradiction),
    ]

    passed = 0
    for desc, ok in checks:
        marker = f"{C_SUCCESS}PASS{C_RESET}" if ok else f"{C_DANGER}FAIL{C_RESET}"
        if ok:
            passed += 1
        print(f"    {marker}  {C_TEXT}{desc}{C_RESET}")

    if passed >= 3:
        print(f"\n  {C_SUCCESS}{C_BOLD}DOMAIN PORTABILITY CONFIRMED ({passed}/4){C_RESET}")
        print(f"  {C_TEXT}The GNN trained on cognitive patterns can reason about")
        print(f"  infrastructure topology without retraining.{C_RESET}")
    elif passed >= 2:
        print(f"\n  {C_GOLD}{C_BOLD}PARTIAL PORTABILITY ({passed}/4){C_RESET}")
        print(f"  {C_TEXT}Some transfer, but the adapter layer needs refinement.{C_RESET}")
    else:
        print(f"\n  {C_DANGER}{C_BOLD}PORTABILITY NOT DEMONSTRATED ({passed}/4){C_RESET}")
        print(f"  {C_TEXT}The model learned CORTEX-specific patterns, not abstract structure.{C_RESET}")

    print()
    return {
        "link_auc": link_auc, "link_f1": link_f1,
        "mrr": rr["mrr"], "type_macro_f1": type_macro_f1,
        "mean_contradiction": mean_contra,
        "passed": passed, "total": 4,
    }


def main():
    parser = argparse.ArgumentParser(description="SABLE Domain Portability Test")
    parser.add_argument("--nodes", type=int, default=150)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_portability_test(args.nodes, args.checkpoint, args.device, args.seed)


if __name__ == "__main__":
    main()
