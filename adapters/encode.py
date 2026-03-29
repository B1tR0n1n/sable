"""
SABLE Pillar Encoder
======================
Converts a SystemSnapshot (from any telemetry adapter) into the tensor
formats each SABLE pillar expects.

GNN (Pillar 1):   x: [N, 1044], edge_index: [2, E], edge_attr: [E, 8]
POMDP (Pillar 2): belief_state: dict[node_id → 8-dim vector]
Mamba (Pillar 3): x_input: [1, 2, max_nodes * 26]

This module is the bridge between real telemetry and the model. It imports
the InfrastructureAdapter (for GNN embeddings) and uses sable_sim's encoding
format for Mamba compatibility.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import (
    SystemSnapshot, NodeSnapshot,
    COMPONENT_TYPES, COMPONENT_TYPE_INDEX,
    Criticality,
)

# ── Constants ─────────────────────────────────────────────────────────────

# GNN constants (1044-dim node features)
GNN_EMBEDDING_DIM = 1024
GNN_N_NODE_TYPES = len(COMPONENT_TYPES)  # 20
GNN_NODE_FEAT_DIM = GNN_EMBEDDING_DIM + GNN_N_NODE_TYPES  # 1044
GNN_RELATION_TYPES = [
    "supports", "contradicts", "elaborates", "depends_on",
    "caused_by", "related", "supersedes",
]
GNN_EDGE_FEAT_DIM = len(GNN_RELATION_TYPES) + 1  # 8

# Mamba constants (26-dim per node, from generate_temporal_data.py)
N_STATES = 5  # healthy, degraded, failed, unreachable, oscillating
N_COMP_TYPES = len(COMPONENT_TYPES)  # 20
MAMBA_NODE_FEAT_DIM = 1 + N_STATES + N_COMP_TYPES  # 26
MAMBA_MAX_NODES = 40

# POMDP constants
POMDP_DIM = 8  # [P_h, P_d, P_f, P_u, confidence, obs_age, contradiction, centrality]

# State name → index
STATE_INDEX = {
    "healthy": 0, "degraded": 1, "failed": 2,
    "unreachable": 3, "oscillating": 4,
}


# ── Encoded Outputs ──────────────────────────────────────────────────────


@dataclass
class GNNInput:
    """Tensor-ready input for Pillar 1 (GNN)."""
    x: np.ndarray              # [N, 1044] node features
    edge_index: np.ndarray     # [2, E] source/target indices
    edge_attr: np.ndarray      # [E, 8] edge features
    node_ids: list[str]        # Ordered node IDs matching x rows
    n_nodes: int
    n_edges: int


@dataclass
class POMDPInput:
    """Belief state input for Pillar 2 (POMDP)."""
    beliefs: dict[str, np.ndarray]  # node_id → [8] belief vector
    node_ids: list[str]


@dataclass
class MambaInput:
    """Temporal input for Pillar 3 (Mamba)."""
    x_input: np.ndarray        # [1, 2, max_nodes * 26] — two ticks
    node_ids: list[str]        # Ordered node IDs (up to max_nodes)
    n_nodes: int


@dataclass
class PillarInputs:
    """All three pillar inputs from a single snapshot pair."""
    gnn: GNNInput
    pomdp: POMDPInput
    mamba: MambaInput
    node_ids: list[str]


# ── Encoder ──────────────────────────────────────────────────────────────


class PillarEncoder:
    """Encodes SystemSnapshots into pillar tensor inputs.

    Requires two consecutive snapshots for Mamba (temporal delta).
    GNN and POMDP only need the latest snapshot.

    The GNN adapter is lazy-loaded and adds pillar paths only at load time,
    keeping module-level imports clean.
    """

    def __init__(self, seed: int = 42):
        self._gnn_adapter = None
        self._seed = seed

    @property
    def gnn_adapter(self):
        if self._gnn_adapter is None:
            import sys
            from pathlib import Path
            _root = Path(__file__).parent.parent
            for p in [str(_root / "pillar1"), str(_root / "pillar3")]:
                if p not in sys.path:
                    sys.path.insert(0, p)
            from domain_portability_test import InfrastructureAdapter
            self._gnn_adapter = InfrastructureAdapter(seed=self._seed)
        return self._gnn_adapter

    def encode(
        self,
        current: SystemSnapshot,
        previous: Optional[SystemSnapshot] = None,
    ) -> PillarInputs:
        """Encode a snapshot pair into all three pillar inputs.

        Args:
            current:  Latest system snapshot (required)
            previous: Previous snapshot for temporal delta (optional;
                      if None, Mamba gets duplicate ticks)
        """
        node_ids = sorted(current.node_ids)

        gnn = self.encode_gnn(current, node_ids)
        pomdp = self.encode_pomdp(current, node_ids)
        mamba = self.encode_mamba(current, previous, node_ids)

        return PillarInputs(gnn=gnn, pomdp=pomdp, mamba=mamba, node_ids=node_ids)

    def encode_gnn(self, snapshot: SystemSnapshot, node_ids: list[str]) -> GNNInput:
        """Encode snapshot into GNN input tensors.

        Node features: 1044-dim (1024 embedding + 20 type one-hot)
        Edge features: 8-dim (7 relation one-hot + confidence)

        Health perturbation is deterministic per (node_id, component_type, health)
        to ensure stable embeddings across poll cycles for the same node state.
        """
        n = len(node_ids)
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        x = np.zeros((n, GNN_NODE_FEAT_DIM), dtype=np.float32)
        for i, nid in enumerate(node_ids):
            node = snapshot.nodes.get(nid)
            if node is None:
                continue
            health = node.health if node.health is not None else 1.0
            x[i] = self._stable_adapt_node(nid, node.component_type, health)

        # Edges
        sources, targets, edge_feats = [], [], []
        for edge in snapshot.edges:
            si = node_id_to_idx.get(edge.source_id)
            ti = node_id_to_idx.get(edge.target_id)
            if si is None or ti is None:
                continue

            _, feat = self.gnn_adapter.adapt_edge(
                edge.dep_type, edge.criticality.value, edge.confidence,
            )
            sources.append(si)
            targets.append(ti)
            edge_feats.append(feat)

            # Bidirectional
            sources.append(ti)
            targets.append(si)
            edge_feats.append(feat)

        if sources:
            edge_index = np.array([sources, targets], dtype=np.int64)
            edge_attr = np.array(edge_feats, dtype=np.float32)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, GNN_EDGE_FEAT_DIM), dtype=np.float32)

        return GNNInput(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            node_ids=node_ids, n_nodes=n, n_edges=edge_index.shape[1],
        )

    def _stable_adapt_node(self, node_id: str, comp_type: str, health: float) -> np.ndarray:
        """Produce a deterministic 1044-dim node feature vector.

        Unlike InfrastructureAdapter.adapt_node() which advances a shared RNG
        (non-deterministic across calls), this method seeds the perturbation
        from the node_id so the same node at the same health always produces
        the same embedding. Eliminates noise injection between poll cycles.
        """
        adapter = self.gnn_adapter
        emb = adapter.type_embeddings.get(comp_type, np.zeros(GNN_EMBEDDING_DIM))

        if health < 1.0:
            # Deterministic perturbation seeded from node identity
            seed = int(hashlib.md5(node_id.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            perturbation = rng.standard_normal(GNN_EMBEDDING_DIM) * (1.0 - health) * 0.2
            emb = emb + perturbation
            emb = emb / (np.linalg.norm(emb) + 1e-8)

        # Native 20-type one-hot
        type_one_hot = np.zeros(GNN_N_NODE_TYPES, dtype=np.float32)
        type_idx = adapter.NODE_TYPE_MAP.get(comp_type, 0)
        type_one_hot[type_idx] = 1.0

        return np.concatenate([emb, type_one_hot]).astype(np.float32)

    def encode_pomdp(self, snapshot: SystemSnapshot, node_ids: list[str]) -> POMDPInput:
        """Encode snapshot into POMDP belief vectors.

        Per-node: [P_healthy, P_degraded, P_failed, P_unreachable,
                   confidence, observation_age, contradiction, hub_centrality]

        The state probability distribution is derived from health value,
        spreading mass across states based on how close health is to
        state boundaries. This gives the POMDP genuine uncertainty to
        reason about rather than near-one-hot vectors.
        """
        # Compute degree centrality for hub_centrality
        degree: dict[str, int] = {}
        for edge in snapshot.edges:
            degree[edge.source_id] = degree.get(edge.source_id, 0) + 1
            degree[edge.target_id] = degree.get(edge.target_id, 0) + 1
        max_degree = max(degree.values()) if degree else 1

        beliefs = {}
        for nid in node_ids:
            node = snapshot.nodes.get(nid)
            if node is None:
                beliefs[nid] = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
                continue

            state = node.state or "healthy"
            health = node.health if node.health is not None else 1.0

            probs = self._health_to_belief(health, state)

            confidence = 1.0 if node.reachable else 0.2
            obs_age = min(node.stale_seconds / 300.0, 1.0)  # Normalize to [0,1]
            contradiction = 0.0  # Filled by fusion layer when topology conflicts with observations
            centrality = degree.get(nid, 0) / max(max_degree, 1)

            beliefs[nid] = np.array([
                probs[0], probs[1], probs[2], probs[3],
                confidence, obs_age, contradiction, centrality,
            ], dtype=np.float32)

        return POMDPInput(beliefs=beliefs, node_ids=node_ids)

    def _health_to_belief(self, health: float, state: str) -> np.ndarray:
        """Convert health scalar + state label into a 4-dim probability distribution.

        Maps health onto state boundaries with smooth transitions:
          health >= 0.8  → mostly healthy
          0.3 ≤ health < 0.8 → degraded with spread toward healthy/failed
          health < 0.3  → mostly failed
          unreachable   → high P(unreachable)
          oscillating   → split between healthy and degraded

        Returns: [P_healthy, P_degraded, P_failed, P_unreachable]
        """
        probs = np.zeros(4, dtype=np.float32)

        if state == "unreachable":
            probs[3] = 0.85
            probs[2] = 0.10  # Might be failed, not just unreachable
            probs[1] = 0.05
            return probs

        if state == "oscillating":
            probs[0] = 0.25
            probs[1] = 0.60
            probs[2] = 0.15
            return probs

        # Health-based distribution with smooth transitions
        if health >= 0.8:
            # Mostly healthy, small degraded tail
            t = (health - 0.8) / 0.2  # 0 at boundary, 1 at perfect
            probs[0] = 0.7 + 0.25 * t
            probs[1] = 0.2 - 0.15 * t
            probs[2] = 0.1 - 0.1 * t
        elif health >= 0.3:
            # Degraded zone — spread between healthy, degraded, failed
            t = (health - 0.3) / 0.5  # 0 at failed boundary, 1 at healthy boundary
            probs[0] = 0.1 + 0.6 * t
            probs[1] = 0.5 + 0.1 * t - 0.3 * t  # peaks in middle
            probs[2] = 0.4 - 0.4 * t
        else:
            # Mostly failed
            t = health / 0.3  # 0 at dead, 1 at degraded boundary
            probs[0] = 0.05 * t
            probs[1] = 0.1 * t
            probs[2] = 0.9 - 0.05 * t

        # Normalize (should already sum to ~1 but clamp for safety)
        total = probs.sum()
        if total > 0:
            probs /= total

        return probs

    def encode_mamba(
        self,
        current: SystemSnapshot,
        previous: Optional[SystemSnapshot],
        node_ids: list[str],
    ) -> MambaInput:
        """Encode two snapshots into Mamba temporal input.

        Output: [1, 2, max_nodes * 26]
        Per-node features (26-dim): health (1) + state one-hot (5) + type one-hot (20)
        """
        active_ids = node_ids[:MAMBA_MAX_NODES]
        n = len(active_ids)

        state_dim = MAMBA_MAX_NODES * MAMBA_NODE_FEAT_DIM
        x_input = np.zeros((1, 2, state_dim), dtype=np.float32)

        # Encode tick 0 (previous or duplicate of current)
        src_prev = previous if previous is not None else current
        x_input[0, 0, :n * MAMBA_NODE_FEAT_DIM] = self._encode_mamba_tick(src_prev, active_ids)

        # Encode tick 1 (current)
        x_input[0, 1, :n * MAMBA_NODE_FEAT_DIM] = self._encode_mamba_tick(current, active_ids)

        return MambaInput(x_input=x_input, node_ids=active_ids, n_nodes=n)

    def _encode_mamba_tick(self, snapshot: SystemSnapshot, node_ids: list[str]) -> np.ndarray:
        """Encode one tick of system state for Mamba.

        Returns: (n_nodes * 26,) flat vector
        """
        features = []
        for nid in node_ids:
            node = snapshot.nodes.get(nid)
            if node is None:
                features.extend([0.0] * MAMBA_NODE_FEAT_DIM)
                continue

            health = node.health if node.health is not None else 1.0
            feat = [health]

            # State one-hot (5 states)
            state_oh = [0.0] * N_STATES
            state_idx = STATE_INDEX.get(node.state or "healthy", 0)
            state_oh[state_idx] = 1.0
            feat.extend(state_oh)

            # Type one-hot (20 types)
            type_oh = [0.0] * N_COMP_TYPES
            type_idx = COMPONENT_TYPE_INDEX.get(node.component_type, 0)
            type_oh[type_idx] = 1.0
            feat.extend(type_oh)

            features.extend(feat)

        return np.array(features, dtype=np.float32)
