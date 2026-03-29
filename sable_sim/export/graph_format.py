"""Export infrastructure graphs in formats suitable for graph neural networks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import logging

import numpy as np

from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.component import ComponentType, ComponentState
from sable_sim.core.dependency import DependencyType, Criticality

logger = logging.getLogger(__name__)

# Mappings for numerical encoding
COMPONENT_TYPE_INDEX = {t: i for i, t in enumerate(ComponentType)}
STATE_INDEX = {s: i for i, s in enumerate(ComponentState)}
DEP_TYPE_INDEX = {t: i for i, t in enumerate(DependencyType)}
CRIT_INDEX = {c: i for i, c in enumerate(Criticality)}

_N_COMP_TYPES = len(ComponentType)
_N_DEP_TYPES = len(DependencyType)


class GraphFormatExporter:
    """Export infrastructure graphs in formats suitable for GNNs."""

    def export_adjacency(
        self, graph: InfrastructureGraph, output_path: str | Path
    ) -> str:
        """Export as adjacency-format JSON.

        Format::

            {
                "node_features": [[...], ...],
                "edge_index": [[src...], [dst...]],
                "edge_features": [[...], ...],
                "node_labels": [...],
                "node_types": [...]
            }

        Returns the output file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        features, labels, types = self._encode_node_features(graph)
        edge_index, edge_features = self._encode_edge_features(graph)

        data = {
            "node_features": features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "node_labels": labels,
            "node_types": types,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        logger.debug("Exported adjacency graph to %s", output_path)
        return str(output_path)

    def export_pyg(
        self, graph: InfrastructureGraph, output_path: str | Path
    ) -> str:
        """Export as a PyTorch Geometric Data object (.pt file).

        Returns the output file path.
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            logger.warning(
                "torch / torch_geometric not installed — skipping PyG export"
            )
            return ""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        features, labels, _ = self._encode_node_features(graph)
        edge_index, edge_features = self._encode_edge_features(graph)

        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            y=torch.tensor(labels, dtype=torch.long),
        )
        torch.save(data, str(output_path))
        logger.debug("Exported PyG data to %s", output_path)
        return str(output_path)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_node_features(
        self, graph: InfrastructureGraph
    ) -> tuple[list[list[float]], list[int], list[int]]:
        """Encode node features as numerical vectors.

        Returns (features_list, labels_list, types_list).
        """
        components = graph.get_all_components()
        features: list[list[float]] = []
        labels: list[int] = []
        types: list[int] = []

        for comp in components:
            # One-hot component type
            one_hot = [0.0] * _N_COMP_TYPES
            one_hot[COMPONENT_TYPE_INDEX[comp.type]] = 1.0

            # Health
            feat = one_hot + [comp.health]

            # Key normalised properties
            props = comp.properties
            feat.append(props.get("cpu_util", 0.0))
            feat.append(props.get("memory_util", 0.0))
            feat.append(min(props.get("throughput", 0) / 10000, 1.0))
            feat.append(min(props.get("latency_ms", 0) / 100, 1.0))

            features.append(feat)
            labels.append(STATE_INDEX.get(comp.state, 0))
            types.append(COMPONENT_TYPE_INDEX[comp.type])

        return features, labels, types

    def _encode_edge_features(
        self, graph: InfrastructureGraph
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Encode edges in COO format with features.

        Returns (edge_index [[src...], [dst...]], edge_features).
        """
        components = graph.get_all_components()
        id_to_idx = {c.id: i for i, c in enumerate(components)}
        deps = graph.get_all_dependencies()

        sources: list[int] = []
        targets: list[int] = []
        edge_feats: list[list[float]] = []

        for dep in deps:
            src_idx = id_to_idx.get(dep.source_id)
            tgt_idx = id_to_idx.get(dep.target_id)
            if src_idx is None or tgt_idx is None:
                continue
            sources.append(src_idx)
            targets.append(tgt_idx)

            # One-hot dep type
            dt = [0.0] * _N_DEP_TYPES
            dt[DEP_TYPE_INDEX[dep.type]] = 1.0

            # Criticality (ordinal)
            crit_val = {Criticality.HARD: 1.0, Criticality.SOFT: 0.5, Criticality.REDUNDANT: 0.2}
            crit = crit_val.get(dep.criticality, 0.5)

            feat = dt + [crit, float(dep.bandwidth_sensitivity), dep.latency_sensitivity / 100.0]
            edge_feats.append(feat)

        return [sources, targets], edge_feats
