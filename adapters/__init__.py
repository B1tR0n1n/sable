"""
SABLE Telemetry Adapters
=========================
Converts real infrastructure monitoring data into SABLE's pillar input formats.

Architecture:
    [Prometheus/SNMP/etc] → TelemetryAdapter → health_scorer → pillar inputs
                            graph_builder ──────────────────→ GNN edge_index

Pillar input formats:
    GNN (Pillar 1):   1044-dim node features (1024 embedding + 20 type one-hot)
                      8-dim edge features (7 relation one-hot + confidence)
    POMDP (Pillar 2): 8-dim belief vector per node
    Mamba (Pillar 3): (B, 2, max_nodes * 26) temporal state tensor
"""

from .base import TelemetryAdapter, NodeSnapshot, EdgeSnapshot, SystemSnapshot, Criticality
from .health_scorer import HealthScorer, HealthConfig
from .encode import PillarEncoder
