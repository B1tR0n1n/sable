"""
SABLE Telemetry Adapter — Base Interface
==========================================
Defines the contract that all telemetry sources must implement.

A TelemetryAdapter converts raw monitoring data from a specific source
(Prometheus, SNMP, Datadog, etc.) into SABLE's normalized SystemSnapshot
format. The snapshot is then consumed by the health scorer and pillar
encoders to produce model inputs.

The adapter does NOT produce tensor formats directly — that's the job of
the pillar encoders in encode.py. The adapter produces structured Python
objects with normalized metrics that the encoders consume.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Component Types (must match sable_sim and GNN_NODE_TYPES) ────────────

COMPONENT_TYPES = [
    "CORE_SWITCH", "ACCESS_SWITCH", "FIREWALL", "ROUTER", "LOAD_BALANCER",
    "SERVER_PHYSICAL", "SERVER_VIRTUAL", "HYPERVISOR", "STORAGE_ARRAY",
    "STORAGE_TARGET", "VDI_BROKER", "VDI_HOST", "DNS_SERVER", "DHCP_SERVER",
    "DOMAIN_CONTROLLER", "CERTIFICATE_AUTHORITY", "MONITORING_SERVER",
    "WAN_LINK", "INTERNET_GATEWAY", "APPLICATION_SERVICE",
]
COMPONENT_TYPE_INDEX = {t: i for i, t in enumerate(COMPONENT_TYPES)}

# ── Dependency Types ─────────────────────────────────────────────────────

DEPENDENCY_TYPES = [
    "NETWORK_PATH", "POWER_DEPENDENCY", "SERVICE_DEPENDENCY",
    "STORAGE_DEPENDENCY", "AUTHENTICATION_DEPENDENCY", "DNS_DEPENDENCY",
    "HOSTING_DEPENDENCY", "REPLICATION_DEPENDENCY", "MONITORING_DEPENDENCY",
]


class Criticality(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"
    REDUNDANT = "REDUNDANT"


# ── Snapshot Data Structures ─────────────────────────────────────────────


@dataclass
class MetricPoint:
    """A single metric reading from a monitoring source."""
    name: str                    # e.g. "cpu_utilization", "mem_used_pct"
    value: float                 # Current value
    unit: str = ""               # e.g. "percent", "bytes", "ms"
    timestamp: float = 0.0       # Unix epoch


@dataclass
class NodeSnapshot:
    """Normalized view of a single infrastructure component at a point in time.

    This is the adapter's output per node — source-agnostic. The health
    scorer converts raw metrics into health/state, or the adapter can
    pre-compute them if the source provides health natively.
    """
    node_id: str                               # Unique ID (hostname, IP, CMDB CI)
    component_type: str                        # One of COMPONENT_TYPES
    metrics: dict[str, float] = field(default_factory=dict)  # name → value
    timestamp: float = 0.0                     # When this snapshot was taken
    labels: dict[str, str] = field(default_factory=dict)  # Source labels/tags

    # Pre-computed by health scorer (or adapter if source provides native health)
    health: Optional[float] = None             # [0, 1] composite health
    state: Optional[str] = None                # healthy/degraded/failed/unreachable/oscillating
    reachable: bool = True                     # Did we get metrics at all?
    stale_seconds: float = 0.0                 # Time since last successful metric poll


@dataclass
class EdgeSnapshot:
    """A dependency relationship between two nodes."""
    source_id: str
    target_id: str
    dep_type: str                  # One of DEPENDENCY_TYPES
    criticality: Criticality = Criticality.SOFT
    confidence: float = 0.8        # How certain we are this edge exists
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class SystemSnapshot:
    """Complete system state at a point in time.

    This is what the pillar encoders consume to build tensor inputs.
    """
    nodes: dict[str, NodeSnapshot] = field(default_factory=dict)  # node_id → snapshot
    edges: list[EdgeSnapshot] = field(default_factory=list)
    timestamp: float = 0.0
    source: str = ""               # Which adapter produced this

    @property
    def node_ids(self) -> list[str]:
        """Sorted node IDs for deterministic ordering."""
        return sorted(self.nodes.keys())

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


# ── Abstract Adapter ─────────────────────────────────────────────────────


class TelemetryAdapter(ABC):
    """Base class for all telemetry source adapters.

    Subclasses implement poll() to fetch current metrics from their source
    and return a SystemSnapshot. The snapshot contains raw metrics per node
    plus topology edges.

    The health scorer runs after poll() to compute health/state from metrics.
    """

    @abstractmethod
    def poll(self) -> SystemSnapshot:
        """Fetch current system state from the monitoring source.

        Returns a SystemSnapshot with raw metrics per node. Health and state
        fields may be None — the health scorer fills those in.
        """
        ...

    @abstractmethod
    def discover_topology(self) -> list[EdgeSnapshot]:
        """Discover or refresh the dependency graph.

        Called less frequently than poll() — topology changes slowly.
        Returns edges representing dependencies between nodes.
        """
        ...

    def poll_with_health(self, scorer) -> SystemSnapshot:
        """Poll and score in one call. Convenience method."""
        snapshot = self.poll()
        for node in snapshot.nodes.values():
            if node.health is None:
                node.health, node.state = scorer.score(node)
        return snapshot
