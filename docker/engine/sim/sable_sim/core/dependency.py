"""Dependency types and edge definitions for the infrastructure graph."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DependencyType(StrEnum):
    """Types of dependencies between infrastructure components."""

    NETWORK_PATH = "NETWORK_PATH"
    POWER_DEPENDENCY = "POWER_DEPENDENCY"
    SERVICE_DEPENDENCY = "SERVICE_DEPENDENCY"
    STORAGE_DEPENDENCY = "STORAGE_DEPENDENCY"
    AUTHENTICATION_DEPENDENCY = "AUTHENTICATION_DEPENDENCY"
    DNS_DEPENDENCY = "DNS_DEPENDENCY"
    HOSTING_DEPENDENCY = "HOSTING_DEPENDENCY"
    REPLICATION_DEPENDENCY = "REPLICATION_DEPENDENCY"
    MONITORING_DEPENDENCY = "MONITORING_DEPENDENCY"


class Criticality(StrEnum):
    """How critical a dependency is to the dependent component."""

    HARD = "HARD"          # Source failure → dependent failure
    SOFT = "SOFT"          # Source failure → dependent degraded
    REDUNDANT = "REDUNDANT"  # Tolerable if other sources remain


@dataclass
class Dependency:
    """A directed edge in the infrastructure graph."""

    source_id: str
    target_id: str
    type: DependencyType
    criticality: Criticality
    bandwidth_sensitivity: bool = False
    latency_sensitivity: float = 0.0  # max ms; 0 = not sensitive

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": str(self.type),
            "criticality": str(self.criticality),
            "bandwidth_sensitivity": self.bandwidth_sensitivity,
            "latency_sensitivity": self.latency_sensitivity,
        }
