"""SABLE -> Finding bridge (PLAN.md Phase 2).

    SableClient      thin client over SABLE's routes and /ws
    to_finding       pure mapping of (tick, recommendations) to a Finding
    FindingStore     dedup, change notifications, JSON persistence
    FindingEmitter   ties them together: ticks in, Findings out
    ground_truth     the health scorer's reading beside the model's, per node
    router/create_app  GET /findings, GET /findings/{id}, WS /findings/stream
"""

from .api import create_app, router  # noqa: F401
from .client import SableClient, SableError  # noqa: F401
from .emitter import FindingEmitter  # noqa: F401
from .ground_truth import attach_ground_truth, ground_truth_states  # noqa: F401
from .mapper import (  # noqa: F401
    MappingError,
    confidence_for,
    dedup_key_for,
    detection_mode_for,
    node_info_from_recs,
    severity_for,
    to_finding,
)
from .store import FindingStore  # noqa: F401

__all__ = [
    "SableClient", "SableError",
    "to_finding", "dedup_key_for", "confidence_for", "detection_mode_for", "severity_for",
    "node_info_from_recs", "MappingError",
    "FindingStore", "FindingEmitter",
    "ground_truth_states", "attach_ground_truth",
    "router", "create_app",
]
