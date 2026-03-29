"""
Project PARALLAX — Canonical State Definitions
=================================================
Single source of truth for classification state space.
EVERY file in the codebase imports from here.

The sim has 4 ComponentStates (healthy/degraded/failed/unreachable).
The classification system has 5 classes — the 5th (oscillating) is a
temporal pattern detected by the chain, not a sim state.
"""

from sable_sim.core.component import ComponentState

# Classification state count — used by all heads, losses, metrics
N_STATES = 5

# Human-readable names indexed by class ID
STATE_NAMES = ["healthy", "degraded", "failed", "unreachable", "oscillating"]

# Map sim states → class IDs (oscillating is detected, not mapped)
STATE_MAP = {
    ComponentState.HEALTHY: 0,
    ComponentState.DEGRADED: 1,
    ComponentState.FAILED: 2,
    ComponentState.UNREACHABLE: 3,
}

# Oscillating class index
OSCILLATING_CLASS = 4
