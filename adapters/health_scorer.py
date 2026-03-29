"""
SABLE Health Scorer
=====================
Converts raw telemetry metrics into normalized health [0,1] and state
classification for each node.

Threshold-based with per-type overrides. An MSP can tune thresholds per
customer environment without touching model code.

States: healthy, degraded, failed, unreachable, oscillating
Health: 1.0 = perfect, 0.0 = dead

The scorer is intentionally simple — threshold logic, not ML. The ML
lives in the pillars. The scorer's job is to normalize diverse metrics
into a consistent input format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .base import NodeSnapshot


@dataclass
class ThresholdSet:
    """Thresholds for a single metric. All values in the metric's native unit."""
    degraded: float = 80.0    # Above this → degraded
    failed: float = 95.0      # Above this → failed
    invert: bool = False       # True for metrics where low = bad (e.g. free_memory)
    binary: bool = False       # True for 0/1 metrics (up, healthy). 1=good, 0=bad.


@dataclass
class HealthConfig:
    """Per-environment health scoring configuration.

    Defaults are conservative enterprise baselines. Override per customer.
    """
    # Metric thresholds: metric_name → ThresholdSet
    thresholds: dict[str, ThresholdSet] = field(default_factory=lambda: {
        # CPU
        "cpu_utilization": ThresholdSet(degraded=80, failed=95),
        "cpu_load_1m": ThresholdSet(degraded=0.8, failed=0.95),  # Normalized to core count
        # Memory
        "mem_used_pct": ThresholdSet(degraded=85, failed=95),
        "mem_available_pct": ThresholdSet(degraded=15, failed=5, invert=True),
        # Disk
        "disk_used_pct": ThresholdSet(degraded=85, failed=95),
        "disk_io_util": ThresholdSet(degraded=80, failed=95),
        # Network
        "net_error_rate": ThresholdSet(degraded=1.0, failed=5.0),
        "net_packet_loss_pct": ThresholdSet(degraded=1.0, failed=5.0),
        "net_bandwidth_util": ThresholdSet(degraded=80, failed=95),
        # Latency / response time
        "response_time_ms": ThresholdSet(degraded=500, failed=2000),
        "latency_ms": ThresholdSet(degraded=100, failed=500),
        # Service (binary: 1=good, 0=bad)
        "up": ThresholdSet(binary=True),
        "healthy": ThresholdSet(binary=True),
        # Switch/Router specific
        "interface_errors": ThresholdSet(degraded=10, failed=100),
        "interface_discards": ThresholdSet(degraded=10, failed=100),
        "bgp_state": ThresholdSet(binary=True),  # 1=established
    })

    # How many seconds without metrics before marking unreachable
    stale_threshold: float = 300.0  # 5 minutes

    # Oscillation detection: if state changes more than N times in the window
    oscillation_changes: int = 3
    oscillation_window: float = 300.0  # 5 minutes

    # Metric weights for composite health (higher = more impact)
    metric_weights: dict[str, float] = field(default_factory=lambda: {
        "cpu_utilization": 1.0,
        "mem_used_pct": 1.0,
        "disk_used_pct": 0.8,
        "net_packet_loss_pct": 1.2,
        "response_time_ms": 1.5,
        "up": 2.0,
    })


class HealthScorer:
    """Scores infrastructure nodes based on their telemetry metrics.

    Converts raw metric values into:
      - health: float [0, 1] — composite health score
      - state: one of healthy/degraded/failed/unreachable/oscillating

    State is derived FROM the composite health, not tracked independently
    per metric. This prevents state/health disagreement.
    """

    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()
        # State history for oscillation detection: node_id → list of (timestamp, state)
        self._state_history: dict[str, list[tuple[float, str]]] = {}

    def score(self, node: NodeSnapshot) -> tuple[float, str]:
        """Score a node's health from its current metrics.

        Returns: (health, state)
            health: float [0, 1]
            state: one of healthy/degraded/failed/unreachable/oscillating
        """
        # Unreachable: no metrics or stale
        if not node.reachable or node.stale_seconds > self.config.stale_threshold:
            return 0.0, "unreachable"

        if not node.metrics:
            return 1.0, "healthy"  # No metrics = assume healthy (new node, no data yet)

        # Score each metric individually
        metric_healths = []

        for name, value in node.metrics.items():
            threshold = self.config.thresholds.get(name)
            if threshold is None:
                continue

            h = self._score_metric(value, threshold)
            weight = self.config.metric_weights.get(name, 1.0)
            metric_healths.append((h, weight))

        # Composite health: weighted average
        if metric_healths:
            total_weight = sum(w for _, w in metric_healths)
            health = sum(h * w for h, w in metric_healths) / total_weight
        else:
            health = 1.0

        health = max(0.0, min(1.0, health))

        # Derive state FROM health — single source of truth
        if health >= 0.8:
            state = "healthy"
        elif health >= 0.3:
            state = "degraded"
        else:
            state = "failed"

        # Oscillation detection
        if self._check_oscillation(node.node_id, state, node.timestamp):
            state = "oscillating"
            health = min(health, 0.5)

        return health, state

    def _score_metric(self, value: float, threshold: ThresholdSet) -> float:
        """Score a single metric against its thresholds.

        Returns health contribution [0, 1] with smooth interpolation:
          - Above failed threshold (or below for inverted): 0.0
          - Between degraded and failed: smooth ramp from 0.8 → 0.0
          - Below degraded threshold (or above for inverted): smooth ramp from 1.0 → 0.8
        """
        if threshold.binary:
            return 1.0 if value >= 1.0 else 0.0

        if threshold.invert:
            # Low values are bad (e.g. available memory)
            if value <= threshold.failed:
                return 0.0
            elif value <= threshold.degraded:
                # Smooth ramp: failed → degraded boundary = 0.0 → 0.8
                ratio = (value - threshold.failed) / max(threshold.degraded - threshold.failed, 1e-6)
                return 0.8 * ratio
            else:
                return 1.0
        else:
            # High values are bad (e.g. CPU utilization)
            if value >= threshold.failed:
                return 0.0
            elif value >= threshold.degraded:
                # Smooth ramp: degraded → failed boundary = 0.8 → 0.0
                ratio = (value - threshold.degraded) / max(threshold.failed - threshold.degraded, 1e-6)
                return 0.8 * (1.0 - ratio)
            else:
                return 1.0

    def _check_oscillation(self, node_id: str, current_state: str, timestamp: float) -> bool:
        """Detect if a node is oscillating between states."""
        history = self._state_history.setdefault(node_id, [])

        # Append current
        history.append((timestamp, current_state))

        # Trim to window
        cutoff = timestamp - self.config.oscillation_window
        self._state_history[node_id] = [
            (t, s) for t, s in history if t >= cutoff
        ]
        history = self._state_history[node_id]

        if len(history) < 3:
            return False

        # Count state transitions
        transitions = sum(
            1 for i in range(1, len(history))
            if history[i][1] != history[i - 1][1]
        )

        return transitions >= self.config.oscillation_changes

    def reset_history(self, node_id: Optional[str] = None):
        """Clear oscillation detection state."""
        if node_id:
            self._state_history.pop(node_id, None)
        else:
            self._state_history.clear()
