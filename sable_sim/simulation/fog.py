"""Fog-of-war / partial-observability layer.

Generates the "operator view" — what an operator can actually see during
an incident, which is incomplete and delayed compared to ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import logging

from sable_sim.core.component import Component, ComponentState, ComponentType
from sable_sim.core.dependency import DependencyType
from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.state import SystemState, StateChange
from sable_sim.utils.random import SeededRandom

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """A single monitoring observation at a point in time."""

    tick: int
    component_id: str
    observed_state: str  # healthy / degraded / failed / unknown
    metrics: dict[str, float] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "component_id": self.component_id,
            "observed_state": self.observed_state,
            "metrics": self.metrics,
            "alerts": list(self.alerts),
        }


class FogOfWar:
    """Generates a partial-observability operator view of the infrastructure."""

    def __init__(
        self,
        monitoring_delay: int = 2,
        monitoring_coverage: float = 0.85,
        polling_interval: int = 2,
        false_positive_rate: float = 0.05,
        metric_noise_stddev: float = 0.05,
        rng: SeededRandom | None = None,
    ) -> None:
        self.monitoring_delay = monitoring_delay
        self.monitoring_coverage = monitoring_coverage
        self.polling_interval = polling_interval
        self.false_positive_rate = false_positive_rate
        self.metric_noise_stddev = metric_noise_stddev
        self._rng = rng or SeededRandom()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_operator_view(self, state: SystemState) -> dict[str, Any]:
        """Build the operator view dict from ground-truth *state*."""
        monitored, unmonitored = self._determine_monitored_components(state)

        observations: list[dict] = []
        false_positives: list[dict] = []

        # Walk through ticks and produce observations at polling boundaries
        for tick in range(0, state.tick + 1):
            if tick % self.polling_interval != 0:
                continue

            # The operator sees state as it was monitoring_delay ticks ago
            visible_tick = max(0, tick - self.monitoring_delay)

            for cid in monitored:
                comp = state.graph.get_component(cid)
                if comp is None:
                    continue

                # Determine observed state at visible_tick
                obs_state = self._state_at_tick(state, cid, visible_tick)

                # Reachability check
                if not self._is_reachable(state, cid, visible_tick):
                    obs_state = "unknown"

                # Metrics with noise
                metrics = self._add_metric_noise(dict(comp.properties))

                # Alerts
                alerts = self._generate_alerts_for(comp, obs_state)

                observations.append(
                    Observation(
                        tick=tick,
                        component_id=cid,
                        observed_state=obs_state,
                        metrics=metrics,
                        alerts=alerts,
                    ).to_dict()
                )

        # False positives
        false_positives = self._generate_false_positives(state)

        return {
            "observable_components": monitored,
            "unobservable_components": unmonitored,
            "observations": observations,
            "false_positives": false_positives,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _determine_monitored_components(
        self, state: SystemState
    ) -> tuple[list[str], list[str]]:
        """Return (monitored, unmonitored) component ID lists."""
        all_ids: list[str] = [c.id for c in state.graph.get_all_components()]
        monitored: list[str] = []
        unmonitored: list[str] = []

        # Monitoring servers are always observable
        mon_servers = state.graph.get_components_by_type(ComponentType.MONITORING_SERVER)
        mon_ids = {m.id for m in mon_servers}

        # If all monitoring servers are failed, coverage drops drastically
        any_mon_alive = any(
            m.state not in (ComponentState.FAILED, ComponentState.UNREACHABLE)
            for m in mon_servers
        )
        effective_coverage = self.monitoring_coverage if any_mon_alive else 0.15

        for cid in all_ids:
            if cid in mon_ids:
                monitored.append(cid)
                continue
            # Check if this component has a monitoring dependency to a failed server
            comp = state.graph.get_component(cid)
            if comp is None:
                continue
            mon_deps = state.graph.get_dependencies_of_type(
                cid, DependencyType.MONITORING_DEPENDENCY
            )
            if mon_deps:
                # If all monitoring sources are down, unmonitored
                if all(
                    m.state in (ComponentState.FAILED, ComponentState.UNREACHABLE)
                    for m, _ in mon_deps
                ):
                    unmonitored.append(cid)
                    continue

            if self._rng.random() < effective_coverage:
                monitored.append(cid)
            else:
                unmonitored.append(cid)

        return monitored, unmonitored

    def _is_reachable(
        self, state: SystemState, component_id: str, tick: int
    ) -> bool:
        """Check if a component is network-reachable from monitoring infra."""
        comp = state.graph.get_component(component_id)
        if comp is None:
            return False
        # Simple heuristic: component is unreachable if it or its
        # network-path sources are failed at this tick
        net_deps = state.graph.get_dependencies_of_type(
            component_id, DependencyType.NETWORK_PATH
        )
        if not net_deps:
            return True  # no network deps, assume reachable
        return any(
            c.state not in (ComponentState.FAILED, ComponentState.UNREACHABLE)
            for c, _ in net_deps
        )

    def _state_at_tick(
        self, state: SystemState, component_id: str, tick: int
    ) -> str:
        """Reconstruct what state a component was in at a given tick."""
        # Walk backwards through history
        current_state = "healthy"
        for change in state.history:
            if change.component_id != component_id:
                continue
            if change.tick > tick:
                break
            current_state = str(change.new_state)
        return current_state

    def _add_metric_noise(
        self, metrics: dict[str, float]
    ) -> dict[str, float]:
        """Add Gaussian noise to numeric metric values."""
        noisy: dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                noise = self._rng.gauss(0, self.metric_noise_stddev * max(abs(v), 1))
                noisy[k] = round(v + noise, 4)
            else:
                noisy[k] = v
        return noisy

    def _generate_false_positives(
        self, state: SystemState
    ) -> list[dict[str, Any]]:
        """Generate spurious alerts that don't correspond to real problems."""
        fps: list[dict[str, Any]] = []
        healthy = state.graph.get_components_by_state(ComponentState.HEALTHY)
        templates = [
            "Brief connectivity loss on {id}",
            "High CPU spike on {id}",
            "SNMP timeout on {id}",
            "Brief latency spike on {id}",
            "Interface flap detected on {id}",
        ]
        for comp in healthy:
            if self._rng.random() < self.false_positive_rate:
                tick = self._rng.randint(0, max(1, state.tick))
                msg = self._rng.choice(templates).format(id=comp.id)
                fps.append({"tick": tick, "component_id": comp.id, "alert": msg})
        return fps

    def _generate_alerts_for(
        self, comp: Component, observed_state: str
    ) -> list[str]:
        """Generate realistic alert strings based on component state."""
        if observed_state == "healthy":
            return []
        alerts: list[str] = []
        if observed_state in ("failed", "unreachable"):
            alerts.append(f"CRITICAL: {comp.id} is {observed_state}")
        elif observed_state == "degraded":
            alerts.append(f"WARNING: {comp.id} performance degraded")
            # Add type-specific alerts
            match comp.type:
                case ComponentType.STORAGE_ARRAY | ComponentType.STORAGE_TARGET:
                    alerts.append(f"Storage latency critical on {comp.id}")
                case ComponentType.SERVER_PHYSICAL | ComponentType.SERVER_VIRTUAL:
                    alerts.append(f"High resource utilization on {comp.id}")
                case ComponentType.CORE_SWITCH | ComponentType.ACCESS_SWITCH:
                    alerts.append(f"High CPU on {comp.id}")
                case ComponentType.FIREWALL:
                    alerts.append(f"Session table near capacity on {comp.id}")
                case ComponentType.WAN_LINK:
                    alerts.append(f"Link degradation on {comp.id}")
                case _:
                    alerts.append(f"Performance degraded on {comp.id}")
        elif observed_state == "unknown":
            alerts.append(f"WARNING: {comp.id} unreachable by monitoring")
        return alerts
