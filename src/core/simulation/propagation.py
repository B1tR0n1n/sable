"""Cascade propagation engine — the heart of the simulator."""

from __future__ import annotations

import logging
from typing import Any

from sable_sim.core.component import Component, ComponentState, ComponentType
from sable_sim.core.dependency import Dependency, DependencyType, Criticality
from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.state import SystemState, StateChange

logger = logging.getLogger(__name__)


class PropagationEngine:
    """Propagates failures through infrastructure dependency graphs.

    Runs in discrete ticks.  At each tick the engine examines every component
    that changed state in the *previous* tick and evaluates the impact on its
    dependents, recording new state changes.  The loop continues until steady
    state is reached or *max_ticks* is exhausted.
    """

    def __init__(
        self,
        soft_impact_factor: float = 0.3,
        degradation_threshold: float = 0.5,
        failure_threshold: float = 0.2,
        dns_cache_ttl: int = 3,
        session_ttl: int = 5,
        max_ticks: int = 50,
    ) -> None:
        self.soft_impact_factor = soft_impact_factor
        self.degradation_threshold = degradation_threshold
        self.failure_threshold = failure_threshold
        self.dns_cache_ttl = dns_cache_ttl
        self.session_ttl = session_ttl
        self.max_ticks = max_ticks

        # Delayed-effect tracking
        self._pending_dns: dict[str, int] = {}   # component_id → expiry tick
        self._pending_auth: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propagate(self, state: SystemState) -> SystemState:
        """Run the propagation loop until steady-state or max_ticks.

        Mutates *state* in place and returns it.
        """
        self._pending_dns.clear()
        self._pending_auth.clear()

        for _ in range(self.max_ticks):
            current_tick = state.tick
            changes_this_tick: list[StateChange] = []
            changed_ids: set[str] = set()

            # Components that changed in the previous tick
            prev_changes = state.get_changes_at_tick(current_tick)
            changed_components = {
                c.component_id: state.graph.get_component(c.component_id)
                for c in prev_changes
            }

            for cid, comp in changed_components.items():
                if comp is None:
                    continue
                for dependent in state.graph.get_dependents(cid):
                    if dependent.id in changed_ids:
                        continue  # already processed this tick
                    # Edge is dependent→cid ("dependent depends on cid")
                    dep = state.graph.get_dependency(dependent.id, cid)
                    if dep is None:
                        continue
                    sc = self._evaluate_impact(state, comp, dependent, dep, current_tick)
                    if sc is not None:
                        changes_this_tick.append(sc)
                        changed_ids.add(dependent.id)

            # Process delayed effects
            delayed = self._process_pending_delays(state, current_tick)
            for sc in delayed:
                if sc.component_id not in changed_ids:
                    changes_this_tick.append(sc)
                    changed_ids.add(sc.component_id)

            # Advance tick
            new_tick = state.advance_tick()

            if not changes_this_tick:
                # Check for steady state
                if state.has_reached_steady_state(lookback=2):
                    logger.debug("Steady state reached at tick %d", new_tick)
                    break
                continue

            # Record all changes with the new tick number
            for sc in changes_this_tick:
                sc.tick = new_tick
                state.record_change(sc)

            logger.debug(
                "Tick %d: %d new state changes", new_tick, len(changes_this_tick)
            )

        return state

    # ------------------------------------------------------------------
    # Impact evaluation
    # ------------------------------------------------------------------

    def _evaluate_impact(
        self,
        state: SystemState,
        changed: Component,
        dependent: Component,
        dep: Dependency,
        current_tick: int,
    ) -> StateChange | None:
        """Determine the cascading effect of *changed* on *dependent*.

        Returns a StateChange if the dependent's state worsens, else None.
        """
        # Already at worst state
        if dependent.state == ComponentState.FAILED:
            return None

        prev_state = dependent.state
        prev_health = dependent.health

        # --- special dependency-type rules first --------------------------

        if dep.type == DependencyType.MONITORING_DEPENDENCY:
            # Monitoring loss doesn't change real state, but we record it
            # so the fog-of-war system can use it.
            if changed.state in (ComponentState.FAILED, ComponentState.UNREACHABLE):
                return StateChange(
                    tick=current_tick,
                    component_id=dependent.id,
                    previous_state=prev_state,
                    new_state=prev_state,  # state unchanged
                    previous_health=prev_health,
                    new_health=prev_health,
                    cause="monitoring_loss",
                    cause_component=changed.id,
                )
            return None

        if dep.type == DependencyType.DNS_DEPENDENCY:
            if changed.state == ComponentState.FAILED:
                # Register delayed effect
                if dependent.id not in self._pending_dns:
                    self._pending_dns[dependent.id] = current_tick + self.dns_cache_ttl
                    logger.debug(
                        "DNS dependency delayed for %s until tick %d",
                        dependent.id,
                        current_tick + self.dns_cache_ttl,
                    )
            return None  # actual effect applied via _process_pending_delays

        if dep.type == DependencyType.AUTHENTICATION_DEPENDENCY:
            if changed.state == ComponentState.FAILED:
                if dependent.id not in self._pending_auth:
                    self._pending_auth[dependent.id] = current_tick + self.session_ttl
                    logger.debug(
                        "Auth dependency delayed for %s until tick %d",
                        dependent.id,
                        current_tick + self.session_ttl,
                    )
            return None

        if dep.type == DependencyType.NETWORK_PATH:
            if changed.state == ComponentState.FAILED:
                # Check if there's an alternate path
                if state.graph.has_alternate_path(
                    dependent.id,
                    changed.id,
                    excluded={changed.id},
                ):
                    logger.debug(
                        "Alternate path exists for %s around %s",
                        dependent.id,
                        changed.id,
                    )
                    return None
                # No alternate path — unreachable
                dependent.state = ComponentState.UNREACHABLE
                dependent.health = 0.0
                return StateChange(
                    tick=current_tick,
                    component_id=dependent.id,
                    previous_state=prev_state,
                    new_state=ComponentState.UNREACHABLE,
                    previous_health=prev_health,
                    new_health=0.0,
                    cause="network_unreachable",
                    cause_component=changed.id,
                )

        if dep.type == DependencyType.HOSTING_DEPENDENCY:
            # Always hard — host dies, everything on it dies
            if changed.state == ComponentState.FAILED:
                dependent.state = ComponentState.FAILED
                dependent.health = 0.0
                return StateChange(
                    tick=current_tick,
                    component_id=dependent.id,
                    previous_state=prev_state,
                    new_state=ComponentState.FAILED,
                    previous_health=prev_health,
                    new_health=0.0,
                    cause="host_failure",
                    cause_component=changed.id,
                )
            elif changed.state == ComponentState.DEGRADED:
                return self._apply_degradation(
                    dependent, prev_state, prev_health, changed.id, current_tick, 0.4
                )
            return None

        # --- criticality-based rules -------------------------------------

        match dep.criticality:
            case Criticality.HARD:
                return self._apply_hard(
                    state, changed, dependent, dep, prev_state, prev_health, current_tick
                )
            case Criticality.SOFT:
                return self._apply_soft(
                    changed, dependent, prev_state, prev_health, current_tick
                )
            case Criticality.REDUNDANT:
                return self._apply_redundant(
                    state, changed, dependent, dep, prev_state, prev_health, current_tick
                )

        return None

    # ------------------------------------------------------------------
    # Criticality handlers
    # ------------------------------------------------------------------

    def _apply_hard(
        self,
        state: SystemState,
        changed: Component,
        dependent: Component,
        dep: Dependency,
        prev_state: ComponentState,
        prev_health: float,
        tick: int,
    ) -> StateChange | None:
        if changed.state == ComponentState.FAILED:
            dependent.state = ComponentState.FAILED
            dependent.health = 0.0
        elif changed.state == ComponentState.DEGRADED:
            return self._apply_degradation(
                dependent, prev_state, prev_health, changed.id, tick, 0.4
            )
        else:
            return None

        if dependent.state == prev_state and dependent.health == prev_health:
            return None

        return StateChange(
            tick=tick,
            component_id=dependent.id,
            previous_state=prev_state,
            new_state=dependent.state,
            previous_health=prev_health,
            new_health=dependent.health,
            cause="hard_dependency_cascade",
            cause_component=changed.id,
        )

    def _apply_soft(
        self,
        changed: Component,
        dependent: Component,
        prev_state: ComponentState,
        prev_health: float,
        tick: int,
    ) -> StateChange | None:
        if changed.state == ComponentState.FAILED:
            return self._apply_degradation(
                dependent, prev_state, prev_health, changed.id, tick, 0.5
            )
        elif changed.state == ComponentState.DEGRADED:
            return self._apply_degradation(
                dependent,
                prev_state,
                prev_health,
                changed.id,
                tick,
                self.soft_impact_factor,
            )
        return None

    def _apply_redundant(
        self,
        state: SystemState,
        changed: Component,
        dependent: Component,
        dep: Dependency,
        prev_state: ComponentState,
        prev_health: float,
        tick: int,
    ) -> StateChange | None:
        healthy, total = self._count_healthy_redundant_targets(
            state, dependent.id, dep.type
        )
        if healthy == 0:
            # No remaining healthy targets — treat as HARD
            return self._apply_hard(
                state, changed, dependent, dep, prev_state, prev_health, tick
            )
        # Proportional degradation
        impact = self.soft_impact_factor * (1.0 - healthy / max(total, 1))
        if impact <= 0:
            return None
        return self._apply_degradation(
            dependent, prev_state, prev_health, changed.id, tick, impact
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_degradation(
        self,
        dependent: Component,
        prev_state: ComponentState,
        prev_health: float,
        cause_id: str,
        tick: int,
        impact: float,
    ) -> StateChange | None:
        """Reduce health and possibly change state.  Only worsens — never heals."""
        new_health = max(0.0, dependent.health - impact)
        if new_health >= prev_health:
            return None  # don't improve

        dependent.health = new_health
        if dependent.health <= self.failure_threshold:
            dependent.state = ComponentState.FAILED
        elif dependent.health <= self.degradation_threshold:
            dependent.state = ComponentState.DEGRADED

        # Only record if something actually changed
        if dependent.state == prev_state and abs(dependent.health - prev_health) < 1e-6:
            return None

        return StateChange(
            tick=tick,
            component_id=dependent.id,
            previous_state=prev_state,
            new_state=dependent.state,
            previous_health=prev_health,
            new_health=dependent.health,
            cause="cascade_degradation",
            cause_component=cause_id,
        )

    def _count_healthy_redundant_targets(
        self,
        state: SystemState,
        component_id: str,
        dep_type: DependencyType,
    ) -> tuple[int, int]:
        """Return (healthy_count, total_count) of sources of *dep_type* for *component_id*."""
        pairs = state.graph.get_dependencies_of_type(component_id, dep_type)
        total = len(pairs)
        healthy = sum(
            1 for comp, _ in pairs if comp.state == ComponentState.HEALTHY
        )
        return healthy, total

    def _process_pending_delays(
        self, state: SystemState, tick: int
    ) -> list[StateChange]:
        """Apply any DNS / auth delays whose TTL has expired."""
        changes: list[StateChange] = []

        # DNS
        expired_dns = [
            cid for cid, expiry in self._pending_dns.items() if tick >= expiry
        ]
        for cid in expired_dns:
            comp = state.graph.get_component(cid)
            if comp is None or comp.state == ComponentState.FAILED:
                del self._pending_dns[cid]
                continue
            prev_state = comp.state
            prev_health = comp.health
            # Check if any DNS source is still healthy
            dns_pairs = state.graph.get_dependencies_of_type(cid, DependencyType.DNS_DEPENDENCY)
            any_healthy = any(c.state == ComponentState.HEALTHY for c, _ in dns_pairs)
            if not any_healthy:
                comp.state = ComponentState.DEGRADED
                comp.health = max(0.0, comp.health - 0.4)
                if comp.health <= self.failure_threshold:
                    comp.state = ComponentState.FAILED
                changes.append(
                    StateChange(
                        tick=tick,
                        component_id=cid,
                        previous_state=prev_state,
                        new_state=comp.state,
                        previous_health=prev_health,
                        new_health=comp.health,
                        cause="dns_cache_expired",
                        cause_component=None,
                    )
                )
            del self._pending_dns[cid]

        # Auth
        expired_auth = [
            cid for cid, expiry in self._pending_auth.items() if tick >= expiry
        ]
        for cid in expired_auth:
            comp = state.graph.get_component(cid)
            if comp is None or comp.state == ComponentState.FAILED:
                del self._pending_auth[cid]
                continue
            prev_state = comp.state
            prev_health = comp.health
            auth_pairs = state.graph.get_dependencies_of_type(
                cid, DependencyType.AUTHENTICATION_DEPENDENCY
            )
            any_healthy = any(c.state == ComponentState.HEALTHY for c, _ in auth_pairs)
            if not any_healthy:
                comp.state = ComponentState.DEGRADED
                comp.health = max(0.0, comp.health - 0.5)
                if comp.health <= self.failure_threshold:
                    comp.state = ComponentState.FAILED
                changes.append(
                    StateChange(
                        tick=tick,
                        component_id=cid,
                        previous_state=prev_state,
                        new_state=comp.state,
                        previous_health=prev_health,
                        new_health=comp.health,
                        cause="session_expired",
                        cause_component=None,
                    )
                )
            del self._pending_auth[cid]

        return changes
