"""Failure types, injection logic, and scenario generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
import logging

from sable_sim.core.component import (
    Component,
    ComponentState,
    ComponentType,
    FAILURE_MODES,
    FailureMode,
)
from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.state import SystemState, StateChange
from sable_sim.utils.random import SeededRandom

logger = logging.getLogger(__name__)


class FailureCategory(StrEnum):
    """Categories of failure scenarios."""

    SINGLE_COMPONENT_FAILURE = "SINGLE_COMPONENT_FAILURE"
    CASCADING_OVERLOAD = "CASCADING_OVERLOAD"
    SILENT_DEGRADATION = "SILENT_DEGRADATION"
    NETWORK_PARTITION = "NETWORK_PARTITION"
    DEPENDENCY_CHAIN = "DEPENDENCY_CHAIN"
    CORRELATED_FAILURE = "CORRELATED_FAILURE"
    INTERMITTENT_FAILURE = "INTERMITTENT_FAILURE"
    CONFIGURATION_DRIFT = "CONFIGURATION_DRIFT"


@dataclass
class FailureInjection:
    """A failure to inject into the simulation."""

    tick: int
    component_id: str
    failure_mode: str
    parameters: dict[str, Any] = field(default_factory=dict)
    category: FailureCategory = FailureCategory.SINGLE_COMPONENT_FAILURE

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "tick": self.tick,
            "component_id": self.component_id,
            "failure_mode": self.failure_mode,
            "parameters": self.parameters,
            "category": str(self.category),
        }


# Mapping from category to preferred target component types
_CATEGORY_TARGETS: dict[FailureCategory, list[ComponentType]] = {
    FailureCategory.SINGLE_COMPONENT_FAILURE: list(ComponentType),
    FailureCategory.CASCADING_OVERLOAD: [
        ComponentType.CORE_SWITCH,
        ComponentType.LOAD_BALANCER,
        ComponentType.ACCESS_SWITCH,
        ComponentType.FIREWALL,
    ],
    FailureCategory.SILENT_DEGRADATION: [
        ComponentType.STORAGE_ARRAY,
        ComponentType.STORAGE_TARGET,
        ComponentType.WAN_LINK,
        ComponentType.SERVER_PHYSICAL,
    ],
    FailureCategory.NETWORK_PARTITION: [
        ComponentType.WAN_LINK,
        ComponentType.CORE_SWITCH,
        ComponentType.FIREWALL,
        ComponentType.ROUTER,
    ],
    FailureCategory.DEPENDENCY_CHAIN: [
        ComponentType.STORAGE_ARRAY,
        ComponentType.HYPERVISOR,
        ComponentType.DNS_SERVER,
        ComponentType.DOMAIN_CONTROLLER,
    ],
    FailureCategory.CORRELATED_FAILURE: list(ComponentType),
    FailureCategory.INTERMITTENT_FAILURE: list(ComponentType),
    FailureCategory.CONFIGURATION_DRIFT: [
        ComponentType.FIREWALL,
        ComponentType.DNS_SERVER,
        ComponentType.ROUTER,
        ComponentType.ACCESS_SWITCH,
    ],
}


class FailureInjector:
    """Generates and applies failure injections to the simulation."""

    def __init__(self, rng: SeededRandom):
        self._rng = rng

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject(
        self, state: SystemState, injection: FailureInjection
    ) -> list[StateChange]:
        """Apply a failure injection to the system state.

        Returns the list of immediate state changes caused by the injection.
        """
        component = state.graph.get_component(injection.component_id)
        if component is None:
            logger.warning("Injection target %s not found", injection.component_id)
            return []

        prev_state = component.state
        prev_health = component.health

        # Determine new state/health based on failure mode characteristics
        fm = self._find_failure_mode(component.type, injection.failure_mode)
        is_hardware = fm.is_hardware if fm else False
        severity = fm.severity if fm else 0.8

        if is_hardware or severity >= 0.9:
            component.state = ComponentState.FAILED
            component.health = 0.0
        elif severity >= 0.5:
            component.state = ComponentState.DEGRADED
            component.health = max(0.0, 1.0 - severity)
        else:
            # Subtle degradation
            component.health = max(0.0, component.health - severity)
            if component.health < 0.5:
                component.state = ComponentState.DEGRADED

        # Apply any parameter overrides
        if "health" in injection.parameters:
            component.health = injection.parameters["health"]
        if "state" in injection.parameters:
            component.state = ComponentState(injection.parameters["state"])

        if component.state == prev_state and component.health == prev_health:
            return []

        change = StateChange(
            tick=injection.tick,
            component_id=injection.component_id,
            previous_state=prev_state,
            new_state=component.state,
            previous_health=prev_health,
            new_health=component.health,
            cause="injected_failure",
            cause_component=None,
        )
        state.record_change(change)
        logger.debug(
            "Injected %s on %s: %s→%s (health %.2f→%.2f)",
            injection.failure_mode,
            injection.component_id,
            prev_state,
            component.state,
            prev_health,
            component.health,
        )
        return [change]

    def generate_failures(
        self,
        state: SystemState,
        categories: list[FailureCategory] | None = None,
        difficulty: str = "medium",
    ) -> list[FailureInjection]:
        """Generate failure injections based on categories and difficulty.

        Args:
            state: Current system state.
            categories: Failure categories to use. If None, picked randomly.
            difficulty: One of 'easy', 'medium', 'hard'.

        Returns:
            List of FailureInjection objects ready to apply.
        """
        if categories is None:
            n_cats = {"easy": 1, "medium": self._rng.randint(1, 2), "hard": self._rng.randint(2, 3)}
            count = n_cats.get(difficulty, 1)
            categories = self._rng.sample(list(FailureCategory), min(count, len(FailureCategory)))

        injections: list[FailureInjection] = []
        for cat in categories:
            cat_injections = self._generate_for_category(state, cat, difficulty)
            injections.extend(cat_injections)

        logger.info(
            "Generated %d failure injections across %d categories (difficulty=%s)",
            len(injections),
            len(categories),
            difficulty,
        )
        return injections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_for_category(
        self,
        state: SystemState,
        category: FailureCategory,
        difficulty: str,
    ) -> list[FailureInjection]:
        """Generate injections for a single failure category."""
        match category:
            case FailureCategory.CORRELATED_FAILURE:
                return self._gen_correlated(state, difficulty)
            case FailureCategory.INTERMITTENT_FAILURE:
                return self._gen_intermittent(state, difficulty)
            case _:
                target = self._select_target(state, category)
                if target is None:
                    return []
                fm = self._select_failure_mode(target, category)
                params: dict[str, Any] = {}
                if category == FailureCategory.SILENT_DEGRADATION:
                    params["health"] = self._rng.uniform(0.3, 0.5)
                    params["state"] = ComponentState.DEGRADED
                elif category == FailureCategory.CONFIGURATION_DRIFT:
                    params["state"] = ComponentState.DEGRADED
                    params["health"] = self._rng.uniform(0.4, 0.7)
                return [
                    FailureInjection(
                        tick=0,
                        component_id=target.id,
                        failure_mode=fm.name if fm else "hardware_failure",
                        parameters=params,
                        category=category,
                    )
                ]

    def _gen_correlated(
        self, state: SystemState, difficulty: str
    ) -> list[FailureInjection]:
        """Generate correlated failures (multiple components of same type)."""
        all_types = list({c.type for c in state.graph.get_all_components()})
        if not all_types:
            return []
        chosen_type = self._rng.choice(all_types)
        candidates = state.graph.get_components_by_type(chosen_type)
        count = min(len(candidates), {"easy": 2, "medium": 2, "hard": 3}.get(difficulty, 2))
        if count < 2:
            count = min(len(candidates), 2)
        targets = self._rng.sample(list(candidates), min(count, len(candidates)))

        correlation_id = f"corr-{self._rng.randint(1000, 9999)}"
        injections: list[FailureInjection] = []
        for i, t in enumerate(targets):
            fm = self._select_failure_mode(t, FailureCategory.CORRELATED_FAILURE)
            injections.append(
                FailureInjection(
                    tick=i,  # Staggered
                    component_id=t.id,
                    failure_mode=fm.name if fm else "hardware_failure",
                    parameters={"correlation_id": correlation_id},
                    category=FailureCategory.CORRELATED_FAILURE,
                )
            )
        return injections

    def _gen_intermittent(
        self, state: SystemState, difficulty: str
    ) -> list[FailureInjection]:
        """Generate intermittent (flapping) failure."""
        target = self._select_target(state, FailureCategory.INTERMITTENT_FAILURE)
        if target is None:
            return []
        fm = self._select_failure_mode(target, FailureCategory.INTERMITTENT_FAILURE)
        flap_interval = {"easy": 5, "medium": 3, "hard": 2}.get(difficulty, 3)
        return [
            FailureInjection(
                tick=0,
                component_id=target.id,
                failure_mode=fm.name if fm else "hardware_failure",
                parameters={"flap_interval": flap_interval, "intermittent": True},
                category=FailureCategory.INTERMITTENT_FAILURE,
            )
        ]

    def _select_target(
        self,
        state: SystemState,
        category: FailureCategory,
    ) -> Component | None:
        """Select a target component weighted by blast radius and category fit."""
        preferred_types = _CATEGORY_TARGETS.get(category, list(ComponentType))
        # Get candidates of preferred types first
        candidates: list[Component] = []
        for ct in preferred_types:
            candidates.extend(state.graph.get_components_by_type(ct))
        # Fallback to all if none found
        if not candidates:
            candidates = state.graph.get_all_components()
        if not candidates:
            return None

        # Weight by blast radius
        weights = [max(1, self._get_blast_radius(state, c.id)) for c in candidates]
        return self._rng.weighted_choice(candidates, weights)

    def _select_failure_mode(
        self,
        component: Component,
        category: FailureCategory,
    ) -> FailureMode | None:
        """Select an appropriate failure mode for the component and category."""
        modes = FAILURE_MODES.get(component.type, [])
        if not modes:
            return None

        # For silent degradation, prefer non-hardware, lower severity
        if category == FailureCategory.SILENT_DEGRADATION:
            soft = [m for m in modes if not m.is_hardware and m.severity < 0.7]
            if soft:
                return self._rng.choice(soft)

        # For configuration drift, prefer misconfig modes
        if category == FailureCategory.CONFIGURATION_DRIFT:
            config = [m for m in modes if "misconfig" in m.name or "config" in m.name]
            if config:
                return self._rng.choice(config)

        return self._rng.choice(modes)

    def _get_blast_radius(self, state: SystemState, component_id: str) -> int:
        """Estimate how many components would be affected if this one fails."""
        visited: set[str] = set()
        queue = [component_id]
        while queue:
            cid = queue.pop(0)
            if cid in visited:
                continue
            visited.add(cid)
            for dep in state.graph.get_dependents(cid):
                if dep.id not in visited:
                    queue.append(dep.id)
        return len(visited) - 1  # Exclude self

    def _find_failure_mode(
        self, comp_type: ComponentType, mode_name: str
    ) -> FailureMode | None:
        """Look up a FailureMode by name for a component type."""
        for fm in FAILURE_MODES.get(comp_type, []):
            if fm.name == mode_name:
                return fm
        return None
