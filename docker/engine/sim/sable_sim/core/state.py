"""Global system state tracking across simulation ticks."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from sable_sim.core.component import Component, ComponentState
from sable_sim.core.graph import InfrastructureGraph


@dataclass
class StateChange:
    """A single state transition recorded during the simulation."""

    tick: int
    component_id: str
    previous_state: ComponentState
    new_state: ComponentState
    previous_health: float
    new_health: float
    cause: str
    cause_component: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "component_id": self.component_id,
            "previous_state": str(self.previous_state),
            "new_state": str(self.new_state),
            "previous_health": round(self.previous_health, 4),
            "new_health": round(self.new_health, 4),
            "cause": self.cause,
            "cause_component": self.cause_component,
        }


class SystemState:
    """Mutable snapshot of the entire infrastructure at a point in time."""

    def __init__(self, graph: InfrastructureGraph) -> None:
        self.graph = graph
        self.tick: int = 0
        self.history: list[StateChange] = []

    # ---- recording -------------------------------------------------------

    def record_change(self, change: StateChange) -> None:
        """Append a state change to the history."""
        self.history.append(change)

    # ---- queries ---------------------------------------------------------

    def get_changes_at_tick(self, tick: int) -> list[StateChange]:
        """All state changes recorded at a specific tick."""
        return [c for c in self.history if c.tick == tick]

    def get_component_history(self, component_id: str) -> list[StateChange]:
        """Ordered list of changes for a single component."""
        return [c for c in self.history if c.component_id == component_id]

    def get_failed_components(self) -> list[Component]:
        """Components currently in FAILED state."""
        return self.graph.get_components_by_state(ComponentState.FAILED)

    def get_degraded_components(self) -> list[Component]:
        """Components currently in DEGRADED state."""
        return self.graph.get_components_by_state(ComponentState.DEGRADED)

    # ---- tick management -------------------------------------------------

    def advance_tick(self) -> int:
        """Increment the tick counter and return the new value."""
        self.tick += 1
        return self.tick

    def has_reached_steady_state(self, lookback: int = 3) -> bool:
        """True if no state changes occurred in the last *lookback* ticks."""
        if self.tick < lookback:
            return False
        for t in range(self.tick - lookback + 1, self.tick + 1):
            if self.get_changes_at_tick(t):
                return False
        return True

    # ---- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "graph": self.graph.to_dict(),
            "history": [c.to_dict() for c in self.history],
        }

    # ---- copying ---------------------------------------------------------

    def copy(self) -> SystemState:
        """Deep copy for branching simulations."""
        new_graph = self.graph.copy()
        new = SystemState(new_graph)
        new.tick = self.tick
        new.history = [
            StateChange(
                tick=c.tick,
                component_id=c.component_id,
                previous_state=c.previous_state,
                new_state=c.new_state,
                previous_health=c.previous_health,
                new_health=c.new_health,
                cause=c.cause,
                cause_component=c.cause_component,
            )
            for c in self.history
        ]
        return new
