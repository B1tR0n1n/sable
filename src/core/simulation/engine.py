"""Main simulation engine — ties together injection, propagation, fog, and decisions."""

from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass, field
from typing import Any

from sable_sim.core.component import ComponentState
from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.state import SystemState, StateChange
from sable_sim.simulation.propagation import PropagationEngine
from sable_sim.simulation.failure_injection import (
    FailureInjector,
    FailureInjection,
    FailureCategory,
)
from sable_sim.simulation.fog import FogOfWar
from sable_sim.simulation.decisions import DecisionGenerator
from sable_sim.utils.random import SeededRandom

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Complete output of a single simulation run."""

    scenario_id: str
    topology: InfrastructureGraph
    initial_state: dict[str, Any]
    failure_injections: list[FailureInjection]
    cascade_trace: list[StateChange]
    final_state: dict[str, dict[str, Any]]
    operator_view: dict[str, Any]
    decision_points: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


class SimulationEngine:
    """Orchestrates failure injection, propagation, fog-of-war, and decision
    generation into a complete simulation run."""

    def __init__(
        self,
        seed: int = 42,
        max_ticks: int = 50,
        soft_impact_factor: float = 0.3,
        degradation_threshold: float = 0.5,
        failure_threshold: float = 0.2,
        dns_cache_ttl: int = 3,
        session_ttl: int = 5,
        monitoring_delay: int = 2,
        monitoring_coverage: float = 0.85,
        decision_interval: int = 3,
    ) -> None:
        self._rng = SeededRandom(seed)
        self._propagation = PropagationEngine(
            soft_impact_factor=soft_impact_factor,
            degradation_threshold=degradation_threshold,
            failure_threshold=failure_threshold,
            dns_cache_ttl=dns_cache_ttl,
            session_ttl=session_ttl,
            max_ticks=max_ticks,
        )
        self._injector = FailureInjector(self._rng)
        self._fog = FogOfWar(
            monitoring_delay=monitoring_delay,
            monitoring_coverage=monitoring_coverage,
            rng=self._rng,
        )
        self._decision_gen = DecisionGenerator(self._rng)
        self._max_ticks = max_ticks
        self._decision_interval = decision_interval

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        graph: InfrastructureGraph,
        failure_categories: list[FailureCategory] | None = None,
        failure_injections: list[FailureInjection] | None = None,
        difficulty: str = "medium",
    ) -> SimulationResult:
        """Run a complete simulation.

        Provide explicit *failure_injections*, or let the engine generate them
        from *failure_categories* and *difficulty*.
        """
        # 1. Deep-copy graph
        sim_graph = graph.copy()
        state = SystemState(sim_graph)

        # 2. Snapshot initial state
        initial = {
            c.id: {"state": str(c.state), "health": c.health}
            for c in sim_graph.get_all_components()
        }

        # 3. Generate or use provided injections
        if failure_injections is None:
            failure_injections = self._injector.generate_failures(
                state, categories=failure_categories, difficulty=difficulty
            )

        # 4. Apply injections at tick 0
        for inj in failure_injections:
            self._injector.inject(state, inj)

        logger.info(
            "Injected %d failures into %d-node topology",
            len(failure_injections),
            sim_graph.node_count,
        )

        # 5. Run propagation
        self._propagation.propagate(state)

        logger.info(
            "Propagation finished at tick %d with %d state changes",
            state.tick,
            len(state.history),
        )

        # 6. Generate decision points
        decision_points = self._decision_gen.generate_decision_points(
            state, interval=self._decision_interval
        )

        # 7. Generate operator view via fog-of-war
        operator_view = self._fog.generate_operator_view(state)

        # 8. Build final state
        final = {
            c.id: {"state": str(c.state), "health": round(c.health, 4)}
            for c in sim_graph.get_all_components()
        }

        # 9. Package result
        return SimulationResult(
            scenario_id=str(uuid.uuid4()),
            topology=sim_graph,
            initial_state=initial,
            failure_injections=failure_injections,
            cascade_trace=list(state.history),
            final_state=final,
            operator_view=operator_view,
            decision_points=[dp.to_dict() for dp in decision_points],
            metadata={
                "node_count": sim_graph.node_count,
                "edge_count": sim_graph.edge_count,
                "total_ticks": state.tick,
                "total_state_changes": len(state.history),
                "difficulty": difficulty,
                "failure_categories": [
                    str(inj.category) for inj in failure_injections
                ],
            },
        )

    def run_batch(
        self,
        graphs: list[InfrastructureGraph],
        categories_list: list[list[FailureCategory]] | None = None,
        difficulty: str = "medium",
    ) -> list[SimulationResult]:
        """Run simulations on multiple topologies."""
        results: list[SimulationResult] = []
        for i, graph in enumerate(graphs):
            cats = categories_list[i] if categories_list else None
            result = self.run(graph, failure_categories=cats, difficulty=difficulty)
            results.append(result)
        return results
