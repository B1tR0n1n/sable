"""Decision-space generation — possible operator actions at each point in a cascade."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
import logging

from sable_sim.core.component import Component, ComponentState, ComponentType
from sable_sim.core.dependency import DependencyType, Criticality
from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.state import SystemState, StateChange
from sable_sim.utils.random import SeededRandom

logger = logging.getLogger(__name__)


class DecisionType(StrEnum):
    """Types of operator decisions."""

    RESTART_SERVICE = "RESTART_SERVICE"
    FAILOVER = "FAILOVER"
    ISOLATE = "ISOLATE"
    ROLLBACK_CHANGE = "ROLLBACK_CHANGE"
    ESCALATE_VENDOR = "ESCALATE_VENDOR"
    REDISTRIBUTE_LOAD = "REDISTRIBUTE_LOAD"
    INVESTIGATE = "INVESTIGATE"
    DO_NOTHING = "DO_NOTHING"


@dataclass
class Decision:
    """A single possible operator action."""

    action: DecisionType
    target: str  # component_id
    secondary_target: str | None = None  # e.g. failover destination
    success_probability: float = 0.5
    time_cost: int = 1  # ticks
    side_effects: list[str] = field(default_factory=list)
    information_gain: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": str(self.action),
            "target": self.target,
            "secondary_target": self.secondary_target,
            "success_probability": round(self.success_probability, 3),
            "time_cost": self.time_cost,
            "side_effects": self.side_effects,
            "information_gain": self.information_gain,
            "prerequisites": self.prerequisites,
        }


@dataclass
class DecisionPoint:
    """A moment in the cascade where the operator can act."""

    tick: int
    available_decisions: list[Decision]
    optimal_decision: str  # DecisionType name
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "available_decisions": [d.to_dict() for d in self.available_decisions],
            "optimal_decision": self.optimal_decision,
            "reasoning": self.reasoning,
        }


class DecisionGenerator:
    """Generates decision points at significant moments during a cascade."""

    def __init__(self, rng: SeededRandom) -> None:
        self._rng = rng

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_decision_points(
        self,
        state: SystemState,
        interval: int = 3,
    ) -> list[DecisionPoint]:
        """Generate decision points at ticks where state changes occurred.

        Only produces a point every *interval* ticks at most.
        """
        points: list[DecisionPoint] = []
        last_point_tick = -interval

        for tick in range(0, state.tick + 1):
            changes = state.get_changes_at_tick(tick)
            if not changes:
                continue
            if tick - last_point_tick < interval:
                continue

            decisions = self._generate_decisions(state, tick)
            if not decisions:
                continue

            root_causes = self._get_root_causes(state, tick)
            optimal, reasoning = self._determine_optimal(
                state, decisions, tick, root_causes
            )

            # Refine success probabilities
            for d in decisions:
                d.success_probability = self._calc_success_probability(
                    d, state, root_causes
                )

            points.append(
                DecisionPoint(
                    tick=tick,
                    available_decisions=decisions,
                    optimal_decision=optimal,
                    reasoning=reasoning,
                )
            )
            last_point_tick = tick

        return points

    # ------------------------------------------------------------------
    # Decision generation
    # ------------------------------------------------------------------

    def _generate_decisions(
        self,
        state: SystemState,
        tick: int,
    ) -> list[Decision]:
        """Generate available decisions at a given tick."""
        decisions: list[Decision] = []
        changes = state.get_changes_at_tick(tick)
        affected_ids = {c.component_id for c in changes if c.cause != "monitoring_loss"}
        if not affected_ids:
            return []

        # Pick the most impactful affected component
        target_id = self._pick_primary_target(state, affected_ids)
        comp = state.graph.get_component(target_id)
        if comp is None:
            return []

        # RESTART_SERVICE
        decisions.append(
            Decision(
                action=DecisionType.RESTART_SERVICE,
                target=target_id,
                time_cost=self._rng.randint(3, 5),
                side_effects=[f"Brief downtime during restart of {target_id}"],
            )
        )

        # FAILOVER — available if redundant targets exist
        failover_target = self._find_failover_target(state, comp)
        if failover_target:
            decisions.append(
                Decision(
                    action=DecisionType.FAILOVER,
                    target=target_id,
                    secondary_target=failover_target.id,
                    time_cost=self._rng.randint(1, 3),
                    side_effects=[
                        f"Increased load on {failover_target.id}"
                    ],
                )
            )

        # ISOLATE
        dependents = state.graph.get_dependents(target_id)
        if dependents:
            decisions.append(
                Decision(
                    action=DecisionType.ISOLATE,
                    target=target_id,
                    time_cost=1,
                    side_effects=[
                        f"Dependents of {target_id} will lose connectivity"
                    ],
                )
            )

        # ROLLBACK_CHANGE
        decisions.append(
            Decision(
                action=DecisionType.ROLLBACK_CHANGE,
                target=target_id,
                time_cost=self._rng.randint(2, 5),
                side_effects=["May reintroduce previous issue"],
                success_probability=0.3,
            )
        )

        # ESCALATE_VENDOR
        decisions.append(
            Decision(
                action=DecisionType.ESCALATE_VENDOR,
                target=target_id,
                time_cost=self._rng.randint(10, 20),
                success_probability=0.7,
            )
        )

        # REDISTRIBUTE_LOAD — if peers exist
        peers = state.graph.get_components_by_type(comp.type)
        healthy_peers = [
            p
            for p in peers
            if p.id != target_id and p.state == ComponentState.HEALTHY
        ]
        if healthy_peers:
            decisions.append(
                Decision(
                    action=DecisionType.REDISTRIBUTE_LOAD,
                    target=target_id,
                    secondary_target=healthy_peers[0].id,
                    time_cost=self._rng.randint(3, 5),
                    side_effects=["Configuration complexity"],
                )
            )

        # INVESTIGATE — always available
        decisions.append(
            Decision(
                action=DecisionType.INVESTIGATE,
                target=target_id,
                time_cost=self._rng.randint(2, 3),
                information_gain=[
                    f"Reveals detailed metrics/logs for {target_id}"
                ],
                success_probability=0.9,
            )
        )

        # DO_NOTHING — always available
        decisions.append(
            Decision(
                action=DecisionType.DO_NOTHING,
                target=target_id,
                time_cost=0,
                side_effects=["Cascade may worsen"],
                success_probability=0.2,
            )
        )

        return decisions

    # ------------------------------------------------------------------
    # Optimal decision logic
    # ------------------------------------------------------------------

    def _determine_optimal(
        self,
        state: SystemState,
        decisions: list[Decision],
        tick: int,
        root_causes: list[StateChange],
    ) -> tuple[str, str]:
        """Return (optimal_action_name, reasoning)."""
        if not root_causes:
            return (
                str(DecisionType.INVESTIGATE),
                "Root cause is unknown; investigation needed before action.",
            )

        root = root_causes[0]
        root_comp = state.graph.get_component(root.component_id)

        # Check if it's hardware
        is_hardware = False
        if root_comp:
            from sable_sim.core.component import FAILURE_MODES

            for fm in FAILURE_MODES.get(root_comp.type, []):
                if fm.is_hardware:
                    is_hardware = True
                    break

        # Is cascade actively spreading?
        recent_changes = state.get_changes_at_tick(tick)
        cascade_spreading = len(recent_changes) >= 3

        # Check if failover is available
        has_failover = any(
            d.action == DecisionType.FAILOVER for d in decisions
        )

        if cascade_spreading:
            return (
                str(DecisionType.ISOLATE),
                f"Cascade is actively spreading ({len(recent_changes)} components affected this tick). "
                f"Isolate {root.component_id} to contain blast radius.",
            )

        if is_hardware and has_failover:
            return (
                str(DecisionType.FAILOVER),
                f"Root cause is hardware failure on {root.component_id}. "
                f"Failover to redundant component while awaiting hardware replacement.",
            )

        if is_hardware:
            return (
                str(DecisionType.ESCALATE_VENDOR),
                f"Root cause is hardware failure on {root.component_id}. "
                f"No failover available — escalate to vendor for replacement.",
            )

        # Software issue
        if root_comp and root_comp.state == ComponentState.FAILED:
            return (
                str(DecisionType.RESTART_SERVICE),
                f"Software failure on {root.component_id}. "
                f"Service restart is the fastest remediation path.",
            )

        # Degradation without clear action → investigate
        return (
            str(DecisionType.INVESTIGATE),
            f"Component {root.component_id} degraded. "
            f"Investigate to determine if the issue is transient or requires action.",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calc_success_probability(
        self,
        decision: Decision,
        state: SystemState,
        root_causes: list[StateChange],
    ) -> float:
        """Calculate success probability based on whether the decision addresses root cause."""
        base = decision.success_probability
        if not root_causes:
            return base

        root = root_causes[0]
        root_comp = state.graph.get_component(root.component_id)
        addresses_root = decision.target == root.component_id

        match decision.action:
            case DecisionType.RESTART_SERVICE:
                is_hw = self._is_hardware_cause(root_comp)
                if is_hw:
                    return 0.1
                elif addresses_root:
                    return 0.8
                else:
                    return 0.3
            case DecisionType.FAILOVER:
                return 0.85 if addresses_root else 0.5
            case DecisionType.ISOLATE:
                return 0.7  # always somewhat effective
            case DecisionType.ESCALATE_VENDOR:
                return 0.7 if self._is_hardware_cause(root_comp) else 0.3
            case DecisionType.INVESTIGATE:
                return 0.9
            case DecisionType.DO_NOTHING:
                return 0.15
            case _:
                return base

    def _is_hardware_cause(self, comp: Component | None) -> bool:
        if comp is None:
            return False
        from sable_sim.core.component import FAILURE_MODES

        modes = FAILURE_MODES.get(comp.type, [])
        return any(m.is_hardware for m in modes)

    def _get_root_causes(
        self, state: SystemState, tick: int
    ) -> list[StateChange]:
        """Find the original injected failures."""
        return [
            c
            for c in state.history
            if c.cause == "injected_failure" and c.tick <= tick
        ]

    def _pick_primary_target(
        self, state: SystemState, affected_ids: set[str]
    ) -> str:
        """Pick the most impactful component from affected set."""
        best_id = ""
        best_score = -1
        for cid in affected_ids:
            dependents = state.graph.get_dependents(cid)
            score = len(dependents)
            if score > best_score:
                best_score = score
                best_id = cid
        return best_id or next(iter(affected_ids))

    def _find_failover_target(
        self, state: SystemState, comp: Component
    ) -> Component | None:
        """Find a healthy component of the same type for failover."""
        peers = state.graph.get_components_by_type(comp.type)
        for p in peers:
            if p.id != comp.id and p.state == ComponentState.HEALTHY:
                return p
        return None
