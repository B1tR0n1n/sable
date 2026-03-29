#!/usr/bin/env python3
"""
Project PARALLAX — Pillar 2: POMCP Solver
============================================
Partially Observable Monte Carlo Planning for infrastructure diagnostics.

Domain-agnostic core: operates on abstract state spaces, actions,
observations, and information gain. The infrastructure adapter maps
domain concepts into this abstract space.

The solver answers: "Given what I can see (partial observations with noise),
what should I check next to maximize information gain toward the root cause?"

Uses the existing sable_sim propagation engine as the transition model
and fog-of-war as the observation model. Pure CPU — V-Cache optimized
tree search on the 9950X3D.

Usage:
    python pomcp.py                                    # Run demo scenario
    python pomcp.py --rollouts 1000 --depth 15         # More rollouts
    python pomcp.py --scenario cascade                 # Specific scenario

Requires: numpy, networkx (no GPU, no torch)
"""

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections import defaultdict
from copy import deepcopy

import numpy as np

# Add sable_sim to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sable_sim.core.component import Component, ComponentState, ComponentType
from sable_sim.core.dependency import Dependency, DependencyType, Criticality
from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.state import SystemState, StateChange
from sable_sim.simulation.propagation import PropagationEngine
from sable_sim.simulation.fog import FogOfWar, Observation

# ── Terminal Colors ────────────────────────────────────────────────────────

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


# ── Abstract Action Space ─────────────────────────────────────────────────
# Domain-agnostic: the POMDP sees action types, not "ping" or "traceroute"


@dataclass
class DiagnosticAction:
    """An action the agent can take to gather information or intervene."""
    name: str
    action_type: str  # "observe", "test", "intervene"
    target_id: str    # Component ID to act on
    cost: float = 1.0  # Time/resource cost
    reveals: list[str] = field(default_factory=list)  # What info this reveals

    def __hash__(self):
        return hash((self.name, self.target_id))

    def __eq__(self, other):
        return self.name == other.name and self.target_id == other.target_id


@dataclass
class ActionResult:
    """What happens when an action is executed."""
    action: DiagnosticAction
    observations: dict[str, str]  # component_id → observed_state
    metrics: dict[str, dict[str, float]]  # component_id → metric dict
    alerts: list[str] = field(default_factory=list)
    information_gained: float = 0.0


# ── Belief State ──────────────────────────────────────────────────────────


class BeliefState:
    """Probability distribution over possible world states.

    Each component has a belief vector: P(healthy), P(degraded), P(failed), P(unreachable).
    The fog-of-war means we don't know the true state of every component.
    Actions reduce uncertainty by revealing information.
    """

    STATES = ["healthy", "degraded", "failed", "unreachable"]
    STATE_IDX = {s: i for i, s in enumerate(STATES)}

    def __init__(self, component_ids: list[str]):
        self.component_ids = list(component_ids)
        self.n_components = len(component_ids)
        # Initialize: assume everything is healthy (prior)
        self.beliefs = {}
        for cid in component_ids:
            self.beliefs[cid] = np.array([0.85, 0.10, 0.03, 0.02])  # Prior

        # Track what we've observed
        self.observed: dict[str, str] = {}  # component_id → last observed state
        self.observation_history: dict[str, list[str]] = {}  # component_id → [obs1, obs2, ...]
        self.observation_age: dict[str, int] = {}  # how many steps since observed
        self.actions_taken: list[str] = []
        self.root_cause_candidates: dict[str, float] = {}  # component_id → score
        self.identified_root_causes: list[str] = []  # confirmed root causes

    def update_from_observation(self, component_id: str, observed_state: str,
                                 confidence: float = 0.9):
        """Bayesian update with consistency-based confidence adjustment.

        Multiple agreeing observations increase confidence.
        An observation contradicting prior observations gets discounted.
        An observation contradicting dependency structure gets discounted.
        """
        if component_id not in self.beliefs:
            return

        # Track observation history for this component
        if component_id not in self.observation_history:
            self.observation_history[component_id] = []
        history = self.observation_history[component_id]
        history.append(observed_state)

        # Consistency-based confidence adjustment
        effective_confidence = confidence
        if len(history) >= 2:
            recent = history[-3:]  # Last 3 observations
            agrees = sum(1 for o in recent if o == observed_state)
            disagrees = len(recent) - agrees

            if agrees == len(recent):
                # All recent observations agree — boost confidence
                effective_confidence = min(0.95, confidence + 0.05 * len(recent))
            elif disagrees >= 2:
                # Conflicting observations — reduce confidence, this is noisy
                effective_confidence = max(0.4, confidence - 0.15 * disagrees)

        belief = self.beliefs[component_id]
        state_idx = self.STATE_IDX.get(observed_state)

        if state_idx is not None:
            # Observation model: P(obs | true_state)
            base = (1.0 - effective_confidence) / 3.0
            likelihood = np.full(4, base)
            likelihood[state_idx] = effective_confidence

            # Bayes rule: posterior ∝ likelihood × prior
            posterior = likelihood * belief
            norm = posterior.sum()
            if norm > 0:
                posterior /= norm
            self.beliefs[component_id] = posterior

        self.observed[component_id] = observed_state
        self.observation_age[component_id] = 0

    def update_from_unknown(self, component_id: str):
        """Mark a component as unobservable — increase uncertainty."""
        if component_id not in self.beliefs:
            return
        # Shift belief toward uniform (maximum uncertainty)
        belief = self.beliefs[component_id]
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        self.beliefs[component_id] = 0.7 * belief + 0.3 * uniform

    def propagate_beliefs(self, graph: InfrastructureGraph):
        """Propagate failure beliefs through dependency edges.

        If we believe component A is likely failed, and B depends on A,
        then B's failure probability should increase too.
        """
        for cid in self.component_ids:
            comp = graph.get_component(cid)
            if comp is None:
                continue

            # Get components this one depends on
            deps_in = comp.dependencies_in
            if not deps_in:
                continue

            # Aggregate failure probability from dependencies
            max_dep_failure_prob = 0.0
            for dep_id in deps_in:
                if dep_id in self.beliefs:
                    dep_belief = self.beliefs[dep_id]
                    # P(failed) + P(unreachable)
                    failure_prob = dep_belief[2] + dep_belief[3]
                    dep_edge = graph.get_dependency(cid, dep_id)
                    if dep_edge and dep_edge.criticality == Criticality.HARD:
                        failure_prob *= 1.0  # Full propagation for hard deps
                    elif dep_edge and dep_edge.criticality == Criticality.SOFT:
                        failure_prob *= 0.5
                    else:
                        failure_prob *= 0.2
                    max_dep_failure_prob = max(max_dep_failure_prob, failure_prob)

            if max_dep_failure_prob > 0.1:
                # Shift belief toward failed proportional to dependency failure
                belief = self.beliefs[cid]
                shift = max_dep_failure_prob * 0.3
                belief[0] = max(0, belief[0] - shift)  # Less likely healthy
                belief[2] = min(1, belief[2] + shift * 0.7)  # More likely failed
                belief[3] = min(1, belief[3] + shift * 0.3)  # More likely unreachable
                # Renormalize
                norm = belief.sum()
                if norm > 0:
                    self.beliefs[cid] = belief / norm

    def detect_structural_contradictions(self, graph: InfrastructureGraph):
        """Detect and correct beliefs that contradict the dependency topology.

        Key insight: if a component is believed healthy but multiple of its
        dependents are believed failed (especially via HARD dependencies),
        the "healthy" observation was likely noisy. Override the belief.

        This is the fog-of-war counter-measure — when what you see doesn't
        match what the structure says should be happening, trust the structure.
        """
        contradictions = []

        for cid in self.component_ids:
            comp = graph.get_component(cid)
            if comp is None:
                continue

            belief = self.beliefs[cid]
            p_healthy = belief[0]
            p_degraded = belief[1]

            # Check components believed healthy OR only mildly degraded
            # A "slightly degraded" node with badly degraded dependents
            # is still a structural contradiction
            if p_healthy + p_degraded < 0.4:
                continue  # Already believed to be in bad shape

            # Count dependents that are believed bad (failed, degraded, or unreachable)
            dependents = graph.get_dependents(cid)
            if not dependents:
                continue

            failed_dependent_count = 0
            degraded_dependent_count = 0
            hard_failed_count = 0
            hard_degraded_count = 0
            total_dependents = len(dependents)

            for dep_comp in dependents:
                dep_belief = self.beliefs.get(dep_comp.id)
                if dep_belief is None:
                    continue
                dep_fail_prob = dep_belief[2] + dep_belief[3]
                dep_degrade_prob = dep_belief[1]
                dep_edge = graph.get_dependency(dep_comp.id, cid)
                is_hard = dep_edge and dep_edge.criticality == Criticality.HARD

                if dep_fail_prob > 0.5:
                    failed_dependent_count += 1
                    if is_hard:
                        hard_failed_count += 1
                elif dep_degrade_prob > 0.4:
                    degraded_dependent_count += 1
                    if is_hard:
                        hard_degraded_count += 1

            # Structural contradiction conditions:
            # 1. Original: healthy node with failed dependents
            # 2. NEW: healthy/mild node with MANY degraded dependents (slow burn)
            # 3. NEW: any node whose dependents are collectively worse than it
            total_bad = failed_dependent_count + degraded_dependent_count
            hard_bad = hard_failed_count + hard_degraded_count

            trigger = (
                hard_failed_count >= 2
                or (failed_dependent_count >= 3 and total_dependents >= 3)
                or (hard_bad >= 3)  # Multiple hard-dep nodes degraded
                or (total_bad >= 4 and total_bad > total_dependents * 0.4)  # >40% of dependents are bad
            )

            if trigger:
                # Override: this component is probably actually failed
                contradiction_strength = min(
                    0.8,
                    0.3 * hard_failed_count + 0.15 * failed_dependent_count
                )
                belief[0] = max(0.02, belief[0] - contradiction_strength)
                belief[2] = min(0.95, belief[2] + contradiction_strength * 0.8)
                belief[3] = min(0.95, belief[3] + contradiction_strength * 0.2)
                # Renormalize
                norm = belief.sum()
                if norm > 0:
                    self.beliefs[cid] = belief / norm

                contradictions.append((cid, hard_failed_count, failed_dependent_count))

            # Also check the reverse: component believed healthy but its
            # upstream dependencies (things IT depends on) are failed
            deps_in = comp.dependencies_in
            if not deps_in:
                continue

            hard_failed_upstream = 0
            for dep_id in deps_in:
                dep_belief = self.beliefs.get(dep_id)
                if dep_belief is None:
                    continue
                if dep_belief[2] + dep_belief[3] > 0.7:
                    dep_edge = graph.get_dependency(cid, dep_id)
                    if dep_edge and dep_edge.criticality == Criticality.HARD:
                        hard_failed_upstream += 1

            if hard_failed_upstream >= 1 and p_healthy > 0.6:
                # This component depends on something that's failed via hard dep
                # — it should also be failed/degraded
                shift = min(0.6, 0.3 * hard_failed_upstream)
                belief = self.beliefs[cid]
                belief[0] = max(0.02, belief[0] - shift)
                belief[2] = min(0.95, belief[2] + shift * 0.5)
                belief[1] = min(0.95, belief[1] + shift * 0.3)  # degraded
                belief[3] = min(0.95, belief[3] + shift * 0.2)
                norm = belief.sum()
                if norm > 0:
                    self.beliefs[cid] = belief / norm

        return contradictions

    def age_observations(self):
        """Increase uncertainty for stale observations."""
        for cid in self.component_ids:
            if cid in self.observation_age:
                self.observation_age[cid] += 1
                age = self.observation_age[cid]
                if age > 3:
                    # Belief decays toward prior
                    prior = np.array([0.85, 0.10, 0.03, 0.02])
                    decay = min(0.1 * (age - 3), 0.5)
                    self.beliefs[cid] = (1 - decay) * self.beliefs[cid] + decay * prior

    def entropy(self, component_id: str) -> float:
        """Shannon entropy of belief for a component. High = uncertain."""
        belief = self.beliefs.get(component_id, np.array([0.25, 0.25, 0.25, 0.25]))
        # Avoid log(0)
        belief = np.clip(belief, 1e-10, 1.0)
        return float(-np.sum(belief * np.log2(belief)))

    def total_entropy(self) -> float:
        """Total entropy across all components."""
        return sum(self.entropy(cid) for cid in self.component_ids)

    def most_uncertain(self, top_k: int = 5) -> list[tuple[str, float]]:
        """Components with highest entropy (most uncertain)."""
        entropies = [(cid, self.entropy(cid)) for cid in self.component_ids]
        entropies.sort(key=lambda x: -x[1])
        return entropies[:top_k]

    def most_likely_failed(self, top_k: int = 5) -> list[tuple[str, float]]:
        """Components most likely to be in a non-healthy state.
        Includes degraded, failed, and unreachable.
        Weighted: failed counts more than degraded.
        """
        probs = []
        for cid in self.component_ids:
            belief = self.beliefs[cid]
            # Weighted: failed/unreachable count full, degraded counts 0.6
            bad_prob = belief[1] * 0.6 + belief[2] + belief[3]
            # Boost by root cause candidate score
            rc_score = self.root_cause_candidates.get(cid, 0)
            boosted = min(1.0, bad_prob + rc_score * 0.05)
            probs.append((cid, float(boosted)))
        probs.sort(key=lambda x: -x[1])
        return probs[:top_k]

    def copy(self) -> "BeliefState":
        new = BeliefState(self.component_ids)
        new.beliefs = {cid: b.copy() for cid, b in self.beliefs.items()}
        new.observed = dict(self.observed)
        new.observation_history = {cid: list(h) for cid, h in self.observation_history.items()}
        new.observation_age = dict(self.observation_age)
        new.actions_taken = list(self.actions_taken)
        new.root_cause_candidates = dict(self.root_cause_candidates)
        new.identified_root_causes = list(self.identified_root_causes)
        return new


# ── Action Generator ──────────────────────────────────────────────────────


class ActionGenerator:
    """Generates available diagnostic actions given current belief state.

    Domain-agnostic: actions are typed (observe, test, intervene) and
    target abstract components. The infrastructure adapter defines
    what specific actions mean.
    """

    def __init__(self, graph: InfrastructureGraph):
        self.graph = graph

    def get_actions(self, belief: BeliefState) -> list[DiagnosticAction]:
        """Generate available actions based on current belief."""
        actions = []
        components = self.graph.get_all_components()

        for comp in components:
            # OBSERVE: check status of a component
            actions.append(DiagnosticAction(
                name=f"check_{comp.id}",
                action_type="observe",
                target_id=comp.id,
                cost=1.0,
                reveals=[comp.id],
            ))

            # TEST: trace dependencies from this component
            if comp.dependencies_in:
                actions.append(DiagnosticAction(
                    name=f"trace_{comp.id}",
                    action_type="test",
                    target_id=comp.id,
                    cost=2.0,
                    reveals=[comp.id] + comp.dependencies_in[:3],
                ))

            # INTERVENE: restart a component (only if believed degraded/failed)
            belief_vec = belief.beliefs.get(comp.id, np.array([1, 0, 0, 0]))
            if belief_vec[1] + belief_vec[2] > 0.3:  # P(degraded|failed) > 0.3
                actions.append(DiagnosticAction(
                    name=f"restart_{comp.id}",
                    action_type="intervene",
                    target_id=comp.id,
                    cost=5.0,
                    reveals=[comp.id],
                ))

        return actions

    def get_priority_actions(self, belief: BeliefState, top_k: int = 10) -> list[DiagnosticAction]:
        """Get top-K actions ranked by expected information gain.

        Prioritizes:
        1. Components with conflicting observation history (re-verify)
        2. Components with high failure probability but low observation count
        3. Components upstream of known failures (trace root cause)
        4. Standard entropy-based information gain
        """
        all_actions = self.get_actions(belief)

        scored = []
        for action in all_actions:
            expected_gain = 0.0
            for cid in action.reveals:
                current_entropy = belief.entropy(cid)
                expected_post = 0.2
                gain = current_entropy - expected_post
                expected_gain += max(0, gain)

                # BONUS: Re-verification of inconsistent observations
                history = belief.observation_history.get(cid, [])
                if len(history) >= 2:
                    unique_obs = set(history[-3:])
                    if len(unique_obs) > 1:
                        # Conflicting observations — high priority to re-check
                        expected_gain += 1.5

                # BONUS: High failure/degradation belief + few observations
                fail_prob = belief.beliefs[cid][2] + belief.beliefs[cid][3]
                degrade_prob = belief.beliefs[cid][1]
                bad_prob = fail_prob + degrade_prob
                obs_count = len(history)
                if bad_prob > 0.25 and obs_count < 2:
                    expected_gain += bad_prob * 2.5

                # BONUS: Upstream of known-bad nodes (failed OR degraded)
                comp = self.graph.get_component(cid)
                if comp:
                    bad_dependent_count = 0
                    for dep_out in comp.dependencies_out:
                        dep_belief = belief.beliefs.get(dep_out)
                        if dep_belief is not None:
                            dep_bad = dep_belief[1] + dep_belief[2] + dep_belief[3]
                            if dep_bad > 0.4:
                                bad_dependent_count += 1
                                expected_gain += 0.8
                    # Extra bonus for nodes upstream of MULTIPLE bad dependents
                    # — these are likely root causes of degradation chains
                    if bad_dependent_count >= 2:
                        expected_gain += 1.5 * bad_dependent_count

                # BONUS: Root cause candidate score from prior observations
                rc_score = belief.root_cause_candidates.get(cid, 0)
                if rc_score > 0 and obs_count < 3:
                    expected_gain += rc_score * 0.5

            # Penalize by cost
            score = expected_gain / action.cost

            # Penalize repeating exact same action recently
            if action.name in belief.actions_taken[-3:]:
                score *= 0.3

            scored.append((action, score))

        scored.sort(key=lambda x: -x[1])
        return [a for a, _ in scored[:top_k]]


# ── POMCP Tree ────────────────────────────────────────────────────────────


class POMCPNode:
    """A node in the POMCP search tree."""

    def __init__(self, parent=None):
        self.parent = parent
        self.children: dict[str, "POMCPNode"] = {}  # action_name → child
        self.observation_children: dict[str, "POMCPNode"] = {}  # obs_hash → child
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.action: DiagnosticAction | None = None

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb1(self, exploration: float = 2.0) -> float:
        """Upper Confidence Bound for action selection."""
        if self.visit_count == 0:
            return float("inf")
        parent_visits = self.parent.visit_count if self.parent else 1
        exploit = self.value
        explore = exploration * math.sqrt(math.log(parent_visits + 1) / self.visit_count)
        return exploit + explore

    def best_child(self, exploration: float = 2.0) -> tuple[str, "POMCPNode"]:
        """Select child with highest UCB1 score."""
        best_name = None
        best_score = -float("inf")
        best_node = None
        for name, child in self.children.items():
            score = child.ucb1(exploration)
            if score > best_score:
                best_score = score
                best_name = name
                best_node = child
        return best_name, best_node


# ── POMCP Solver ──────────────────────────────────────────────────────────


class POMCPSolver:
    """Partially Observable Monte Carlo Planning.

    Uses the sable_sim propagation engine as the transition model
    and fog-of-war as the observation model. Plans diagnostic actions
    to maximize information gain toward identifying root cause.

    Domain-agnostic: operates on BeliefState, DiagnosticAction,
    and ActionResult abstractions. The infrastructure specifics
    are handled by the simulator.
    """

    def __init__(
        self,
        graph: InfrastructureGraph,
        propagation: PropagationEngine | None = None,
        fog: FogOfWar | None = None,
        rollouts: int = 500,
        max_depth: int = 10,
        exploration: float = 2.0,
        discount: float = 0.95,
        seed: int = 42,
    ):
        self.graph = graph
        self.propagation = propagation or PropagationEngine()
        self.fog = fog or FogOfWar()
        self.rollouts = rollouts
        self.max_depth = max_depth
        self.exploration = exploration
        self.discount = discount
        self.rng = np.random.default_rng(seed)
        self.action_gen = ActionGenerator(graph)

    def plan(self, belief: BeliefState) -> tuple[DiagnosticAction, dict]:
        """Run POMCP to find the best next action.

        Returns (best_action, info_dict).
        """
        root = POMCPNode()

        # Get available actions
        actions = self.action_gen.get_priority_actions(belief, top_k=15)
        if not actions:
            return None, {"error": "No actions available"}

        # Initialize children for each action
        for action in actions:
            child = POMCPNode(parent=root)
            child.action = action
            root.children[action.name] = child

        # Run rollouts
        for _ in range(self.rollouts):
            # Sample a state from the belief
            sampled_state = self._sample_state(belief)
            self._simulate(root, belief.copy(), sampled_state, 0, actions)

        # Select best action (most visited — robust to noise)
        best_name = max(root.children, key=lambda n: root.children[n].visit_count)
        best_node = root.children[best_name]
        best_action = best_node.action

        # Compile diagnostics
        info = {
            "total_rollouts": self.rollouts,
            "action_scores": {},
        }
        for name, child in sorted(root.children.items(),
                                    key=lambda x: -x[1].visit_count):
            info["action_scores"][name] = {
                "visits": child.visit_count,
                "avg_value": round(child.value, 4),
                "ucb1": round(child.ucb1(self.exploration), 4),
            }

        return best_action, info

    def _simulate(self, node: POMCPNode, belief: BeliefState,
                  state: SystemState, depth: int,
                  actions: list[DiagnosticAction]) -> float:
        """Single MCTS simulation from a node."""
        if depth >= self.max_depth:
            return 0.0

        # Select action via UCB1
        if not node.children:
            return self._rollout(belief, state, depth, actions)

        _, child = node.best_child(self.exploration)
        if child.action is None:
            return 0.0

        # Execute action in simulation
        reward, obs = self._execute_action(child.action, belief, state)

        # Observation hash for tree branching
        obs_hash = str(sorted(obs.items())) if obs else "empty"

        # Get or create observation child
        if obs_hash not in child.observation_children:
            child.observation_children[obs_hash] = POMCPNode(parent=child)

        obs_node = child.observation_children[obs_hash]

        # If obs_node has no children, expand
        if not obs_node.children and depth < self.max_depth - 1:
            next_actions = self.action_gen.get_priority_actions(belief, top_k=10)
            for a in next_actions:
                if a.name not in [belief.actions_taken[-1]] if belief.actions_taken else []:
                    sub = POMCPNode(parent=obs_node)
                    sub.action = a
                    obs_node.children[a.name] = sub

        # Recurse
        future = self.discount * self._simulate(
            obs_node, belief, state, depth + 1, actions
        )
        total = reward + future

        # Backpropagate
        child.visit_count += 1
        child.value_sum += total
        obs_node.visit_count += 1
        obs_node.value_sum += total

        return total

    def _rollout(self, belief: BeliefState, state: SystemState,
                 depth: int, actions: list[DiagnosticAction]) -> float:
        """Random rollout from current position."""
        total_reward = 0.0
        discount = 1.0
        b = belief.copy()

        for _ in range(depth, self.max_depth):
            # Pick random action from priority set
            avail = self.action_gen.get_priority_actions(b, top_k=5)
            if not avail:
                break
            action = avail[self.rng.integers(0, len(avail))]
            reward, _ = self._execute_action(action, b, state)
            total_reward += discount * reward
            discount *= self.discount

        return total_reward

    def _execute_action(self, action: DiagnosticAction, belief: BeliefState,
                        state: SystemState) -> tuple[float, dict]:
        """Execute an action and return (reward, observations).

        Reward = information gain (entropy reduction) / cost.
        """
        pre_entropy = belief.total_entropy()
        observations = {}

        for cid in action.reveals:
            comp = state.graph.get_component(cid)
            if comp is None:
                continue

            # Get true state (god view) with noise
            true_state = str(comp.state)

            # Add observation noise (fog of war)
            # Re-checks get lower noise — you're being more careful
            obs_count = len(belief.observation_history.get(cid, []))
            noise_rate = max(0.02, 0.10 - 0.025 * obs_count)  # 10% → 7.5% → 5% → 2.5% → 2%
            if self.rng.random() < noise_rate:
                noise_states = ["healthy", "degraded", "failed"]
                true_state = noise_states[self.rng.integers(0, 3)]

            observations[cid] = true_state
            # Confidence increases with repeated observations
            confidence = min(0.95, 0.80 + 0.05 * obs_count)
            belief.update_from_observation(cid, true_state, confidence=confidence)

        # Propagate beliefs through graph structure
        belief.propagate_beliefs(self.graph)
        # Detect and correct structural contradictions
        belief.detect_structural_contradictions(self.graph)
        belief.actions_taken.append(action.name)
        belief.age_observations()

        # Reward = entropy reduction / cost
        post_entropy = belief.total_entropy()
        info_gain = pre_entropy - post_entropy
        reward = info_gain / max(action.cost, 0.1)

        # Bonus for observing non-healthy components
        for cid, obs_state in observations.items():
            if obs_state == "failed":
                reward += 3.0
                belief.root_cause_candidates[cid] = belief.root_cause_candidates.get(cid, 0) + 2.0
            elif obs_state == "degraded":
                reward += 2.0  # Degradation is signal, not noise
                belief.root_cause_candidates[cid] = belief.root_cause_candidates.get(cid, 0) + 1.0
            elif obs_state == "unreachable":
                reward += 2.5
                belief.root_cause_candidates[cid] = belief.root_cause_candidates.get(cid, 0) + 1.5

        # Upstream tracing bonus: if we found a degraded/failed node,
        # reward is higher if it's UPSTREAM of other known-bad nodes.
        # This drives the solver toward root causes, not symptoms.
        for cid, obs_state in observations.items():
            if obs_state in ("failed", "degraded", "unreachable"):
                comp = self.graph.get_component(cid)
                if comp is None:
                    continue
                # Count how many of this node's dependents are also believed bad
                dependents = self.graph.get_dependents(cid)
                bad_dependents = 0
                for dep in dependents:
                    dep_belief = belief.beliefs.get(dep.id)
                    if dep_belief is not None:
                        if dep_belief[1] + dep_belief[2] + dep_belief[3] > 0.4:
                            bad_dependents += 1
                if bad_dependents >= 2:
                    # This node is upstream of multiple bad nodes — likely root cause
                    reward += 2.0 * bad_dependents
                    belief.root_cause_candidates[cid] = (
                        belief.root_cause_candidates.get(cid, 0) + 1.5 * bad_dependents
                    )

        return reward, observations

    def _sample_state(self, belief: BeliefState) -> SystemState:
        """Sample a concrete state from the belief distribution.

        Creates a SystemState where each component's state is sampled
        from the belief probability vector.
        """
        state = SystemState(self.graph)

        for cid in belief.component_ids:
            comp = self.graph.get_component(cid)
            if comp is None:
                continue

            # Sample state from belief
            belief_vec = belief.beliefs[cid]
            state_idx = self.rng.choice(4, p=belief_vec)
            sampled = [
                ComponentState.HEALTHY,
                ComponentState.DEGRADED,
                ComponentState.FAILED,
                ComponentState.UNREACHABLE,
            ][state_idx]

            comp.state = sampled
            if sampled == ComponentState.HEALTHY:
                comp.health = self.rng.uniform(0.7, 1.0)
            elif sampled == ComponentState.DEGRADED:
                comp.health = self.rng.uniform(0.3, 0.5)
            elif sampled == ComponentState.FAILED:
                comp.health = self.rng.uniform(0.0, 0.2)
            else:
                comp.health = 0.0

        return state

    def _initialize_belief_from_fog(self, true_state, component_ids):
        """Build initial belief state from fog-of-war observations."""
        belief = BeliefState(component_ids)
        operator_view = self.fog.generate_operator_view(true_state)

        obs_by_component: dict[str, list] = {}
        for obs in operator_view.get("observations", []):
            cid = obs["component_id"]
            obs_by_component.setdefault(cid, []).append(obs)

        for cid, obs_list in obs_by_component.items():
            obs_list.sort(key=lambda o: o.get("tick", 0))
            self._apply_chronological_observations(belief, cid, obs_list)

        for cid in operator_view.get("unobservable_components", []):
            belief.update_from_unknown(cid)

        belief.propagate_beliefs(self.graph)
        belief.detect_structural_contradictions(self.graph)
        return belief

    def _apply_chronological_observations(self, belief, cid, obs_list):
        """Apply observations chronologically with recency weighting and alert/trend boosting."""
        for i, obs in enumerate(obs_list):
            recency_weight = 0.7 + 0.3 * (i / max(len(obs_list) - 1, 1))
            confidence = 0.75 * recency_weight
            belief.update_from_observation(cid, obs["observed_state"], confidence=confidence)

        latest = obs_list[-1]
        for alert in latest.get("alerts", []):
            if "CRITICAL" in alert:
                belief.root_cause_candidates[cid] = belief.root_cause_candidates.get(cid, 0) + 3.0
            elif "WARNING" in alert or "degraded" in alert.lower():
                belief.root_cause_candidates[cid] = belief.root_cause_candidates.get(cid, 0) + 1.5

        states_over_time = [o["observed_state"] for o in obs_list]
        state_severity = {"healthy": 0, "degraded": 1, "failed": 2, "unreachable": 2, "unknown": 1}
        severities = [state_severity.get(s, 0) for s in states_over_time]
        if len(severities) >= 2 and severities[-1] > severities[0]:
            belief.root_cause_candidates[cid] = belief.root_cause_candidates.get(cid, 0) + 2.0

    def _update_root_causes(self, belief, prob_threshold=0.6, high_prob=0.85, rc_threshold=3.0):
        """Identify new root causes from current belief state."""
        for cid, prob in belief.most_likely_failed(5):
            if prob > prob_threshold and cid not in belief.identified_root_causes:
                history = belief.observation_history.get(cid, [])
                seen_bad = any(o in ("failed", "degraded", "unreachable") for o in history)
                rc_score = belief.root_cause_candidates.get(cid, 0)
                if seen_bad or prob > high_prob or rc_score > rc_threshold:
                    belief.identified_root_causes.append(cid)

    def _find_unverified_hubs(self, belief, component_ids):
        """Find high-centrality nodes that need verification."""
        known_ids = set(belief.identified_root_causes)
        unverified = []
        for cid in component_ids:
            if cid in known_ids:
                continue
            comp = self.graph.get_component(cid)
            if comp is None:
                continue
            dependents = self.graph.get_dependents(cid)
            n_dependents = len(dependents)
            obs_count = len(belief.observation_history.get(cid, []))
            b = belief.beliefs[cid]
            bad_prob = b[1] + b[2] + b[3]
            rc_score = belief.root_cause_candidates.get(cid, 0)

            is_hub = n_dependents >= 3
            needs_check = (obs_count == 0 or (bad_prob > 0.3 and obs_count < 3) or rc_score > 1.0)
            if is_hub and needs_check:
                priority = n_dependents + rc_score * 2 + bad_prob * 3
                unverified.append((cid, priority))

        unverified.sort(key=lambda x: -x[1])
        return unverified

    def _verify_hub(self, hub_id, belief, true_state, step, session_log):
        """Execute hub verification action and log it."""
        hub_action = DiagnosticAction(
            name=f"verify_hub_{hub_id}",
            action_type="test",
            target_id=hub_id,
            cost=2.0,
            reveals=[hub_id] + (
                self.graph.get_component(hub_id).dependencies_in[:2]
                if self.graph.get_component(hub_id) else []
            ),
        )
        reward, obs = self._execute_action(hub_action, belief, true_state)
        hub_entry = {
            "step": step + 2,
            "action": hub_action.name,
            "action_type": "hub_verify",
            "target": hub_id,
            "cost": hub_action.cost,
            "observations": obs,
            "reward": round(reward, 4),
            "total_entropy": round(belief.total_entropy(), 4),
            "top_candidates": belief.most_likely_failed(5),
            "top_uncertain": belief.most_uncertain(3),
            "top_action_scores": {},
        }
        session_log.append(hub_entry)
        self._update_root_causes(belief, prob_threshold=0.5, high_prob=0.7, rc_threshold=2.0)

    def run_diagnostic_session(
        self, true_state: SystemState, max_steps: int = 10
    ) -> list[dict]:
        """Run a full diagnostic session against a ground-truth scenario.

        Returns a log of actions taken, observations received, and
        belief state evolution.
        """
        component_ids = [c.id for c in self.graph.get_all_components()]
        belief = self._initialize_belief_from_fog(true_state, component_ids)

        session_log = []

        for step in range(max_steps):
            action, info = self.plan(belief)
            if action is None:
                break

            reward, obs = self._execute_action(action, belief, true_state)

            entry = {
                "step": step + 1,
                "action": action.name,
                "action_type": action.action_type,
                "target": action.target_id,
                "cost": action.cost,
                "observations": obs,
                "reward": round(reward, 4),
                "total_entropy": round(belief.total_entropy(), 4),
                "top_candidates": belief.most_likely_failed(3),
                "top_uncertain": belief.most_uncertain(3),
                "top_action_scores": {
                    k: v for k, v in list(info.get("action_scores", {}).items())[:5]
                },
            }
            session_log.append(entry)

            self._update_root_causes(belief)

            if belief.identified_root_causes:
                known_ids = set(belief.identified_root_causes)
                residual_uncertain = [
                    (cid, belief.entropy(cid))
                    for cid in component_ids
                    if cid not in known_ids and belief.entropy(cid) > 1.2
                ]
                unexplained_failures = [
                    cid for cid, prob in belief.most_likely_failed(10)
                    if prob > 0.5 and cid not in known_ids
                    and not any(
                        dep_id in known_ids
                        for dep_id in (self.graph.get_component(cid).dependencies_in if self.graph.get_component(cid) else [])
                    )
                ]

                unverified_hubs = self._find_unverified_hubs(belief, component_ids)
                if unverified_hubs and step < max_steps - 1:
                    self._verify_hub(unverified_hubs[0][0], belief, true_state, step, session_log)
                    continue

                if not unexplained_failures and len(residual_uncertain) < len(component_ids) * 0.3:
                    causes_str = ", ".join(belief.identified_root_causes)
                    entry["verdict"] = f"Root cause(s) identified: {causes_str}"
                    break

        return session_log


# ── Demo Scenario ─────────────────────────────────────────────────────────


def build_demo_scenario(seed: int = 42) -> tuple[InfrastructureGraph, SystemState]:
    """Build a small infrastructure graph with a cascading failure.

    Scenario: Core switch fails → access switches become unreachable →
    servers behind them degrade → application services fail.
    The POMDP solver must diagnose this through partial observations.
    """
    from sable_sim.core.component import DEFAULT_PROPERTIES

    graph = InfrastructureGraph()

    # Components
    components = {
        "core-sw-1": ComponentType.CORE_SWITCH,
        "core-sw-2": ComponentType.CORE_SWITCH,
        "access-sw-1": ComponentType.ACCESS_SWITCH,
        "access-sw-2": ComponentType.ACCESS_SWITCH,
        "access-sw-3": ComponentType.ACCESS_SWITCH,
        "fw-1": ComponentType.FIREWALL,
        "srv-phys-1": ComponentType.SERVER_PHYSICAL,
        "srv-phys-2": ComponentType.SERVER_PHYSICAL,
        "hyp-1": ComponentType.HYPERVISOR,
        "vm-1": ComponentType.SERVER_VIRTUAL,
        "vm-2": ComponentType.SERVER_VIRTUAL,
        "vm-3": ComponentType.SERVER_VIRTUAL,
        "storage-1": ComponentType.STORAGE_ARRAY,
        "dns-1": ComponentType.DNS_SERVER,
        "dc-1": ComponentType.DOMAIN_CONTROLLER,
        "app-1": ComponentType.APPLICATION_SERVICE,
        "app-2": ComponentType.APPLICATION_SERVICE,
        "mon-1": ComponentType.MONITORING_SERVER,
        "wan-1": ComponentType.WAN_LINK,
        "gw-1": ComponentType.INTERNET_GATEWAY,
    }

    for cid, ctype in components.items():
        props = dict(DEFAULT_PROPERTIES.get(ctype, {}))
        graph.add_component(Component(id=cid, type=ctype, properties=props))

    # Dependencies — realistic enterprise topology
    deps = [
        ("access-sw-1", "core-sw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("access-sw-2", "core-sw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("access-sw-3", "core-sw-2", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("core-sw-1", "fw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("core-sw-2", "fw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("fw-1", "gw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("gw-1", "wan-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("srv-phys-1", "access-sw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("srv-phys-2", "access-sw-2", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("hyp-1", "access-sw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("vm-1", "hyp-1", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        ("vm-2", "hyp-1", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        ("vm-3", "srv-phys-2", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        ("vm-1", "storage-1", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        ("vm-2", "storage-1", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        ("app-1", "vm-1", DependencyType.SERVICE_DEPENDENCY, Criticality.HARD),
        ("app-2", "vm-3", DependencyType.SERVICE_DEPENDENCY, Criticality.HARD),
        ("app-1", "dns-1", DependencyType.DNS_DEPENDENCY, Criticality.SOFT),
        ("app-2", "dns-1", DependencyType.DNS_DEPENDENCY, Criticality.SOFT),
        ("app-1", "dc-1", DependencyType.AUTHENTICATION_DEPENDENCY, Criticality.HARD),
        ("app-2", "dc-1", DependencyType.AUTHENTICATION_DEPENDENCY, Criticality.HARD),
        ("mon-1", "core-sw-1", DependencyType.MONITORING_DEPENDENCY, Criticality.REDUNDANT),
        ("mon-1", "core-sw-2", DependencyType.MONITORING_DEPENDENCY, Criticality.REDUNDANT),
    ]

    for src, tgt, dtype, crit in deps:
        graph.add_dependency(Dependency(
            source_id=src, target_id=tgt, type=dtype, criticality=crit
        ))

    # Create system state and inject failure
    state = SystemState(graph)

    # ROOT CAUSE: core-sw-1 fails (firmware crash)
    core_sw = graph.get_component("core-sw-1")
    core_sw.state = ComponentState.FAILED
    core_sw.health = 0.0
    state.record_change(StateChange(
        tick=0, component_id="core-sw-1",
        previous_state=ComponentState.HEALTHY,
        new_state=ComponentState.FAILED,
        previous_health=1.0, new_health=0.0,
        cause="firmware_crash", cause_component=None
    ))

    # Propagate cascade
    engine = PropagationEngine()
    engine.propagate(state)

    return graph, state


def run_demo():
    """Run the POMDP solver on the demo scenario."""
    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Pillar 2: POMCP Solver{C_RESET}")
    print(f"  {C_DIM}{'═' * 50}{C_RESET}")
    print(f"  {C_TEXT}Scenario: Core switch firmware crash → cascade{C_RESET}")
    print(f"  {C_TEXT}Objective: Identify root cause through partial observations{C_RESET}\n")

    # Build scenario
    graph, true_state = build_demo_scenario()

    # Show ground truth (god view)
    print(f"  {C_INFO}Ground Truth (God View):{C_RESET}")
    failed = true_state.get_failed_components()
    degraded = true_state.get_degraded_components()
    print(f"    {C_DANGER}Failed:   {', '.join(c.id for c in failed)}{C_RESET}")
    print(f"    {C_GOLD}Degraded: {', '.join(c.id for c in degraded) or 'none'}{C_RESET}")
    print(f"    {C_DIM}Cascade history: {len(true_state.history)} state changes{C_RESET}")

    # Run POMCP
    print(f"\n  {C_INFO}Running POMCP (500 rollouts, depth 10)...{C_RESET}\n")
    solver = POMCPSolver(
        graph, rollouts=500, max_depth=10, seed=42,
    )

    t0 = time.time()
    session_log = solver.run_diagnostic_session(true_state, max_steps=8)
    elapsed = time.time() - t0

    # Display session
    print(f"  {C_DIM}{'step':>4s}  {'action':<30s} {'type':<10s} {'reward':>7s} {'entropy':>8s}  observations{C_RESET}")
    print(f"  {C_DIM}{'─' * 90}{C_RESET}")

    for entry in session_log:
        obs_str = ", ".join(f"{k}={v}" for k, v in entry["observations"].items())
        if len(obs_str) > 35:
            obs_str = obs_str[:35] + "..."
        print(
            f"  {C_TEXT}{entry['step']:4d}  "
            f"{entry['action']:<30s} "
            f"{entry['action_type']:<10s} "
            f"{entry['reward']:7.2f} "
            f"{entry['total_entropy']:8.2f}{C_RESET}  "
            f"{C_DIM}{obs_str}{C_RESET}"
        )

        if "verdict" in entry:
            print(f"\n  {C_SUCCESS}{C_BOLD}  {entry['verdict']}{C_RESET}")

    # Final analysis
    print(f"\n  {C_GOLD}{C_BOLD}  Diagnostic Summary{C_RESET}")
    print(f"  {C_DIM}{'─' * 50}{C_RESET}")
    print(f"  {C_TEXT}Steps taken:    {C_BRIGHT}{len(session_log)}{C_RESET}")
    print(f"  {C_TEXT}Time:           {C_BRIGHT}{elapsed:.2f}s{C_RESET}")
    print(f"  {C_TEXT}Rollouts/step:  {C_BRIGHT}500{C_RESET}")

    if session_log:
        last = session_log[-1]
        print(f"\n  {C_INFO}Top root cause candidates:{C_RESET}")
        for cid, prob in last["top_candidates"]:
            is_root = cid == "core-sw-1"
            marker = f"{C_SUCCESS}← ROOT CAUSE{C_RESET}" if is_root else ""
            print(f"    {C_TEXT}{cid:<20s} P(failed)={prob:.3f}{C_RESET} {marker}")

        print(f"\n  {C_INFO}Remaining uncertainty:{C_RESET}")
        for cid, ent in last["top_uncertain"]:
            print(f"    {C_DIM}{cid:<20s} H={ent:.3f}{C_RESET}")

    # Check if root cause was identified
    if session_log:
        last = session_log[-1]
        top = last["top_candidates"]
        if top and top[0][0] == "core-sw-1":
            print(f"\n  {C_SUCCESS}{C_BOLD}ROOT CAUSE CORRECTLY IDENTIFIED{C_RESET}")
        else:
            print(f"\n  {C_DANGER}Root cause not in top position{C_RESET}")
            if any(c[0] == "core-sw-1" for c in top):
                rank = next(i for i, c in enumerate(top) if c[0] == "core-sw-1") + 1
                print(f"  {C_TEXT}core-sw-1 ranked #{rank}{C_RESET}")

    print()


if __name__ == "__main__":
    run_demo()
