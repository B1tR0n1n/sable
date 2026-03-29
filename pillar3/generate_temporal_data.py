#!/usr/bin/env python3
"""
Project PARALLAX — Pillar 3: Temporal Dataset Generator
=========================================================
Generates time-series state sequences from the SABLE infrastructure
simulator for Mamba training.

Each sequence captures the full system state at every tick during
a failure cascade. The Mamba model learns to predict future state
from past state — temporal reasoning.

Domain-agnostic output: state vectors are abstract (health values +
state one-hot), not infrastructure-specific. The adapter layer in
the simulator handles the domain mapping.

Usage:
    python generate_temporal_data.py --count 5000 --output temporal_data.pt
    python generate_temporal_data.py --count 10000 --max-ticks 30

Requires: ml-env with torch, numpy, networkx
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sable_sim.core.states import N_STATES, STATE_NAMES, STATE_MAP
from sable_sim.core.component import Component, ComponentState, ComponentType, DEFAULT_PROPERTIES
from sable_sim.core.dependency import Dependency, DependencyType, Criticality
from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.state import SystemState, StateChange
from sable_sim.simulation.propagation import PropagationEngine
from sable_sim.simulation.failure_injection import FailureInjector
from sable_sim.utils.random import SeededRandom

# ── Terminal Colors ────────────────────────────────────────────────────────

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"

# State encoding imported from sable_sim.core.states

# Component type encoding: 20 types → index
COMP_TYPES = list(ComponentType)
COMP_TYPE_MAP = {t: i for i, t in enumerate(COMP_TYPES)}
N_COMP_TYPES = len(COMP_TYPES)

# Per-node feature dim: health (1) + state one-hot (5) + type one-hot (20) = 26
NODE_FEAT_DIM = 1 + N_STATES + N_COMP_TYPES


def encode_system_state(graph: InfrastructureGraph, component_ids: list[str]) -> np.ndarray:
    """Encode the current system state as a flat feature vector.

    Returns: (N_nodes * NODE_FEAT_DIM,) vector
    """
    features = []
    for cid in component_ids:
        comp = graph.get_component(cid)
        if comp is None:
            features.extend([0.0] * NODE_FEAT_DIM)
            continue

        # Health (scalar)
        feat = [comp.health]

        # State one-hot
        state_oh = [0.0] * N_STATES
        state_idx = STATE_MAP.get(comp.state, 0)
        state_oh[state_idx] = 1.0
        feat.extend(state_oh)

        # Type one-hot
        type_oh = [0.0] * N_COMP_TYPES
        type_idx = COMP_TYPE_MAP.get(comp.type, 0)
        type_oh[type_idx] = 1.0
        feat.extend(type_oh)

        features.extend(feat)

    return np.array(features, dtype=np.float32)


def build_random_topology(rng: SeededRandom, min_nodes: int = 15, max_nodes: int = 40) -> InfrastructureGraph:
    """Build a random but realistic infrastructure topology."""
    graph = InfrastructureGraph()
    n_nodes = rng.randint(min_nodes, max_nodes)

    # Type distribution
    type_weights = {
        ComponentType.CORE_SWITCH: 0.05,
        ComponentType.ACCESS_SWITCH: 0.10,
        ComponentType.FIREWALL: 0.05,
        ComponentType.ROUTER: 0.05,
        ComponentType.SERVER_PHYSICAL: 0.12,
        ComponentType.SERVER_VIRTUAL: 0.18,
        ComponentType.HYPERVISOR: 0.08,
        ComponentType.STORAGE_ARRAY: 0.04,
        ComponentType.DNS_SERVER: 0.04,
        ComponentType.DOMAIN_CONTROLLER: 0.04,
        ComponentType.APPLICATION_SERVICE: 0.15,
        ComponentType.MONITORING_SERVER: 0.03,
        ComponentType.WAN_LINK: 0.03,
        ComponentType.INTERNET_GATEWAY: 0.04,
    }
    types = list(type_weights.keys())
    probs = np.array(list(type_weights.values()))
    probs = probs / probs.sum()

    # Generate nodes
    for i in range(n_nodes):
        ctype = rng.weighted_choice(types, probs.tolist())
        props = dict(DEFAULT_PROPERTIES.get(ctype, {}))
        graph.add_component(Component(id=f"n{i}", type=ctype, properties=props))

    components = graph.get_all_components()
    by_type = {}
    for c in components:
        by_type.setdefault(c.type, []).append(c)

    # Generate dependencies based on type relationships
    dep_patterns = [
        (ComponentType.ACCESS_SWITCH, ComponentType.CORE_SWITCH, DependencyType.NETWORK_PATH, Criticality.HARD),
        (ComponentType.SERVER_PHYSICAL, ComponentType.ACCESS_SWITCH, DependencyType.NETWORK_PATH, Criticality.HARD),
        (ComponentType.HYPERVISOR, ComponentType.ACCESS_SWITCH, DependencyType.NETWORK_PATH, Criticality.HARD),
        (ComponentType.SERVER_VIRTUAL, ComponentType.HYPERVISOR, DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        (ComponentType.SERVER_VIRTUAL, ComponentType.STORAGE_ARRAY, DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        (ComponentType.APPLICATION_SERVICE, ComponentType.SERVER_VIRTUAL, DependencyType.SERVICE_DEPENDENCY, Criticality.HARD),
        (ComponentType.APPLICATION_SERVICE, ComponentType.DNS_SERVER, DependencyType.DNS_DEPENDENCY, Criticality.SOFT),
        (ComponentType.APPLICATION_SERVICE, ComponentType.DOMAIN_CONTROLLER, DependencyType.AUTHENTICATION_DEPENDENCY, Criticality.HARD),
        (ComponentType.CORE_SWITCH, ComponentType.FIREWALL, DependencyType.NETWORK_PATH, Criticality.HARD),
        (ComponentType.FIREWALL, ComponentType.INTERNET_GATEWAY, DependencyType.NETWORK_PATH, Criticality.HARD),
        (ComponentType.INTERNET_GATEWAY, ComponentType.WAN_LINK, DependencyType.NETWORK_PATH, Criticality.HARD),
    ]

    for src_type, tgt_type, dep_type, crit in dep_patterns:
        srcs = by_type.get(src_type, [])
        tgts = by_type.get(tgt_type, [])
        if not srcs or not tgts:
            continue
        for src in srcs:
            # Connect to a random target of the right type
            tgt = rng.choice(tgts)
            if src.id != tgt.id:
                graph.add_dependency(Dependency(
                    source_id=src.id, target_id=tgt.id,
                    type=dep_type, criticality=crit,
                ))

    return graph


def generate_sequence(
    rng: SeededRandom,
    max_ticks: int = 30,
    min_nodes: int = 15,
    max_nodes: int = 40,
) -> dict | None:
    """Generate a single temporal sequence from a simulation run.

    Returns dict with:
        - states: (T, N*NODE_FEAT_DIM) array — system state at each tick
        - n_nodes: int
        - n_ticks: int
        - root_cause: str (component ID that was injected)
    """
    # Build random topology
    graph = build_random_topology(rng, min_nodes, max_nodes)
    components = graph.get_all_components()
    if len(components) < min_nodes:
        return None

    component_ids = [c.id for c in components]
    n_nodes = len(component_ids)

    # Create state and snapshot initial
    state = SystemState(graph)
    states = [encode_system_state(graph, component_ids)]

    # Inject a random failure
    injector = FailureInjector(rng)
    try:
        injections = injector.generate_failures(state, difficulty=rng.choice(["easy", "medium", "hard"]))
    except Exception:
        return None

    if not injections:
        return None

    for inj in injections:
        injector.inject(state, inj)

    # Snapshot post-injection
    states.append(encode_system_state(graph, component_ids))

    # Propagate tick-by-tick, capturing state at each tick
    engine = PropagationEngine(max_ticks=max_ticks)

    for tick in range(max_ticks):
        current_tick = state.tick
        changes_this_tick = []
        changed_ids = set()

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
                    continue
                dep = state.graph.get_dependency(dependent.id, cid)
                if dep is None:
                    continue
                sc = engine._evaluate_impact(state, comp, dependent, dep, current_tick)
                if sc is not None:
                    changes_this_tick.append(sc)
                    changed_ids.add(dependent.id)

        delayed = engine._process_pending_delays(state, current_tick)
        for sc in delayed:
            if sc.component_id not in changed_ids:
                changes_this_tick.append(sc)
                changed_ids.add(sc.component_id)

        new_tick = state.advance_tick()

        if not changes_this_tick:
            if state.has_reached_steady_state(lookback=2):
                break
            # Snapshot unchanged state
            states.append(encode_system_state(graph, component_ids))
            continue

        for sc in changes_this_tick:
            sc.tick = new_tick
            state.record_change(sc)

        # Snapshot after this tick's changes
        states.append(encode_system_state(graph, component_ids))

    if len(states) < 3:
        return None

    # Trim static tail — find last tick where state actually changed
    states_arr = np.array(states, dtype=np.float32)
    last_dynamic = 1  # At least keep tick 0 and 1
    for t in range(1, len(states_arr)):
        diff = np.abs(states_arr[t] - states_arr[t - 1]).sum()
        if diff > 0.01:
            last_dynamic = t

    # Keep up to 2 ticks past last change (show stabilization)
    trim_to = min(last_dynamic + 2, len(states_arr))
    states_arr = states_arr[:trim_to]

    if len(states_arr) < 3:
        return None

    root_cause = injections[0].component_id if injections else "unknown"

    return {
        "states": states_arr,
        "n_nodes": n_nodes,
        "n_ticks": len(states_arr),
        "root_cause": root_cause,
        "component_ids": component_ids,
    }


def generate_dataset(
    count: int = 5000,
    max_ticks: int = 30,
    min_nodes: int = 15,
    max_nodes: int = 40,
    output_path: str = "temporal_data.pt",
    seed: int = 42,
):
    """Generate a full temporal dataset for Mamba training."""

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Pillar 3: Temporal Data Generation{C_RESET}")
    print(f"  {C_DIM}{'═' * 55}{C_RESET}\n")
    print(f"  {C_TEXT}target sequences: {C_BRIGHT}{count}{C_RESET}")
    print(f"  {C_TEXT}max ticks/seq:    {C_BRIGHT}{max_ticks}{C_RESET}")
    print(f"  {C_TEXT}node range:       {C_BRIGHT}{min_nodes}-{max_nodes}{C_RESET}")

    rng = SeededRandom(seed)
    sequences = []
    failed = 0
    t0 = time.time()

    for i in range(count * 2):  # Oversample to account for failures
        if len(sequences) >= count:
            break

        seq = generate_sequence(rng, max_ticks, min_nodes, max_nodes)
        if seq is None:
            failed += 1
            continue

        sequences.append(seq)

        if len(sequences) % 500 == 0:
            elapsed = time.time() - t0
            rate = len(sequences) / elapsed
            print(f"  {C_DIM}{len(sequences):5d}/{count} ({rate:.0f}/s){C_RESET}")

    elapsed = time.time() - t0
    print(f"\n  {C_TEXT}generated: {C_BRIGHT}{len(sequences)}{C_RESET} sequences ({failed} failed)")
    print(f"  {C_TEXT}time:      {C_BRIGHT}{elapsed:.1f}s{C_RESET}")

    # Pad sequences to uniform length for batching
    # Find max nodes and max ticks across all sequences
    max_n = max(s["n_nodes"] for s in sequences)
    max_t = max(s["n_ticks"] for s in sequences)
    feat_dim = NODE_FEAT_DIM

    print(f"\n  {C_INFO}Padding to uniform shape...{C_RESET}")
    print(f"  {C_TEXT}max nodes:  {C_BRIGHT}{max_n}{C_RESET}")
    print(f"  {C_TEXT}max ticks:  {C_BRIGHT}{max_t}{C_RESET}")
    print(f"  {C_TEXT}feat dim:   {C_BRIGHT}{feat_dim}{C_RESET}")
    print(f"  {C_TEXT}state dim:  {C_BRIGHT}{max_n * feat_dim}{C_RESET} (nodes × features)")

    # Pack into tensors
    # Shape: (N_sequences, max_ticks, max_nodes * feat_dim)
    state_dim = max_n * feat_dim
    X = torch.zeros(len(sequences), max_t, state_dim, dtype=torch.float32)
    lengths = torch.zeros(len(sequences), dtype=torch.long)
    n_nodes_list = torch.zeros(len(sequences), dtype=torch.long)

    for i, seq in enumerate(sequences):
        t_len = seq["n_ticks"]
        n = seq["n_nodes"]
        actual_dim = n * feat_dim
        states = seq["states"]  # (t_len, actual_dim)
        X[i, :t_len, :actual_dim] = torch.tensor(states)
        lengths[i] = t_len
        n_nodes_list[i] = n

    # Train/val/test split (80/10/10)
    n_total = len(sequences)
    perm = torch.randperm(n_total)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    dataset = {
        "X": X,
        "lengths": lengths,
        "n_nodes": n_nodes_list,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "max_nodes": max_n,
        "max_ticks": max_t,
        "feat_dim": feat_dim,
        "state_dim": state_dim,
        "n_sequences": n_total,
        "node_feat_dim": feat_dim,
        "n_states": N_STATES,
        "n_comp_types": N_COMP_TYPES,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, str(output))

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"\n  {C_SUCCESS}{C_BOLD}Saved:{C_RESET} {C_TEXT}{output}{C_RESET} ({size_mb:.1f} MB)")

    print(f"\n  {C_GOLD}{C_BOLD}  Dataset Summary{C_RESET}")
    print(f"  {C_DIM}{'─' * 40}{C_RESET}")
    print(f"  {C_TEXT}sequences:  {C_BRIGHT}{n_total}{C_RESET}")
    print(f"  {C_TEXT}train:      {C_BRIGHT}{len(train_idx)}{C_RESET}")
    print(f"  {C_TEXT}val:        {C_BRIGHT}{len(val_idx)}{C_RESET}")
    print(f"  {C_TEXT}test:       {C_BRIGHT}{len(test_idx)}{C_RESET}")
    print(f"  {C_TEXT}shape:      {C_BRIGHT}({n_total}, {max_t}, {state_dim}){C_RESET}")
    print(f"  {C_TEXT}tensor size:{C_BRIGHT} {size_mb:.1f} MB{C_RESET}")

    # Distribution stats
    tick_counts = lengths.numpy()
    node_counts = n_nodes_list.numpy()
    print(f"\n  {C_INFO}Tick distribution:{C_RESET}")
    print(f"    {C_DIM}min={tick_counts.min()} median={int(np.median(tick_counts))} "
          f"max={tick_counts.max()} mean={tick_counts.mean():.1f}{C_RESET}")
    print(f"  {C_INFO}Node distribution:{C_RESET}")
    print(f"    {C_DIM}min={node_counts.min()} median={int(np.median(node_counts))} "
          f"max={node_counts.max()} mean={node_counts.mean():.1f}{C_RESET}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate temporal training data")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--max-ticks", type=int, default=30)
    parser.add_argument("--min-nodes", type=int, default=15)
    parser.add_argument("--max-nodes", type=int, default=40)
    parser.add_argument("--output", type=str, default="temporal_data.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        count=args.count,
        max_ticks=args.max_ticks,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        output_path=args.output,
        seed=args.seed,
    )
