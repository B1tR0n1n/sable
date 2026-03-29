#!/usr/bin/env python3
"""
Project PARALLAX — Pillar 3: Temporal Data v3
================================================
Generates training data with the RIGHT objectives for cascade prediction:

1. Input: first K ticks of a cascade (the "early signal")
2. Target: final steady-state (the "outcome")
3. Per-node labels: healthy/degraded/failed at outcome

The Mamba model learns: "given the early signs, predict the full cascade."
This is what temporal reasoning actually means in infrastructure.

Usage:
    python generate_temporal_data_v3.py --count 10000 --output temporal_v3.pt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sable_sim.core.component import Component, ComponentState, ComponentType, DEFAULT_PROPERTIES
from sable_sim.core.dependency import Dependency, DependencyType, Criticality
from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.state import SystemState, StateChange
from sable_sim.simulation.propagation import PropagationEngine
from sable_sim.simulation.failure_injection import FailureInjector
from sable_sim.utils.random import SeededRandom

from generate_temporal_data import (
    build_random_topology, encode_system_state,
    NODE_FEAT_DIM, STATE_MAP, N_STATES, N_COMP_TYPES,
    C_GOLD, C_DIM, C_TEXT, C_BRIGHT, C_SUCCESS, C_INFO, C_RESET, C_BOLD,
)

# Per-node target: health (1) + state class (1) = 2
NODE_TARGET_DIM = 2


def encode_node_targets(graph: InfrastructureGraph, component_ids: list[str]) -> np.ndarray:
    """Encode per-node outcome targets.

    Returns: (N_nodes, 2) — [health, state_class] per node
        state_class: 0=healthy, 1=degraded, 2=failed, 3=unreachable
    """
    targets = []
    for cid in component_ids:
        comp = graph.get_component(cid)
        if comp is None:
            targets.append([0.0, 2.0])  # Unknown → assume failed
            continue
        state_class = float(STATE_MAP.get(comp.state, 0))
        targets.append([comp.health, state_class])
    return np.array(targets, dtype=np.float32)


def generate_cascade_sample(
    rng: SeededRandom,
    max_ticks: int = 30,
    min_nodes: int = 15,
    max_nodes: int = 40,
    input_ticks: int = 2,
) -> dict | None:
    """Generate a single cascade prediction sample.

    Returns:
        input_seq: (input_ticks, N*NODE_FEAT_DIM) — early cascade state
        outcome_state: (N*NODE_FEAT_DIM,) — final steady state
        node_targets: (N, 2) — per-node [health, state_class] at outcome
        cascade_severity: float — fraction of nodes affected
        n_affected: int — number of nodes that changed state
    """
    graph = build_random_topology(rng, min_nodes, max_nodes)
    components = graph.get_all_components()
    if len(components) < min_nodes:
        return None

    component_ids = [c.id for c in components]
    n_nodes = len(component_ids)

    # Snapshot initial state (all healthy)
    initial_state = encode_system_state(graph, component_ids)

    # Create state and inject failure
    state = SystemState(graph)
    injector = FailureInjector(rng)
    try:
        injections = injector.generate_failures(
            state, difficulty=rng.weighted_choice(
                ["easy", "medium", "hard"], [0.2, 0.5, 0.3]
            )
        )
    except Exception:
        return None

    if not injections:
        return None

    # 40% of the time, convert failures to degradations (soft cascades)
    # This trains Mamba on the slow-burn pattern
    use_degradation = rng.random() < 0.4
    for inj in injections:
        injector.inject(state, inj)
        if use_degradation:
            comp = graph.get_component(inj.component_id)
            if comp and comp.state == ComponentState.FAILED:
                comp.state = ComponentState.DEGRADED
                comp.health = rng.uniform(0.25, 0.45)

    # Snapshot post-injection (tick 1)
    post_injection = encode_system_state(graph, component_ids)

    # Build input sequence: [initial, post-injection, ...]
    input_seq = [initial_state, post_injection]

    # Propagate — use softer impact for degradation scenarios
    soft_factor = 0.4 if use_degradation else 0.3
    engine = PropagationEngine(max_ticks=max_ticks, soft_impact_factor=soft_factor)

    # Run propagation tick by tick, capture first few ticks
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
            if len(input_seq) < input_ticks:
                input_seq.append(encode_system_state(graph, component_ids))
            continue

        for sc in changes_this_tick:
            sc.tick = new_tick
            state.record_change(sc)

        if len(input_seq) < input_ticks:
            input_seq.append(encode_system_state(graph, component_ids))

    # Pad input sequence to exactly input_ticks
    while len(input_seq) < input_ticks:
        input_seq.append(input_seq[-1].copy())

    input_seq = input_seq[:input_ticks]

    # Final steady state = outcome
    outcome_state = encode_system_state(graph, component_ids)
    node_targets = encode_node_targets(graph, component_ids)

    # Compute cascade severity
    n_affected = sum(
        1 for c in components
        if c.state != ComponentState.HEALTHY
    )
    cascade_severity = n_affected / max(n_nodes, 1)

    return {
        "input_seq": np.array(input_seq, dtype=np.float32),  # (input_ticks, N*feat_dim)
        "outcome_state": outcome_state,  # (N*feat_dim,)
        "node_targets": node_targets,  # (N, 2)
        "n_nodes": n_nodes,
        "cascade_severity": cascade_severity,
        "n_affected": n_affected,
        "root_cause": injections[0].component_id if injections else "unknown",
    }


def generate_dataset(
    count: int = 10000,
    input_ticks: int = 2,
    max_ticks: int = 30,
    min_nodes: int = 15,
    max_nodes: int = 40,
    output_path: str = "temporal_v3.pt",
    seed: int = 42,
):
    """Generate full cascade prediction dataset."""

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Pillar 3: Cascade Dataset v3{C_RESET}")
    print(f"  {C_DIM}{'═' * 55}{C_RESET}\n")
    print(f"  {C_TEXT}target sequences: {C_BRIGHT}{count}{C_RESET}")
    print(f"  {C_TEXT}input ticks:      {C_BRIGHT}{input_ticks}{C_RESET}")
    print(f"  {C_TEXT}node range:       {C_BRIGHT}{min_nodes}-{max_nodes}{C_RESET}")

    rng = SeededRandom(seed)
    samples = []
    failed = 0
    t0 = time.time()

    for i in range(count * 2):
        if len(samples) >= count:
            break
        sample = generate_cascade_sample(rng, max_ticks, min_nodes, max_nodes, input_ticks)
        if sample is None:
            failed += 1
            continue
        samples.append(sample)
        if len(samples) % 1000 == 0:
            elapsed = time.time() - t0
            rate = len(samples) / elapsed
            print(f"  {C_DIM}{len(samples):5d}/{count} ({rate:.0f}/s){C_RESET}")

    elapsed = time.time() - t0
    print(f"\n  {C_TEXT}generated: {C_BRIGHT}{len(samples)}{C_RESET} ({failed} failed, {elapsed:.1f}s)")

    max_n = max(s["n_nodes"] for s in samples)
    feat_dim = NODE_FEAT_DIM
    state_dim = max_n * feat_dim

    # Pack input sequences: (N_samples, input_ticks, state_dim)
    X_input = torch.zeros(len(samples), input_ticks, state_dim, dtype=torch.float32)
    # Pack outcome: (N_samples, state_dim)
    X_outcome = torch.zeros(len(samples), state_dim, dtype=torch.float32)
    # Pack per-node targets: (N_samples, max_n, 2)
    Y_nodes = torch.zeros(len(samples), max_n, NODE_TARGET_DIM, dtype=torch.float32)
    # Cascade severity: (N_samples,)
    Y_severity = torch.zeros(len(samples), dtype=torch.float32)
    # Node counts
    N_nodes = torch.zeros(len(samples), dtype=torch.long)

    for i, s in enumerate(samples):
        n = s["n_nodes"]
        actual_dim = n * feat_dim
        for t in range(input_ticks):
            X_input[i, t, :actual_dim] = torch.tensor(s["input_seq"][t])
        X_outcome[i, :actual_dim] = torch.tensor(s["outcome_state"])
        Y_nodes[i, :n, :] = torch.tensor(s["node_targets"])
        Y_severity[i] = s["cascade_severity"]
        N_nodes[i] = n

    # Split
    n_total = len(samples)
    perm = torch.randperm(n_total)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    dataset = {
        "X_input": X_input,
        "X_outcome": X_outcome,
        "Y_nodes": Y_nodes,
        "Y_severity": Y_severity,
        "N_nodes": N_nodes,
        "train_idx": perm[:n_train],
        "val_idx": perm[n_train:n_train + n_val],
        "test_idx": perm[n_train + n_val:],
        "max_nodes": max_n,
        "input_ticks": input_ticks,
        "feat_dim": feat_dim,
        "state_dim": state_dim,
        "n_samples": n_total,
        "node_target_dim": NODE_TARGET_DIM,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, str(output))
    size_mb = output.stat().st_size / (1024 * 1024)

    # Stats
    severities = Y_severity.numpy()
    print(f"\n  {C_GOLD}{C_BOLD}  Dataset Summary{C_RESET}")
    print(f"  {C_DIM}{'─' * 45}{C_RESET}")
    print(f"  {C_TEXT}samples:      {C_BRIGHT}{n_total}{C_RESET}")
    print(f"  {C_TEXT}train/val/test:{C_BRIGHT} {n_train}/{n_val}/{n_total - n_train - n_val}{C_RESET}")
    print(f"  {C_TEXT}input shape:  {C_BRIGHT}({n_total}, {input_ticks}, {state_dim}){C_RESET}")
    print(f"  {C_TEXT}outcome shape:{C_BRIGHT} ({n_total}, {state_dim}){C_RESET}")
    print(f"  {C_TEXT}node targets: {C_BRIGHT}({n_total}, {max_n}, {NODE_TARGET_DIM}){C_RESET}")
    print(f"  {C_TEXT}file size:    {C_BRIGHT}{size_mb:.1f} MB{C_RESET}")
    print(f"\n  {C_INFO}Cascade severity distribution:{C_RESET}")
    print(f"    {C_DIM}min={severities.min():.2f} median={np.median(severities):.2f} "
          f"max={severities.max():.2f} mean={severities.mean():.2f}{C_RESET}")
    bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01]
    for j in range(len(bins) - 1):
        count_bin = ((severities >= bins[j]) & (severities < bins[j + 1])).sum()
        bar = "█" * (count_bin // 50)
        print(f"    {C_DIM}{bins[j]:.1f}-{bins[j+1]:.1f}: {count_bin:5d}{C_RESET} {C_GOLD}{bar}{C_RESET}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--input-ticks", type=int, default=2)
    parser.add_argument("--output", type=str, default="temporal_v3.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        count=args.count, input_ticks=args.input_ticks,
        output_path=args.output, seed=args.seed,
    )
