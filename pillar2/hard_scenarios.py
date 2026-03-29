#!/usr/bin/env python3
"""
Project PARALLAX — Pillar 2: Hard POMCP Scenarios
====================================================
Three scenarios designed to break naive diagnostic approaches.

Scenario 1: BLIND CASCADE
  Monitoring server fails first, THEN core switch fails.
  The solver loses visibility before the real problem starts.
  Must reason about what it CAN'T see.

Scenario 2: DUAL ROOT CAUSE
  Storage array and DNS server fail independently at the same time.
  Symptoms overlap — VMs degrade from storage, apps degrade from DNS.
  Solver must identify TWO independent root causes, not chase one.

Scenario 3: DELAYED POISON
  Domain controller has intermittent auth failures.
  Effects don't cascade immediately — auth sessions have TTL.
  By the time apps start failing, the DC looks healthy again.
  Solver must reason about temporal causality.

Usage:
    python hard_scenarios.py                    # Run all three
    python hard_scenarios.py --scenario 1       # Run specific scenario
    python hard_scenarios.py --rollouts 1000    # More rollouts for harder scenarios
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sable_sim.core.component import Component, ComponentState, ComponentType, DEFAULT_PROPERTIES
from sable_sim.core.dependency import Dependency, DependencyType, Criticality
from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.state import SystemState, StateChange
from sable_sim.simulation.propagation import PropagationEngine
from sable_sim.simulation.fog import FogOfWar
from sable_sim.utils.random import SeededRandom

from pomcp import POMCPSolver, BeliefState

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


# ── Shared Topology ───────────────────────────────────────────────────────

def build_enterprise_topology() -> InfrastructureGraph:
    """30-node enterprise topology used by all scenarios."""
    graph = InfrastructureGraph()

    components = {
        # Network layer
        "core-sw-1": ComponentType.CORE_SWITCH,
        "core-sw-2": ComponentType.CORE_SWITCH,
        "access-sw-1": ComponentType.ACCESS_SWITCH,
        "access-sw-2": ComponentType.ACCESS_SWITCH,
        "access-sw-3": ComponentType.ACCESS_SWITCH,
        "access-sw-4": ComponentType.ACCESS_SWITCH,
        "fw-1": ComponentType.FIREWALL,
        "gw-1": ComponentType.INTERNET_GATEWAY,
        "wan-1": ComponentType.WAN_LINK,
        # Compute layer
        "srv-1": ComponentType.SERVER_PHYSICAL,
        "srv-2": ComponentType.SERVER_PHYSICAL,
        "hyp-1": ComponentType.HYPERVISOR,
        "hyp-2": ComponentType.HYPERVISOR,
        "vm-1": ComponentType.SERVER_VIRTUAL,
        "vm-2": ComponentType.SERVER_VIRTUAL,
        "vm-3": ComponentType.SERVER_VIRTUAL,
        "vm-4": ComponentType.SERVER_VIRTUAL,
        "vm-5": ComponentType.SERVER_VIRTUAL,
        # Storage
        "stor-1": ComponentType.STORAGE_ARRAY,
        "stor-tgt-1": ComponentType.STORAGE_TARGET,
        "stor-tgt-2": ComponentType.STORAGE_TARGET,
        # Services
        "dns-1": ComponentType.DNS_SERVER,
        "dns-2": ComponentType.DNS_SERVER,
        "dc-1": ComponentType.DOMAIN_CONTROLLER,
        "dc-2": ComponentType.DOMAIN_CONTROLLER,
        "dhcp-1": ComponentType.DHCP_SERVER,
        # Applications
        "app-1": ComponentType.APPLICATION_SERVICE,
        "app-2": ComponentType.APPLICATION_SERVICE,
        "app-3": ComponentType.APPLICATION_SERVICE,
        # Management
        "mon-1": ComponentType.MONITORING_SERVER,
    }

    for cid, ctype in components.items():
        props = dict(DEFAULT_PROPERTIES.get(ctype, {}))
        graph.add_component(Component(id=cid, type=ctype, properties=props))

    # Dense dependency mesh
    deps = [
        # Network backbone
        ("access-sw-1", "core-sw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("access-sw-2", "core-sw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("access-sw-3", "core-sw-2", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("access-sw-4", "core-sw-2", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("core-sw-1", "fw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("core-sw-2", "fw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("fw-1", "gw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("gw-1", "wan-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        # Compute → network
        ("srv-1", "access-sw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("srv-2", "access-sw-2", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("hyp-1", "access-sw-3", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("hyp-2", "access-sw-4", DependencyType.NETWORK_PATH, Criticality.HARD),
        # VM hosting
        ("vm-1", "hyp-1", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        ("vm-2", "hyp-1", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        ("vm-3", "hyp-2", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        ("vm-4", "srv-1", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        ("vm-5", "srv-2", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        # Storage
        ("stor-tgt-1", "stor-1", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        ("stor-tgt-2", "stor-1", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        ("vm-1", "stor-tgt-1", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        ("vm-2", "stor-tgt-1", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        ("vm-3", "stor-tgt-2", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        # Services
        ("app-1", "vm-1", DependencyType.SERVICE_DEPENDENCY, Criticality.HARD),
        ("app-2", "vm-3", DependencyType.SERVICE_DEPENDENCY, Criticality.HARD),
        ("app-3", "vm-5", DependencyType.SERVICE_DEPENDENCY, Criticality.HARD),
        ("app-1", "dns-1", DependencyType.DNS_DEPENDENCY, Criticality.SOFT),
        ("app-2", "dns-1", DependencyType.DNS_DEPENDENCY, Criticality.SOFT),
        ("app-3", "dns-2", DependencyType.DNS_DEPENDENCY, Criticality.SOFT),
        ("app-1", "dc-1", DependencyType.AUTHENTICATION_DEPENDENCY, Criticality.HARD),
        ("app-2", "dc-1", DependencyType.AUTHENTICATION_DEPENDENCY, Criticality.HARD),
        ("app-3", "dc-2", DependencyType.AUTHENTICATION_DEPENDENCY, Criticality.HARD),
        ("dc-1", "dc-2", DependencyType.REPLICATION_DEPENDENCY, Criticality.SOFT),
        ("dc-2", "dc-1", DependencyType.REPLICATION_DEPENDENCY, Criticality.SOFT),
        # Monitoring
        ("mon-1", "access-sw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("mon-1", "core-sw-1", DependencyType.MONITORING_DEPENDENCY, Criticality.REDUNDANT),
        ("srv-1", "mon-1", DependencyType.MONITORING_DEPENDENCY, Criticality.REDUNDANT),
        ("srv-2", "mon-1", DependencyType.MONITORING_DEPENDENCY, Criticality.REDUNDANT),
        ("hyp-1", "mon-1", DependencyType.MONITORING_DEPENDENCY, Criticality.REDUNDANT),
        ("hyp-2", "mon-1", DependencyType.MONITORING_DEPENDENCY, Criticality.REDUNDANT),
        ("stor-1", "mon-1", DependencyType.MONITORING_DEPENDENCY, Criticality.REDUNDANT),
    ]

    for src, tgt, dtype, crit in deps:
        graph.add_dependency(Dependency(
            source_id=src, target_id=tgt, type=dtype, criticality=crit
        ))

    return graph


# ── Scenario 1: BLIND CASCADE ─────────────────────────────────────────────

def scenario_blind_cascade() -> tuple[InfrastructureGraph, SystemState, list[str]]:
    """Monitoring dies first, then real failure starts. Solver is blind."""
    graph = build_enterprise_topology()
    state = SystemState(graph)

    # Step 1: Monitoring server fails (you lose eyes)
    mon = graph.get_component("mon-1")
    mon.state = ComponentState.FAILED
    mon.health = 0.0
    state.record_change(StateChange(
        tick=0, component_id="mon-1",
        previous_state=ComponentState.HEALTHY,
        new_state=ComponentState.FAILED,
        previous_health=1.0, new_health=0.0,
        cause="storage_exhaustion", cause_component=None
    ))

    # Step 2: Core switch fails (but monitoring can't see it)
    core = graph.get_component("core-sw-1")
    core.state = ComponentState.FAILED
    core.health = 0.0
    state.record_change(StateChange(
        tick=1, component_id="core-sw-1",
        previous_state=ComponentState.HEALTHY,
        new_state=ComponentState.FAILED,
        previous_health=1.0, new_health=0.0,
        cause="firmware_crash", cause_component=None
    ))

    # Propagate
    engine = PropagationEngine()
    engine.propagate(state)

    return graph, state, ["mon-1", "core-sw-1"]


# ── Scenario 2: DUAL ROOT CAUSE ───────────────────────────────────────────

def scenario_dual_root_cause() -> tuple[InfrastructureGraph, SystemState, list[str]]:
    """Two independent failures. Overlapping symptoms."""
    graph = build_enterprise_topology()
    state = SystemState(graph)

    # Root cause 1: Storage array controller failure
    stor = graph.get_component("stor-1")
    stor.state = ComponentState.FAILED
    stor.health = 0.0
    state.record_change(StateChange(
        tick=0, component_id="stor-1",
        previous_state=ComponentState.HEALTHY,
        new_state=ComponentState.FAILED,
        previous_health=1.0, new_health=0.0,
        cause="controller_failure", cause_component=None
    ))

    # Root cause 2: DNS server cache poisoning (independent)
    dns = graph.get_component("dns-1")
    dns.state = ComponentState.FAILED
    dns.health = 0.0
    state.record_change(StateChange(
        tick=0, component_id="dns-1",
        previous_state=ComponentState.HEALTHY,
        new_state=ComponentState.FAILED,
        previous_health=1.0, new_health=0.0,
        cause="cache_poisoning", cause_component=None
    ))

    # Propagate both cascades
    engine = PropagationEngine()
    engine.propagate(state)

    return graph, state, ["stor-1", "dns-1"]


# ── Scenario 3: DELAYED POISON ────────────────────────────────────────────

def scenario_delayed_poison() -> tuple[InfrastructureGraph, SystemState, list[str]]:
    """DC fails but effects are delayed by session TTL.
    By the time apps fail, the DC might look recovered.
    Tests temporal reasoning."""
    graph = build_enterprise_topology()
    state = SystemState(graph)

    # Root cause: DC-1 kerberos failure
    dc = graph.get_component("dc-1")
    dc.state = ComponentState.FAILED
    dc.health = 0.0
    state.record_change(StateChange(
        tick=0, component_id="dc-1",
        previous_state=ComponentState.HEALTHY,
        new_state=ComponentState.FAILED,
        previous_health=1.0, new_health=0.0,
        cause="kerberos_failure", cause_component=None
    ))

    # Propagate with auth delay
    engine = PropagationEngine(session_ttl=3)
    engine.propagate(state)

    # After propagation, "recover" the DC (it was intermittent)
    # This makes it harder — the root cause looks healthy now
    dc.state = ComponentState.DEGRADED
    dc.health = 0.6
    state.record_change(StateChange(
        tick=state.tick, component_id="dc-1",
        previous_state=ComponentState.FAILED,
        new_state=ComponentState.DEGRADED,
        previous_health=0.0, new_health=0.6,
        cause="partial_recovery", cause_component=None
    ))

    return graph, state, ["dc-1"]


# ── Runner ─────────────────────────────────────────────────────────────────

def run_scenario(name: str, description: str,
                 graph: InfrastructureGraph, true_state: SystemState,
                 root_causes: list[str], rollouts: int = 500,
                 max_steps: int = 10):
    """Run POMCP against a scenario and report results."""

    print(f"\n{C_GOLD}{C_BOLD}  Scenario: {name}{C_RESET}")
    print(f"  {C_DIM}{description}{C_RESET}\n")

    # Ground truth
    failed = true_state.get_failed_components()
    degraded = true_state.get_degraded_components()
    print(f"  {C_INFO}Ground Truth:{C_RESET}")
    print(f"    {C_DANGER}Failed:      {', '.join(c.id for c in failed) or 'none'}{C_RESET}")
    print(f"    {C_GOLD}Degraded:    {', '.join(c.id for c in degraded) or 'none'}{C_RESET}")
    print(f"    {C_BRIGHT}Root causes: {', '.join(root_causes)}{C_RESET}")
    print(f"    {C_DIM}Cascade:     {len(true_state.history)} state changes{C_RESET}")

    # Configure fog with reduced coverage for harder scenarios
    fog = FogOfWar(
        monitoring_coverage=0.70,
        monitoring_delay=2,
        polling_interval=2,
        false_positive_rate=0.08,
        metric_noise_stddev=0.10,
        rng=SeededRandom(42),
    )

    solver = POMCPSolver(
        graph, rollouts=rollouts, max_depth=12,
        exploration=2.5, fog=fog, seed=42,
    )

    t0 = time.time()
    session_log = solver.run_diagnostic_session(true_state, max_steps=max_steps)
    elapsed = time.time() - t0

    # Display
    print(f"\n  {C_DIM}{'step':>4s}  {'action':<30s} {'type':<10s} {'reward':>7s} {'entropy':>8s}  observations{C_RESET}")
    print(f"  {C_DIM}{'─' * 95}{C_RESET}")

    for entry in session_log:
        obs_str = ", ".join(f"{k}={v}" for k, v in entry["observations"].items())
        if len(obs_str) > 40:
            obs_str = obs_str[:40] + "..."

        # Highlight when we find a root cause
        found_root = any(
            k in root_causes and v in ("failed", "degraded")
            for k, v in entry["observations"].items()
        )
        marker = f" {C_SUCCESS}◆{C_RESET}" if found_root else ""

        print(
            f"  {C_TEXT}{entry['step']:4d}  "
            f"{entry['action']:<30s} "
            f"{entry['action_type']:<10s} "
            f"{entry['reward']:7.2f} "
            f"{entry['total_entropy']:8.2f}{C_RESET}  "
            f"{C_DIM}{obs_str}{C_RESET}{marker}"
        )

        if "verdict" in entry:
            print(f"\n  {C_SUCCESS}{C_BOLD}  → {entry['verdict']}{C_RESET}")

    # Results
    print(f"\n  {C_GOLD}{C_BOLD}  Results{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")
    print(f"  {C_TEXT}Steps:     {C_BRIGHT}{len(session_log)}{C_RESET}")
    print(f"  {C_TEXT}Time:      {C_BRIGHT}{elapsed:.2f}s{C_RESET}")

    if session_log:
        last = session_log[-1]
        print(f"\n  {C_INFO}Top root cause candidates:{C_RESET}")
        for cid, prob in last["top_candidates"]:
            is_root = cid in root_causes
            marker = f" {C_SUCCESS}← ROOT CAUSE{C_RESET}" if is_root else ""
            print(f"    {C_TEXT}{cid:<20s} P(failed)={prob:.3f}{C_RESET}{marker}")

        # Score: did we find all root causes?
        top_5_ids = [c[0] for c in last["top_candidates"][:5]]
        found = [rc for rc in root_causes if rc in top_5_ids]
        missed = [rc for rc in root_causes if rc not in top_5_ids]

        print(f"\n  {C_INFO}Root cause identification:{C_RESET}")
        for rc in found:
            rank = top_5_ids.index(rc) + 1
            print(f"    {C_SUCCESS}FOUND: {rc} (rank #{rank}){C_RESET}")
        for rc in missed:
            print(f"    {C_DANGER}MISSED: {rc}{C_RESET}")

        accuracy = len(found) / len(root_causes) if root_causes else 0
        if accuracy >= 1.0 - 1e-9:
            print(f"\n  {C_SUCCESS}{C_BOLD}ALL ROOT CAUSES IDENTIFIED ({len(found)}/{len(root_causes)}){C_RESET}")
        elif accuracy > 0:
            print(f"\n  {C_GOLD}{C_BOLD}PARTIAL ({len(found)}/{len(root_causes)} root causes found){C_RESET}")
        else:
            print(f"\n  {C_DANGER}{C_BOLD}FAILED — no root causes in top 5{C_RESET}")

    print()
    return {
        "name": name,
        "steps": len(session_log),
        "time": elapsed,
        "root_causes": root_causes,
        "found": found if session_log else [],
        "accuracy": accuracy if session_log else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Hard POMCP Scenarios")
    parser.add_argument("--scenario", type=int, default=0, help="1, 2, or 3. 0=all")
    parser.add_argument("--rollouts", type=int, default=500)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Pillar 2: Hard Scenarios{C_RESET}")
    print(f"  {C_DIM}{'═' * 55}{C_RESET}")

    scenarios = []

    if args.scenario in (0, 1):
        graph, state, roots = scenario_blind_cascade()
        scenarios.append(run_scenario(
            "BLIND CASCADE",
            "Monitoring dies first, then core switch fails. Solver has no eyes.",
            graph, state, roots, args.rollouts, args.steps,
        ))

    if args.scenario in (0, 2):
        graph, state, roots = scenario_dual_root_cause()
        scenarios.append(run_scenario(
            "DUAL ROOT CAUSE",
            "Storage array + DNS server fail independently. Overlapping symptoms.",
            graph, state, roots, args.rollouts, args.steps,
        ))

    if args.scenario in (0, 3):
        graph, state, roots = scenario_delayed_poison()
        scenarios.append(run_scenario(
            "DELAYED POISON",
            "DC fails intermittently. By the time apps die, DC looks recovered.",
            graph, state, roots, args.rollouts, args.steps,
        ))

    # Summary
    if len(scenarios) > 1:
        print(f"\n{C_GOLD}{C_BOLD}  Summary{C_RESET}")
        print(f"  {C_DIM}{'─' * 60}{C_RESET}")
        print(f"  {C_DIM}{'Scenario':<25s} {'Steps':>5s} {'Time':>7s} {'Found':>10s} {'Result':>10s}{C_RESET}")
        for s in scenarios:
            if s["accuracy"] >= 1.0 - 1e-9:
                result = "PASS"
            elif s["accuracy"] > 0:
                result = "PARTIAL"
            else:
                result = "FAIL"

            if result == "PASS":
                c = C_SUCCESS
            elif result == "PARTIAL":
                c = C_GOLD
            else:
                c = C_DANGER
            print(
                f"  {C_TEXT}{s['name']:<25s} "
                f"{s['steps']:5d} "
                f"{s['time']:6.2f}s "
                f"{len(s['found'])}/{len(s['root_causes'])}      "
                f"{c}{result:>10s}{C_RESET}"
            )
        print()


if __name__ == "__main__":
    main()
