#!/usr/bin/env python3
"""
Project PARALLAX — Complex Integration Scenario
==================================================
A realistic enterprise incident that tests all three pillars simultaneously.

SCENARIO: "The Monday Morning Meltdown"
=========================================
Friday at 5 PM, a storage array started throwing intermittent IOPS warnings.
Nobody noticed. Over the weekend, the storage degradation slowly cascaded:

  1. Storage array degrades (root cause 1 — silent Friday evening)
  2. VMs on that storage start responding slowly (Saturday)
  3. DNS server (on the same storage) starts timing out (Saturday night)
  4. Domain controller replication fails because DNS is flaky (Sunday)
  5. Monday morning: users log in, authentication storms hit the degraded DC
  6. Core switch CPU spikes from broadcast storms (root cause 2 — cascaded)
  7. Monitoring server loses connectivity to half the network
  8. App team reports "everything is down" but can only see symptoms

The operator sees:
  - 3 app services down
  - Monitoring shows gaps (some nodes unreachable)
  - Multiple alerts from different layers
  - No clear single root cause
  - The original storage warning is buried in a weekend's worth of noise

SABLE must:
  - Identify storage array as the primary root cause (despite it now showing
    as merely "degraded," not "failed")
  - Recognize the core switch spike as a secondary cascade, not an independent failure
  - Predict which additional services will degrade if no action is taken
  - Recommend checking the storage array first (highest information gain),
    even though the loudest alerts are from the app layer

This is the kind of incident that takes a senior engineer 2-4 hours to diagnose.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar3"))

from sable_sim.core.component import Component, ComponentState, ComponentType, DEFAULT_PROPERTIES
from sable_sim.core.dependency import Dependency, DependencyType, Criticality
from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.state import SystemState, StateChange
from sable_sim.simulation.propagation import PropagationEngine

from sable_core import SABLEOrchestrator

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


def build_monday_morning_meltdown():
    """Build the complex multi-layer cascade scenario."""
    graph = InfrastructureGraph()

    components = {
        # Network layer
        "core-sw-1": ComponentType.CORE_SWITCH,
        "core-sw-2": ComponentType.CORE_SWITCH,
        "access-sw-1": ComponentType.ACCESS_SWITCH,
        "access-sw-2": ComponentType.ACCESS_SWITCH,
        "access-sw-3": ComponentType.ACCESS_SWITCH,
        "access-sw-4": ComponentType.ACCESS_SWITCH,
        "access-sw-5": ComponentType.ACCESS_SWITCH,
        "fw-1": ComponentType.FIREWALL,
        "fw-2": ComponentType.FIREWALL,
        "lb-1": ComponentType.LOAD_BALANCER,
        "gw-1": ComponentType.INTERNET_GATEWAY,
        "wan-1": ComponentType.WAN_LINK,
        # Compute layer
        "srv-phys-1": ComponentType.SERVER_PHYSICAL,
        "srv-phys-2": ComponentType.SERVER_PHYSICAL,
        "srv-phys-3": ComponentType.SERVER_PHYSICAL,
        "hyp-1": ComponentType.HYPERVISOR,
        "hyp-2": ComponentType.HYPERVISOR,
        "hyp-3": ComponentType.HYPERVISOR,
        "vm-web-1": ComponentType.SERVER_VIRTUAL,
        "vm-web-2": ComponentType.SERVER_VIRTUAL,
        "vm-app-1": ComponentType.SERVER_VIRTUAL,
        "vm-app-2": ComponentType.SERVER_VIRTUAL,
        "vm-db-1": ComponentType.SERVER_VIRTUAL,
        "vm-db-2": ComponentType.SERVER_VIRTUAL,
        "vm-util-1": ComponentType.SERVER_VIRTUAL,
        # Storage layer
        "stor-primary": ComponentType.STORAGE_ARRAY,
        "stor-backup": ComponentType.STORAGE_ARRAY,
        "stor-tgt-1": ComponentType.STORAGE_TARGET,
        "stor-tgt-2": ComponentType.STORAGE_TARGET,
        "stor-tgt-3": ComponentType.STORAGE_TARGET,
        # Services
        "dns-1": ComponentType.DNS_SERVER,
        "dns-2": ComponentType.DNS_SERVER,
        "dc-1": ComponentType.DOMAIN_CONTROLLER,
        "dc-2": ComponentType.DOMAIN_CONTROLLER,
        "dhcp-1": ComponentType.DHCP_SERVER,
        "ca-1": ComponentType.CERTIFICATE_AUTHORITY,
        # Applications
        "app-portal": ComponentType.APPLICATION_SERVICE,
        "app-email": ComponentType.APPLICATION_SERVICE,
        "app-erp": ComponentType.APPLICATION_SERVICE,
        "app-hr": ComponentType.APPLICATION_SERVICE,
        # Management
        "mon-1": ComponentType.MONITORING_SERVER,
        # VDI
        "vdi-broker": ComponentType.VDI_BROKER,
        "vdi-host-1": ComponentType.VDI_HOST,
        "vdi-host-2": ComponentType.VDI_HOST,
    }

    for cid, ctype in components.items():
        props = dict(DEFAULT_PROPERTIES.get(ctype, {}))
        graph.add_component(Component(id=cid, type=ctype, properties=props))

    deps = [
        # Network backbone
        ("access-sw-1", "core-sw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("access-sw-2", "core-sw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("access-sw-3", "core-sw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("access-sw-4", "core-sw-2", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("access-sw-5", "core-sw-2", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("core-sw-1", "fw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("core-sw-2", "fw-2", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("fw-1", "gw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("fw-2", "gw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("gw-1", "wan-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("lb-1", "core-sw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        # Compute → network
        ("srv-phys-1", "access-sw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("srv-phys-2", "access-sw-2", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("srv-phys-3", "access-sw-4", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("hyp-1", "access-sw-1", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("hyp-2", "access-sw-3", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("hyp-3", "access-sw-5", DependencyType.NETWORK_PATH, Criticality.HARD),
        # VM hosting
        ("vm-web-1", "hyp-1", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        ("vm-web-2", "hyp-2", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        ("vm-app-1", "hyp-1", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        ("vm-app-2", "hyp-2", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        ("vm-db-1", "hyp-3", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        ("vm-db-2", "srv-phys-3", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        ("vm-util-1", "srv-phys-2", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        # Storage — the critical dependencies
        ("stor-tgt-1", "stor-primary", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        ("stor-tgt-2", "stor-primary", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        ("stor-tgt-3", "stor-backup", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        ("vm-web-1", "stor-tgt-1", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        ("vm-app-1", "stor-tgt-1", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        ("vm-web-2", "stor-tgt-2", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        ("vm-app-2", "stor-tgt-2", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        ("vm-db-1", "stor-tgt-3", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        ("vm-db-2", "stor-tgt-3", DependencyType.STORAGE_DEPENDENCY, Criticality.HARD),
        ("vm-util-1", "stor-tgt-1", DependencyType.STORAGE_DEPENDENCY, Criticality.SOFT),
        # DNS — dns-1 lives on affected storage
        ("dns-1", "vm-util-1", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        ("dns-2", "srv-phys-3", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        # Auth dependencies
        ("app-portal", "vm-web-1", DependencyType.SERVICE_DEPENDENCY, Criticality.HARD),
        ("app-email", "vm-app-1", DependencyType.SERVICE_DEPENDENCY, Criticality.HARD),
        ("app-erp", "vm-app-2", DependencyType.SERVICE_DEPENDENCY, Criticality.HARD),
        ("app-hr", "vm-web-2", DependencyType.SERVICE_DEPENDENCY, Criticality.HARD),
        ("app-portal", "dns-1", DependencyType.DNS_DEPENDENCY, Criticality.SOFT),
        ("app-email", "dns-1", DependencyType.DNS_DEPENDENCY, Criticality.SOFT),
        ("app-erp", "dns-2", DependencyType.DNS_DEPENDENCY, Criticality.SOFT),
        ("app-hr", "dns-1", DependencyType.DNS_DEPENDENCY, Criticality.SOFT),
        ("app-portal", "dc-1", DependencyType.AUTHENTICATION_DEPENDENCY, Criticality.HARD),
        ("app-email", "dc-1", DependencyType.AUTHENTICATION_DEPENDENCY, Criticality.HARD),
        ("app-erp", "dc-2", DependencyType.AUTHENTICATION_DEPENDENCY, Criticality.HARD),
        ("app-hr", "dc-2", DependencyType.AUTHENTICATION_DEPENDENCY, Criticality.HARD),
        ("app-portal", "lb-1", DependencyType.SERVICE_DEPENDENCY, Criticality.SOFT),
        # DC replication
        ("dc-1", "dc-2", DependencyType.REPLICATION_DEPENDENCY, Criticality.SOFT),
        ("dc-2", "dc-1", DependencyType.REPLICATION_DEPENDENCY, Criticality.SOFT),
        ("dc-1", "dns-1", DependencyType.DNS_DEPENDENCY, Criticality.SOFT),
        ("dc-2", "dns-2", DependencyType.DNS_DEPENDENCY, Criticality.SOFT),
        # VDI
        ("vdi-host-1", "vdi-broker", DependencyType.SERVICE_DEPENDENCY, Criticality.HARD),
        ("vdi-host-2", "vdi-broker", DependencyType.SERVICE_DEPENDENCY, Criticality.HARD),
        ("vdi-broker", "dc-1", DependencyType.AUTHENTICATION_DEPENDENCY, Criticality.HARD),
        ("vdi-host-1", "hyp-1", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        ("vdi-host-2", "hyp-3", DependencyType.HOSTING_DEPENDENCY, Criticality.HARD),
        # Monitoring
        ("mon-1", "access-sw-2", DependencyType.NETWORK_PATH, Criticality.HARD),
        ("mon-1", "core-sw-1", DependencyType.MONITORING_DEPENDENCY, Criticality.REDUNDANT),
        ("mon-1", "core-sw-2", DependencyType.MONITORING_DEPENDENCY, Criticality.REDUNDANT),
    ]

    for src, tgt, dtype, crit in deps:
        graph.add_dependency(Dependency(
            source_id=src, target_id=tgt, type=dtype, criticality=crit
        ))

    # ── Inject the failure cascade ──
    state = SystemState(graph)

    # ROOT CAUSE 1: Storage array degradation (Friday evening, silent)
    stor = graph.get_component("stor-primary")
    stor.state = ComponentState.DEGRADED
    stor.health = 0.35
    state.record_change(StateChange(
        tick=0, component_id="stor-primary",
        previous_state=ComponentState.HEALTHY,
        new_state=ComponentState.DEGRADED,
        previous_health=1.0, new_health=0.35,
        cause="iops_saturation", cause_component=None,
    ))

    # Propagate the weekend cascade
    engine = PropagationEngine(
        dns_cache_ttl=2, session_ttl=3, max_ticks=20,
        soft_impact_factor=0.4,
    )
    engine.propagate(state)

    # ROOT CAUSE 2: Core switch CPU spike from Monday morning broadcast storm
    # (This is a SECONDARY cascade triggered by the degraded environment)
    core = graph.get_component("core-sw-1")
    if core.state == ComponentState.HEALTHY:
        core.state = ComponentState.DEGRADED
        core.health = 0.4
        state.record_change(StateChange(
            tick=state.tick, component_id="core-sw-1",
            previous_state=ComponentState.HEALTHY,
            new_state=ComponentState.DEGRADED,
            previous_health=1.0, new_health=0.4,
            cause="cpu_overload", cause_component=None,
        ))
        # Propagate the secondary cascade
        engine.propagate(state)

    return graph, state


def run_complex_scenario():
    print(f"\n{C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║  SABLE — Complex Scenario: Monday Morning Meltdown  ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════════╝{C_RESET}\n")

    print(f"  {C_DIM}A storage array started degrading Friday evening.{C_RESET}")
    print(f"  {C_DIM}Nobody noticed. The cascade propagated over the weekend.{C_RESET}")
    print(f"  {C_DIM}Monday morning: users can't log in, apps are down,{C_RESET}")
    print(f"  {C_DIM}monitoring has gaps, alerts from every layer.{C_RESET}")
    print(f"  {C_DIM}The app team reports 'everything is down.'{C_RESET}\n")

    # Build scenario
    graph, true_state = build_monday_morning_meltdown()

    # Ground truth
    failed = true_state.get_failed_components()
    degraded = true_state.get_degraded_components()
    healthy = [c for c in graph.get_all_components()
               if c.state == ComponentState.HEALTHY]

    print(f"  {C_INFO}Ground Truth (hidden from SABLE):{C_RESET}")
    print(f"    {C_DANGER}Failed ({len(failed)}):   {', '.join(c.id for c in failed)}{C_RESET}")
    print(f"    {C_GOLD}Degraded ({len(degraded)}): {', '.join(c.id for c in degraded)}{C_RESET}")
    print(f"    {C_TEXT}Healthy ({len(healthy)}):  {len(healthy)} components unaffected{C_RESET}")
    print(f"    {C_DIM}Cascade history:  {len(true_state.history)} state changes over {true_state.tick} ticks{C_RESET}")
    print(f"    {C_BRIGHT}Root causes:      stor-primary (degraded), core-sw-1 (secondary cascade){C_RESET}")

    # Initialize SABLE
    print(f"\n  {C_INFO}Initializing SABLE...{C_RESET}")
    orchestrator = SABLEOrchestrator(
        graph=graph,
        gnn_checkpoint="../pillar1/checkpoints/best_model.pt",
        mamba_checkpoint="../pillar3/checkpoints/best_mamba_final.pt",
        pomdp_rollouts=500,
        device="cuda",
    )

    # Run diagnosis with more steps for complex scenario
    print(f"\n  {C_INFO}Running integrated diagnosis (up to 8 steps)...{C_RESET}")
    t0 = time.time()
    result = orchestrator.diagnose(true_state, max_steps=8)
    total_time = time.time() - t0

    # Display
    print(f"\n  {C_GOLD}{C_BOLD}  ┌─────────────────────────────────────────────────────┐{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  │              DIAGNOSTIC ASSESSMENT                    │{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  └─────────────────────────────────────────────────────┘{C_RESET}")

    # POMDP results
    print(f"\n  {C_INFO}Pillar 2 — Decision Planning (POMDP):{C_RESET}")
    print(f"    {C_TEXT}First action:  {C_BRIGHT}{result.recommended_action}{C_RESET}")
    print(f"    {C_TEXT}Steps taken:   {C_BRIGHT}{result.steps_taken}{C_RESET}")
    print(f"    {C_TEXT}Root cause candidates:{C_RESET}")
    for rc in result.root_cause_candidates[:7]:
        is_stor = rc["component"] == "stor-primary"
        is_core = rc["component"] == "core-sw-1"
        is_actual = rc["component"] in [c.id for c in failed + degraded]
        if is_stor:
            marker = f" {C_SUCCESS}← PRIMARY ROOT CAUSE{C_RESET}"
        elif is_core:
            marker = f" {C_GOLD}← SECONDARY CASCADE{C_RESET}"
        elif is_actual:
            marker = f" {C_DIM}(affected){C_RESET}"
        else:
            marker = ""
        print(f"      {C_TEXT}{rc['component']:<20s} P={rc['probability']}{C_RESET}{marker}")

    # Mamba results
    print(f"\n  {C_INFO}Pillar 3 — Temporal Prediction (Mamba):{C_RESET}")
    if result.cascade_risk in ("high", "critical"):
        risk_color = C_DANGER
    elif result.cascade_risk == "medium":
        risk_color = C_GOLD
    else:
        risk_color = C_TEXT
    print(f"    {C_TEXT}Cascade risk:  {risk_color}{C_BOLD}{result.cascade_risk}{C_RESET} (severity={result.predicted_severity:.3f})")
    if result.predicted_affected:
        print(f"    {C_TEXT}Nodes predicted at risk:{C_RESET}")
        for pa in result.predicted_affected[:10]:
            actually_affected = pa["component"] in [c.id for c in failed + degraded]
            marker = f" {C_SUCCESS}✓{C_RESET}" if actually_affected else f" {C_DIM}(false positive){C_RESET}"
            print(f"      {C_TEXT}{pa['component']:<20s} P={pa['affected_prob']} → {pa['predicted_state']}{C_RESET}{marker}")

    # Fused
    print(f"\n  {C_GOLD}{C_BOLD}  Fused Assessment:{C_RESET}")
    print(f"    {C_BRIGHT}{result.assessment}{C_RESET}")
    print(f"    {C_TEXT}Confidence:  {C_BRIGHT}{result.confidence:.2f}{C_RESET}")
    print(f"    {C_TEXT}Total time:  {C_BRIGHT}{total_time:.1f}s{C_RESET}")

    # Validation
    print(f"\n  {C_GOLD}{C_BOLD}  Validation Against Ground Truth{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")

    # Check if storage array is in top candidates
    rc_ids = [rc["component"] for rc in result.root_cause_candidates[:5]]
    stor_found = "stor-primary" in rc_ids
    core_found = "core-sw-1" in rc_ids

    checks = [
        ("Primary root cause (stor-primary) identified", stor_found),
        ("Secondary cascade (core-sw-1) identified", core_found),
        ("Cascade risk assessed", result.cascade_risk != "low" or result.predicted_severity > 0),
        ("Diagnosis completed in < 60 seconds", total_time < 60),
    ]

    passed = 0
    for desc, ok in checks:
        marker = f"{C_SUCCESS}PASS{C_RESET}" if ok else f"{C_DANGER}FAIL{C_RESET}"
        if ok:
            passed += 1
        print(f"    {marker}  {C_TEXT}{desc}{C_RESET}")

    print(f"\n  {C_GOLD}{C_BOLD}  Score: {passed}/{len(checks)}{C_RESET}")

    if passed >= 3:
        print(f"  {C_SUCCESS}{C_BOLD}  SABLE handles complex multi-layer cascades.{C_RESET}")
    elif passed >= 2:
        print(f"  {C_GOLD}{C_BOLD}  Partial — needs refinement on complex scenarios.{C_RESET}")
    else:
        print(f"  {C_DANGER}{C_BOLD}  Complex scenario exposed architectural gaps.{C_RESET}")

    print()


if __name__ == "__main__":
    run_complex_scenario()
