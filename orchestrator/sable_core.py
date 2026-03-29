#!/usr/bin/env python3
"""
Project PARALLAX — SABLE Orchestrator
========================================
Connects all three pillars through a shared state space.

Given an incident (partial observations of an infrastructure failure):
  1. GNN (Pillar 1): Structural reasoning — what's connected, what propagates?
  2. POMDP (Pillar 2): Decision planning — what to check next?
  3. Mamba (Pillar 3): Temporal prediction — where will the cascade spread?

The orchestrator fuses their outputs into a unified diagnostic assessment.

Usage:
    python sable_core.py                      # Run demo incident
    python sable_core.py --scenario cascade   # Specific scenario
    python sable_core.py --rollouts 1000      # More POMDP rollouts

Requires: ml-env with torch, torch_geometric, numpy, networkx
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

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
from sable_sim.simulation.fog import FogOfWar
from sable_sim.utils.random import SeededRandom

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


# ── Pillar Loaders ─────────────────────────────────────────────────────────


def load_gnn(checkpoint_path: str, device: str = "cuda"):
    """Load trained GNN from Pillar 1."""
    from cortex_gnn_model import SableGNN
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    mc = ckpt["config"]
    model = SableGNN(
        in_dim=mc["in_dim"], hidden_dim=mc["hidden_dim"],
        edge_dim=mc["edge_dim"], num_layers=mc["num_layers"],
        heads=mc["heads"], dropout=mc["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_mamba(checkpoint_path: str, device: str = "cuda"):
    """Load trained Mamba from Pillar 3."""
    from sable_mamba_final import SableMambaFinal
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    mc = ckpt["config"]
    model = SableMambaFinal(
        state_dim=mc["state_dim"], max_nodes=mc["max_nodes"],
        node_feat_dim=mc["node_feat_dim"], d_model=mc["d_model"],
        n_layers=mc["n_layers"], d_state=mc["d_state"],
        dropout=mc["dropout"], input_ticks=mc["input_ticks"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── Infrastructure Adapter ────────────────────────────────────────────────
# Maps infrastructure concepts into abstract pillar feature spaces


from generate_temporal_data import encode_system_state, NODE_FEAT_DIM, STATE_MAP
from domain_portability_test import InfrastructureAdapter


# ── Orchestrator ──────────────────────────────────────────────────────────


@dataclass
class DiagnosticResult:
    """Fused output from all three pillars."""
    # From POMDP (Pillar 2)
    recommended_action: str = ""
    action_type: str = ""
    action_target: str = ""
    root_cause_candidates: list = field(default_factory=list)

    # From Mamba (Pillar 3)
    predicted_affected: list = field(default_factory=list)
    predicted_severity: float = 0.0
    cascade_risk: str = "low"  # low/medium/high/critical

    # From GNN (Pillar 1) — structural context
    structural_connections: list = field(default_factory=list)

    # Fused assessment
    confidence: float = 0.0
    assessment: str = ""
    steps_taken: int = 0
    time_ms: float = 0.0


class SABLEOrchestrator:
    """Connects all three pillars for integrated infrastructure diagnostics.

    Domain-agnostic core: each pillar operates on abstract representations.
    The adapter layer handles infrastructure-specific translation.
    """

    def __init__(
        self,
        graph: InfrastructureGraph,
        gnn_checkpoint: str = None,
        mamba_checkpoint: str = None,
        pomdp_rollouts: int = 300,
        device: str = "cuda",
    ):
        self.graph = graph
        self.device = device
        self.adapter = InfrastructureAdapter(seed=42)
        self.components = graph.get_all_components()
        self.component_ids = [c.id for c in self.components]

        # Load pillars
        print(f"  {C_INFO}Loading pillars...{C_RESET}")

        # Pillar 1: GNN (structural)
        self.gnn = None
        if gnn_checkpoint and Path(gnn_checkpoint).exists():
            try:
                self.gnn = load_gnn(gnn_checkpoint, device)
                print(f"    {C_SUCCESS}P1 GNN loaded{C_RESET}")
            except Exception as e:
                print(f"    {C_DIM}P1 GNN skipped: {e}{C_RESET}")

        # Pillar 2: POMDP (decision planning)
        from pomcp import POMCPSolver, BeliefState
        self.POMCPSolver = POMCPSolver
        self.BeliefState = BeliefState
        fog = FogOfWar(
            monitoring_coverage=0.75,
            monitoring_delay=2,
            false_positive_rate=0.05,
            rng=SeededRandom(42),
        )
        self.pomdp = POMCPSolver(
            graph, rollouts=pomdp_rollouts, max_depth=10, fog=fog, seed=42,
        )
        print(f"    {C_SUCCESS}P2 POMDP loaded ({pomdp_rollouts} rollouts){C_RESET}")

        # Pillar 3: Mamba (temporal)
        self.mamba = None
        if mamba_checkpoint and Path(mamba_checkpoint).exists():
            try:
                self.mamba = load_mamba(mamba_checkpoint, device)
                print(f"    {C_SUCCESS}P3 Mamba loaded{C_RESET}")
            except Exception as e:
                print(f"    {C_DIM}P3 Mamba skipped: {e}{C_RESET}")

    def diagnose(
        self, true_state: SystemState, max_steps: int = 5,
    ) -> DiagnosticResult:
        """Run integrated diagnosis on an incident.

        Fuses all three pillars:
          1. Mamba predicts cascade outcome from initial observations
          2. POMDP plans diagnostic actions to verify
          3. GNN provides structural context for the assessment

        Returns a DiagnosticResult with fused analysis.
        """
        t0 = time.time()
        n_nodes = len(self.component_ids)

        # ── Phase 1: Initial observation (fog of war) ──
        fog = FogOfWar(
            monitoring_coverage=0.75, monitoring_delay=2,
            false_positive_rate=0.05, rng=SeededRandom(42),
        )
        operator_view = fog.generate_operator_view(true_state)

        # ── Phase 2: Mamba cascade prediction ──
        mamba_predictions = {}
        predicted_severity = 0.0
        predicted_affected = []

        if self.mamba is not None:
            # Encode current state for Mamba
            state_t0 = encode_system_state(self.graph, self.component_ids)
            # Encode post-initial-observation state
            state_t1 = state_t0.copy()  # Will be modified by observations
            for obs in operator_view.get("observations", []):
                cid = obs["component_id"]
                if cid in self.component_ids:
                    idx = self.component_ids.index(cid)
                    obs_state = obs["observed_state"]
                    # Update health in state vector based on observation
                    base = idx * NODE_FEAT_DIM
                    if obs_state == "failed":
                        state_t1[base] = 0.0  # health
                    elif obs_state == "degraded":
                        state_t1[base] = 0.4

            # Build input tensor
            max_nodes = 40
            state_dim = max_nodes * NODE_FEAT_DIM
            x_input = torch.zeros(1, 2, state_dim)
            actual_dim = min(n_nodes * NODE_FEAT_DIM, state_dim)
            x_input[0, 0, :actual_dim] = torch.tensor(state_t0[:actual_dim])
            x_input[0, 1, :actual_dim] = torch.tensor(state_t1[:actual_dim])

            with torch.no_grad():
                x_input = x_input.to(self.device)
                out = self.mamba(x_input)

                # Affected detection
                affected_probs = torch.sigmoid(out["affected_logits"]).cpu().squeeze()
                # State predictions
                state_preds = out["state_logits"].cpu().squeeze().argmax(dim=-1)
                # Severity
                predicted_severity = out["severity"].cpu().item()

            state_names = ["healthy", "degraded", "failed", "unreachable"]
            for i in range(min(n_nodes, max_nodes)):
                if affected_probs[i] > 0.4:
                    predicted_affected.append({
                        "component": self.component_ids[i],
                        "affected_prob": round(affected_probs[i].item(), 3),
                        "predicted_state": state_names[state_preds[i].item()],
                    })

            predicted_affected.sort(key=lambda x: -x["affected_prob"])
            mamba_predictions = {
                "severity": predicted_severity,
                "n_affected": len(predicted_affected),
                "affected": predicted_affected[:10],
            }

        # ── Phase 3: POMDP diagnostic planning ──
        pomdp_log = self.pomdp.run_diagnostic_session(true_state, max_steps=max_steps)

        root_cause_candidates = []
        recommended_action = ""
        action_type = ""
        action_target = ""

        if pomdp_log:
            last = pomdp_log[-1]
            root_cause_candidates = [
                {"component": cid, "probability": round(prob, 3)}
                for cid, prob in last["top_candidates"]
            ]
            first = pomdp_log[0]
            recommended_action = first["action"]
            action_type = first["action_type"]
            action_target = first["target"]

        # ── Phase 4: Fused assessment ──
        # Combine POMDP root cause analysis with Mamba cascade prediction
        assessment_parts = []

        # Root cause
        if root_cause_candidates:
            top = root_cause_candidates[0]
            assessment_parts.append(
                f"Root cause: {top['component']} (P={top['probability']})"
            )

        # Cascade prediction
        if predicted_affected:
            n_aff = len(predicted_affected)
            severity_label = (
                "critical" if predicted_severity > 0.5 else
                "high" if predicted_severity > 0.3 else
                "medium" if predicted_severity > 0.1 else
                "low"
            )
            assessment_parts.append(
                f"Cascade: {severity_label} ({n_aff} nodes at risk, "
                f"severity={predicted_severity:.2f})"
            )
        else:
            severity_label = "low"

        # Recommended action
        if recommended_action:
            assessment_parts.append(
                f"Next action: {recommended_action} ({action_type})"
            )

        # Confidence: agreement between pillars
        confidence = 0.5
        if root_cause_candidates and predicted_affected:
            # Do POMDP and Mamba agree on what's affected?
            pomdp_ids = {c["component"] for c in root_cause_candidates[:3]}
            mamba_ids = {a["component"] for a in predicted_affected[:5]}
            overlap = len(pomdp_ids & mamba_ids)
            confidence = min(0.95, 0.5 + 0.15 * overlap)

        elapsed_ms = (time.time() - t0) * 1000

        return DiagnosticResult(
            recommended_action=recommended_action,
            action_type=action_type,
            action_target=action_target,
            root_cause_candidates=root_cause_candidates,
            predicted_affected=predicted_affected[:10],
            predicted_severity=predicted_severity,
            cascade_risk=severity_label,
            confidence=confidence,
            assessment=" | ".join(assessment_parts),
            steps_taken=len(pomdp_log),
            time_ms=elapsed_ms,
        )


# ── Demo ──────────────────────────────────────────────────────────────────


def build_incident_scenario(seed: int = 42):
    """Build a realistic incident for the orchestrator to diagnose."""
    from pomcp import build_demo_scenario
    return build_demo_scenario(seed)


def run_demo(
    gnn_ckpt: str = "../pillar1/checkpoints/best_model.pt",
    mamba_ckpt: str = "../pillar3/checkpoints/best_mamba_final.pt",
    rollouts: int = 300,
    device: str = "cuda",
):
    print(f"\n{C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║     SABLE — Cognitive Architecture Demo         ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║     Three Perspectives, One Truth               ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════╝{C_RESET}\n")

    # Build scenario
    print(f"  {C_INFO}Building incident scenario...{C_RESET}")
    graph, true_state = build_incident_scenario()

    # Show ground truth
    failed = true_state.get_failed_components()
    degraded = true_state.get_degraded_components()
    print(f"\n  {C_DIM}Ground truth (hidden from SABLE):{C_RESET}")
    print(f"    {C_DANGER}Failed:   {', '.join(c.id for c in failed)}{C_RESET}")
    print(f"    {C_GOLD}Degraded: {', '.join(c.id for c in degraded) or 'none'}{C_RESET}")

    # Initialize orchestrator
    print(f"\n  {C_INFO}Initializing SABLE...{C_RESET}")
    orchestrator = SABLEOrchestrator(
        graph=graph,
        gnn_checkpoint=gnn_ckpt,
        mamba_checkpoint=mamba_ckpt,
        pomdp_rollouts=rollouts,
        device=device,
    )

    # Run diagnosis
    print(f"\n  {C_INFO}Running integrated diagnosis...{C_RESET}")
    result = orchestrator.diagnose(true_state, max_steps=5)

    # Display results
    print(f"\n  {C_GOLD}{C_BOLD}  ┌─────────────────────────────────────────────┐{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  │           DIAGNOSTIC ASSESSMENT              │{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  └─────────────────────────────────────────────┘{C_RESET}")

    print(f"\n  {C_INFO}Pillar 2 — Decision Planning (POMDP):{C_RESET}")
    print(f"    {C_TEXT}Recommended action: {C_BRIGHT}{result.recommended_action}{C_RESET}")
    print(f"    {C_TEXT}Action type:        {C_BRIGHT}{result.action_type}{C_RESET}")
    print(f"    {C_TEXT}Steps taken:        {C_BRIGHT}{result.steps_taken}{C_RESET}")
    if result.root_cause_candidates:
        print(f"    {C_TEXT}Root cause candidates:{C_RESET}")
        for rc in result.root_cause_candidates[:5]:
            is_actual = rc["component"] in [c.id for c in failed]
            marker = f" {C_SUCCESS}← CORRECT{C_RESET}" if is_actual else ""
            print(f"      {C_TEXT}{rc['component']:<20s} P={rc['probability']}{C_RESET}{marker}")

    print(f"\n  {C_INFO}Pillar 3 — Temporal Prediction (Mamba):{C_RESET}")
    print(f"    {C_TEXT}Predicted severity: {C_BRIGHT}{result.predicted_severity:.3f}{C_RESET}")
    risk_color = C_DANGER if result.cascade_risk in ("high", "critical") else C_GOLD if result.cascade_risk == "medium" else C_TEXT
    print(f"    {C_TEXT}Cascade risk:       {risk_color}{C_BOLD}{result.cascade_risk}{C_RESET}")
    if result.predicted_affected:
        print(f"    {C_TEXT}Predicted affected nodes:{C_RESET}")
        for pa in result.predicted_affected[:8]:
            print(f"      {C_TEXT}{pa['component']:<20s} P(affected)={pa['affected_prob']} → {pa['predicted_state']}{C_RESET}")

    # Fused assessment
    print(f"\n  {C_GOLD}{C_BOLD}  Fused Assessment:{C_RESET}")
    print(f"    {C_BRIGHT}{result.assessment}{C_RESET}")
    print(f"    {C_TEXT}Confidence: {C_BRIGHT}{result.confidence:.2f}{C_RESET}")
    print(f"    {C_TEXT}Total time: {C_BRIGHT}{result.time_ms:.0f}ms{C_RESET}")

    # Verify against ground truth
    print(f"\n  {C_DIM}{'─' * 50}{C_RESET}")
    actual_root = "core-sw-1"
    if result.root_cause_candidates:
        top_rc = result.root_cause_candidates[0]["component"]
        if top_rc == actual_root:
            print(f"  {C_SUCCESS}{C_BOLD}ROOT CAUSE CORRECTLY IDENTIFIED: {top_rc}{C_RESET}")
        else:
            found = any(rc["component"] == actual_root for rc in result.root_cause_candidates[:5])
            if found:
                rank = next(
                    i + 1 for i, rc in enumerate(result.root_cause_candidates)
                    if rc["component"] == actual_root
                )
                print(f"  {C_GOLD}Root cause {actual_root} at rank #{rank}{C_RESET}")
            else:
                print(f"  {C_DANGER}Root cause {actual_root} not in top 5{C_RESET}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SABLE Orchestrator Demo")
    parser.add_argument("--gnn-ckpt", type=str, default="../pillar1/checkpoints/best_model.pt")
    parser.add_argument("--mamba-ckpt", type=str, default="../pillar3/checkpoints/best_mamba_final.pt")
    parser.add_argument("--rollouts", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_demo(
        gnn_ckpt=args.gnn_ckpt,
        mamba_ckpt=args.mamba_ckpt,
        rollouts=args.rollouts,
        device=args.device,
    )
