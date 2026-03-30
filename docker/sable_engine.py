"""
SABLE Engine — Single entry point for all inference.
Wraps the three-pillar fusion + temporal chain into one clean interface.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Resolve source paths — works both locally (../pillar1) and in Docker (engine/pillar1)
_here = Path(__file__).parent
_engine_dir = _here / "engine"
if _engine_dir.exists():
    # Docker container: build.sh copied source into engine/
    sys.path.insert(0, str(_engine_dir))
    sys.path.insert(0, str(_engine_dir / "fusion"))
    sys.path.insert(0, str(_engine_dir / "pillar1"))
    sys.path.insert(0, str(_engine_dir / "pillar2"))
    sys.path.insert(0, str(_engine_dir / "pillar3"))
    sys.path.insert(0, str(_engine_dir / "sim"))
else:
    # Local dev: import from real source
    _root = _here.parent
    sys.path.insert(0, str(_root))
    sys.path.insert(0, str(_root / "fusion"))
    sys.path.insert(0, str(_root / "pillar1"))
    sys.path.insert(0, str(_root / "pillar2"))
    sys.path.insert(0, str(_root / "pillar3"))
    sys.path.insert(0, str(_root / "adapters"))

from sable_sim.core.states import N_STATES, STATE_NAMES
from temporal_chain import TemporalChainFusion, TemporalState
from staged_fusion_v3 import SharpRoutedFusion
from generate_temporal_data import NODE_FEAT_DIM
from shared_latent_space import GNN_DIM, POMDP_DIM


class SableEngine:
    """SABLE reasoning engine. Stateful — maintains temporal context across infer() calls."""

    def __init__(self, device: str = "cuda"):
        # Verify CUDA actually works (not just available — sm_120 may not be supported)
        use_cuda = False
        if device == "cuda" and torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda")
                use_cuda = True
            except RuntimeError:
                pass
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = None
        self.temporal_state = None
        self.cycle = 0
        self.n_nodes = 0
        self.history = []  # per-tick prediction history for the timeline
        self.MAX_HISTORY = 1000
        self.lora_active = False

    def load_checkpoints(self, checkpoint_dir: str = "checkpoints"):
        """Load fusion + temporal chain + optional LoRA adapter weights."""
        ckpt_dir = Path(checkpoint_dir)

        # Step 1: Build base fusion
        base = SharpRoutedFusion(mamba_dim=NODE_FEAT_DIM)
        fusion_path = ckpt_dir / "fusion.pt"
        if fusion_path.exists():
            ckpt = torch.load(fusion_path, weights_only=False, map_location=self.device)
            base.load_state_dict(ckpt["model_state_dict"])

        # Step 2: Wrap with temporal chain
        model = TemporalChainFusion(base)
        temporal_path = ckpt_dir / "temporal.pt"
        if temporal_path.exists():
            ckpt = torch.load(temporal_path, weights_only=False, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)

        # Step 3: Apply LoRA to the FULL model (fusion + temporal chain)
        lora_path = ckpt_dir / "lora_adapter.pt"
        if lora_path.exists():
            from lora_finetune import apply_lora
            lora_ckpt = torch.load(lora_path, weights_only=False, map_location=self.device)
            apply_lora(model, rank=lora_ckpt["rank"], alpha=lora_ckpt["alpha"])
            # Load trained LoRA weights into the full model
            model_state = model.state_dict()
            loaded = 0
            for k, v in lora_ckpt["lora_state_dict"].items():
                if k in model_state:
                    model_state[k] = v
                    loaded += 1
            model.load_state_dict(model_state)
            self.lora_active = True
            print(f"  LoRA adapter loaded ({loaded}/{len(lora_ckpt['lora_state_dict'])} keys, "
                  f"rank={lora_ckpt['rank']}, F1={lora_ckpt.get('best_macro_f1', 0):.3f})")

        for p in model.parameters():
            p.requires_grad = False

        model.eval().to(self.device)
        self.model = model
        return True

    def toggle_lora(self, enabled: bool):
        """Toggle LoRA adapter on/off at runtime. Instant, no reload."""
        from lora_finetune import set_lora_enabled
        set_lora_enabled(self.model, enabled)
        self.lora_active = enabled

    def reset_state(self, n_nodes: int):
        """Reset temporal state for a new scenario."""
        self.n_nodes = n_nodes
        self.temporal_state = TemporalState.cold_start(n_nodes, device=self.device)
        self.cycle = 0
        self.history = []

    @torch.no_grad()
    def infer(self, gnn: torch.Tensor, pomdp: torch.Tensor, mamba: torch.Tensor,
              ground_truth: torch.Tensor | None = None,
              mc_samples: int = 0) -> dict:
        """Run one inference cycle. Temporal state updates automatically.

        Args:
            gnn:   (1, N, GNN_DIM) - GNN node embeddings
            pomdp: (1, N, POMDP_DIM) - POMDP belief vectors
            mamba: (1, N, NODE_FEAT_DIM) - Mamba state features
            ground_truth: (N,) long - optional, for accuracy tracking
            mc_samples: int - if > 0, run MC dropout with this many forward
                         passes for variance-based confidence. The standard
                         eval-mode pass still runs for routing/transition/
                         temporal state. MC adds ~5ms per sample.
                         0 = standard single-pass inference (default).

        Returns:
            dict with per-node predictions, confidence, routing, transitions.
            When mc_samples > 0, mc_agreement and mc_variance fields are added.
        """
        gnn = gnn.to(self.device)
        pomdp = pomdp.to(self.device)
        mamba = mamba.to(self.device)

        # Standard eval-mode forward pass (always runs)
        # Provides routing, transition, temporal state update
        out = self.model(gnn, pomdp, mamba, temporal_state=self.temporal_state)
        logits = out["revised_logits"][0]  # (N, N_STATES)
        probs = torch.softmax(logits, dim=-1)  # (N, N_STATES)
        preds = probs.argmax(dim=-1)  # (N,)
        transition = torch.softmax(out["transition"][0], dim=-1)  # (N, 3)
        route_weights = out["route_weights"][0]  # (N, 4)

        # Confidence: either standard (max softmax) or MC dropout (variance-based)
        mc_meta = None
        if mc_samples > 0:
            confidence, mc_meta = self._mc_dropout_confidence(
                gnn, pomdp, mamba, preds, mc_samples)
        else:
            confidence = probs.max(dim=-1).values  # (N,)

        # Class counts
        class_counts = {STATE_NAMES[c]: int((preds == c).sum().item()) for c in range(N_STATES)}

        # Per-class routing (which expert gets most traffic per class)
        expert_names = ["GNN", "POMDP", "Mamba", "Fusion"]
        routing = {}
        for c in range(N_STATES):
            mask = preds == c
            if mask.sum() > 0:
                avg_weights = route_weights[mask].mean(dim=0)
                top = expert_names[avg_weights.argmax().item()]
                routing[STATE_NAMES[c]] = {
                    "expert": top,
                    "weight": float(avg_weights.max().item()),
                    "all_weights": [float(w) for w in avg_weights.tolist()],
                }

        # Node details
        nodes = []
        for i in range(self.n_nodes):
            node = {
                "id": i,
                "state": STATE_NAMES[preds[i].item()],
                "state_idx": int(preds[i].item()),
                "confidence": float(confidence[i].item()),
                "probs": {STATE_NAMES[c]: float(probs[i, c].item()) for c in range(N_STATES)},
                "transition": {
                    "improving": float(transition[i, 0].item()),
                    "stable": float(transition[i, 1].item()),
                    "deteriorating": float(transition[i, 2].item()),
                },
                "trend": ["improving", "stable", "deteriorating"][transition[i].argmax().item()],
                "routing": {expert_names[j]: float(route_weights[i, j].item()) for j in range(4)},
            }
            if mc_meta:
                node["mc_agreement"] = float(mc_meta["agreement"][i].item())
                node["mc_variance"] = float(mc_meta["variance"][i].item())
            nodes.append(node)

        # Accuracy if ground truth provided
        accuracy = None
        if ground_truth is not None:
            gt = ground_truth.to(self.device)
            accuracy = float((preds == gt).float().mean().item())

        # Build tick record
        tick_record = {
            "cycle": self.cycle,
            "nodes": nodes,
            "class_counts": class_counts,
            "routing": routing,
            "avg_confidence": float(confidence.mean().item()),
            "min_confidence": {
                "node": int(confidence.argmin().item()),
                "value": float(confidence.min().item()),
            },
            "max_confidence": {
                "node": int(confidence.argmax().item()),
                "value": float(confidence.max().item()),
            },
            "accuracy": accuracy,
        }

        if mc_meta:
            tick_record["mc_samples"] = mc_samples
            tick_record["mc_avg_agreement"] = float(mc_meta["agreement"].mean().item())
            tick_record["mc_avg_variance"] = float(mc_meta["variance"].mean().item())

        # Store history
        self.history.append({
            "cycle": self.cycle,
            "predictions": preds.cpu().tolist(),
            "ground_truth": ground_truth.tolist() if ground_truth is not None else None,
        })
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]

        # Update temporal state
        self.temporal_state.update(
            out["revised_logits"].detach(),
            out["confidence"].detach(),
            out.get("z_fused").detach() if out.get("z_fused") is not None else None,
        )
        self.cycle += 1

        return tick_record

    def _mc_dropout_confidence(
        self, gnn: torch.Tensor, pomdp: torch.Tensor, mamba: torch.Tensor,
        eval_preds: torch.Tensor, n_samples: int,
    ) -> tuple[torch.Tensor, dict]:
        """Run MC dropout forward passes for variance-based confidence.

        Enables dropout at inference time, runs n_samples passes, measures
        how much the predictions vary. High variance = low confidence.

        Args:
            gnn, pomdp, mamba: input tensors (already on device)
            eval_preds: (N,) predictions from the standard eval pass
            n_samples: number of MC forward passes

        Returns:
            confidence: (N,) tensor, 0-1 per node
            mc_meta: dict with agreement and variance tensors
        """
        # Enable dropout for stochastic forward passes
        self.model.train()

        all_probs = []
        all_preds = []
        for _ in range(n_samples):
            out = self.model(gnn, pomdp, mamba, temporal_state=self.temporal_state)
            logits = out["revised_logits"][0]  # (N, N_STATES)
            p = torch.softmax(logits, dim=-1)
            all_probs.append(p)
            all_preds.append(p.argmax(dim=-1))

        # Back to eval mode
        self.model.eval()

        # Stack: (n_samples, N, N_STATES)
        probs_stack = torch.stack(all_probs)
        preds_stack = torch.stack(all_preds)  # (n_samples, N)

        # Variance of predicted probabilities per node (mean across states)
        prob_variance = probs_stack.var(dim=0).mean(dim=-1)  # (N,)

        # Agreement: fraction of MC samples that agree with the eval prediction
        agreement = (preds_stack == eval_preds.unsqueeze(0)).float().mean(dim=0)  # (N,)

        # Confidence: combine agreement and inverse variance
        # High agreement + low variance = high confidence
        # Scale variance to [0, 1] range (max theoretical variance for uniform 5-class = 0.04)
        norm_variance = (prob_variance / 0.04).clamp(0, 1)
        confidence = (0.6 * agreement + 0.4 * (1.0 - norm_variance)).clamp(0, 1)

        return confidence, {
            "agreement": agreement,
            "variance": prob_variance,
        }

    def get_node_report(self, node_idx: int) -> dict:
        """Get detailed report for a specific node including trajectory."""
        if not self.history:
            return {"error": "No inference cycles yet"}

        trajectory = []
        for h in self.history:
            entry = {"cycle": h["cycle"], "prediction": STATE_NAMES[h["predictions"][node_idx]]}
            if h["ground_truth"] is not None:
                entry["truth"] = STATE_NAMES[h["ground_truth"][node_idx]]
                entry["correct"] = h["predictions"][node_idx] == h["ground_truth"][node_idx]
            trajectory.append(entry)

        # Get current state from latest tick
        latest = self.history[-1]
        return {
            "node_id": node_idx,
            "current_state": STATE_NAMES[latest["predictions"][node_idx]],
            "trajectory": trajectory,
            "n_state_changes": sum(
                1 for i in range(1, len(trajectory))
                if trajectory[i]["prediction"] != trajectory[i-1]["prediction"]
            ),
        }

    def get_recommendations(self) -> dict:
        """Generate actionable fix recommendations from current state.

        Returns prioritized actions based on node states, trajectories,
        and cascade structure.
        """
        if not self.history or len(self.history) < 2:
            return {"actions": [], "summary": "Insufficient data — need at least 2 inference cycles."}

        latest = self.history[-1]
        preds = latest["predictions"]

        # Identify problem nodes by severity
        failed = []
        unreachable = []
        degraded = []
        oscillating = []

        for i in range(self.n_nodes):
            state = preds[i]
            # Build trajectory for this node
            traj = [h["predictions"][i] for h in self.history]
            n_changes = sum(1 for j in range(1, len(traj)) if traj[j] != traj[j-1])
            first_bad = next((j for j, s in enumerate(traj) if s != 0), len(traj))

            node_info = {
                "node": i,
                "state": STATE_NAMES[state],
                "first_affected_tick": first_bad,
                "n_state_changes": n_changes,
                "ticks_in_current_state": sum(1 for j in range(len(traj)-1, -1, -1) if traj[j] == state),
            }

            if state == 2: failed.append(node_info)
            elif state == 3: unreachable.append(node_info)
            elif state == 1: degraded.append(node_info)
            elif state == 4: oscillating.append(node_info)

        # Sort by first_affected_tick — earliest failures are likely root causes
        failed.sort(key=lambda x: x["first_affected_tick"])
        unreachable.sort(key=lambda x: x["first_affected_tick"])

        # Build prioritized action list
        actions = []
        priority = 1

        # Root cause identification: earliest failed node
        if failed:
            root = failed[0]
            actions.append({
                "priority": priority,
                "action": "INVESTIGATE ROOT CAUSE",
                "target": f"Node {root['node']:02d}",
                "reason": f"First failure at tick {root['first_affected_tick']}. "
                          f"{len(failed)} total failed nodes likely depend on this.",
                "recommendation": "Restart service, check hardware, review recent changes.",
            })
            priority += 1

        # Failed nodes after root
        for node_info in failed[1:]:
            actions.append({
                "priority": priority,
                "action": "RESTART/RECOVER",
                "target": f"Node {node_info['node']:02d}",
                "reason": f"Failed since tick {node_info['first_affected_tick']}. "
                          f"May recover if root cause (Node {failed[0]['node']:02d}) is fixed.",
                "recommendation": "Wait for root cause fix, then verify. Manual restart if no auto-recovery.",
            })
            priority += 1

        # Unreachable — connectivity issue
        for node_info in unreachable:
            actions.append({
                "priority": priority,
                "action": "CHECK CONNECTIVITY",
                "target": f"Node {node_info['node']:02d}",
                "reason": f"Unreachable since tick {node_info['first_affected_tick']}. "
                          f"Likely isolated by upstream failures.",
                "recommendation": "Verify network path. May auto-resolve when upstream nodes recover.",
            })
            priority += 1

        # Oscillating — stabilize
        for node_info in oscillating:
            actions.append({
                "priority": priority,
                "action": "STABILIZE",
                "target": f"Node {node_info['node']:02d}",
                "reason": f"Flapping between states ({node_info['n_state_changes']} transitions). "
                          f"Unstable — likely a load balancer or service with intermittent dependency.",
                "recommendation": "Drain traffic, isolate from pool, investigate flapping dependency.",
            })
            priority += 1

        # Degraded — monitor
        for node_info in degraded:
            actions.append({
                "priority": priority,
                "action": "MONITOR/INVESTIGATE",
                "target": f"Node {node_info['node']:02d}",
                "reason": f"Degraded since tick {node_info['first_affected_tick']}. "
                          f"Performance impacted but operational.",
                "recommendation": "Check resource utilization. May worsen — watch trajectory.",
            })
            priority += 1

        # Impact summary
        total_affected = len(failed) + len(unreachable) + len(degraded) + len(oscillating)
        summary_parts = []
        if failed: summary_parts.append(f"{len(failed)} failed")
        if unreachable: summary_parts.append(f"{len(unreachable)} unreachable")
        if degraded: summary_parts.append(f"{len(degraded)} degraded")
        if oscillating: summary_parts.append(f"{len(oscillating)} oscillating")

        if not summary_parts:
            summary = "All nodes healthy. No action required."
        else:
            summary = (f"{total_affected}/{self.n_nodes} nodes affected: "
                      + ", ".join(summary_parts) + ". "
                      + (f"Root cause likely Node {failed[0]['node']:02d} (first failure at tick {failed[0]['first_affected_tick']})."
                         if failed else "No hard failures — monitor degraded nodes."))

        return {
            "summary": summary,
            "total_affected": total_affected,
            "actions": actions,
            "root_cause": failed[0]["node"] if failed else None,
        }

    def get_summary(self) -> dict:
        """Get full engine summary."""
        return {
            "n_nodes": self.n_nodes,
            "cycle": self.cycle,
            "n_states": N_STATES,
            "state_names": STATE_NAMES,
            "device": str(self.device),
            "model_loaded": self.model is not None,
            "temporal_active": self.temporal_state is not None and self.temporal_state.cycle > 0,
        }
