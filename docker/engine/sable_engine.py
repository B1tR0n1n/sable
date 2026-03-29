"""
SABLE Engine — Single entry point for all inference.
Wraps the three-pillar fusion + temporal chain into one clean interface.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Engine needs access to the model definitions
ENGINE_DIR = Path(__file__).parent / "engine"
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(ENGINE_DIR / "fusion"))
sys.path.insert(0, str(ENGINE_DIR / "pillar1"))
sys.path.insert(0, str(ENGINE_DIR / "pillar2"))
sys.path.insert(0, str(ENGINE_DIR / "pillar3"))
sys.path.insert(0, str(ENGINE_DIR / "sim"))

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

    def load_checkpoints(self, checkpoint_dir: str = "checkpoints"):
        """Load fusion + temporal chain weights."""
        ckpt_dir = Path(checkpoint_dir)

        # Build model
        base = SharpRoutedFusion(mamba_dim=NODE_FEAT_DIM)
        fusion_path = ckpt_dir / "fusion.pt"
        if fusion_path.exists():
            ckpt = torch.load(fusion_path, weights_only=False, map_location=self.device)
            base.load_state_dict(ckpt["model_state_dict"])

        for p in base.parameters():
            p.requires_grad = False

        model = TemporalChainFusion(base)
        temporal_path = ckpt_dir / "temporal.pt"
        if temporal_path.exists():
            ckpt = torch.load(temporal_path, weights_only=False, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])

        model.eval().to(self.device)
        self.model = model
        return True

    def reset_state(self, n_nodes: int):
        """Reset temporal state for a new scenario."""
        self.n_nodes = n_nodes
        self.temporal_state = TemporalState.cold_start(n_nodes, device=self.device)
        self.cycle = 0
        self.history = []

    @torch.no_grad()
    def infer(self, gnn: torch.Tensor, pomdp: torch.Tensor, mamba: torch.Tensor,
              ground_truth: torch.Tensor | None = None) -> dict:
        """Run one inference cycle. Temporal state updates automatically.

        Args:
            gnn:   (1, N, GNN_DIM) — GNN node embeddings
            pomdp: (1, N, POMDP_DIM) — POMDP belief vectors
            mamba: (1, N, NODE_FEAT_DIM) — Mamba state features
            ground_truth: (N,) long — optional, for accuracy tracking

        Returns:
            dict with per-node predictions, confidence, routing, transitions
        """
        gnn = gnn.to(self.device)
        pomdp = pomdp.to(self.device)
        mamba = mamba.to(self.device)

        out = self.model(gnn, pomdp, mamba, temporal_state=self.temporal_state)

        # Extract predictions
        logits = out["revised_logits"][0]  # (N, N_STATES)
        probs = torch.softmax(logits, dim=-1)  # (N, N_STATES)
        preds = probs.argmax(dim=-1)  # (N,)
        confidence = out["confidence"][0].squeeze(-1)  # (N,)
        transition = torch.softmax(out["transition"][0], dim=-1)  # (N, 3)
        route_weights = out["route_weights"][0]  # (N, 4)

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

        # Store history
        self.history.append({
            "cycle": self.cycle,
            "predictions": preds.cpu().tolist(),
            "ground_truth": ground_truth.tolist() if ground_truth is not None else None,
        })

        # Update temporal state
        self.temporal_state.update(
            out["revised_logits"].detach(),
            out["confidence"].detach(),
            out.get("z_fused").detach() if out.get("z_fused") is not None else None,
        )
        self.cycle += 1

        return tick_record

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
