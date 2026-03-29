#!/usr/bin/env python3
"""
Project PARALLAX — GNN Link Suggestion Server
================================================
Lightweight HTTP server exposing the trained SableGNN as a tool.
Runs on the 5070 alongside the model.

Endpoints:
    POST /suggest        — Suggest links for a thought_id
    POST /score          — Score a specific pair of thoughts
    POST /contradictions — Find contradiction candidates
    GET  /health         — Health check
    GET  /stats          — Model and graph stats

Usage:
    python gnn_server.py                          # Default port 5070
    python gnn_server.py --port 5071              # Custom port
    python gnn_server.py --device cpu             # Force CPU

Requires: ml-env with torch, torch_geometric, httpx
"""

import argparse
import json
import sys
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cortex_gnn_model import SableGNN

# ── Configuration ──────────────────────────────────────────────────────────

RELATION_TYPES = [
    "supports", "contradicts", "elaborates",
    "depends_on", "caused_by", "related", "supersedes",
]

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_DANGER = "\033[38;2;201;74;58m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


# ── GNN Engine ─────────────────────────────────────────────────────────────


class GNNEngine:
    """Loaded GNN model with precomputed node embeddings."""

    def __init__(self, data_path: str, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.data = torch.load(data_path, weights_only=False)
        self.node_ids = self.data.node_ids
        self.id_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.num_nodes = len(self.node_ids)

        # Build existing edge set for filtering
        ei = self.data.edge_index
        self.existing_edges = set()
        for i in range(ei.size(1)):
            self.existing_edges.add((ei[0, i].item(), ei[1, i].item()))

        # Load model
        ckpt = torch.load(checkpoint_path, weights_only=False)
        mc = ckpt["config"]
        self.model = SableGNN(
            in_dim=mc["in_dim"], hidden_dim=mc["hidden_dim"],
            edge_dim=mc["edge_dim"], num_layers=mc["num_layers"],
            heads=mc["heads"], dropout=mc["dropout"],
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.hidden_dim = mc["hidden_dim"]
        self.epoch = ckpt["epoch"]
        self.val_loss = ckpt["val_loss"]

        # Precompute node embeddings
        with torch.no_grad():
            self.node_emb = self.model.encode(
                self.data.x.to(device),
                self.data.edge_index.to(device),
                self.data.edge_attr.to(device),
            )

    def suggest_links(self, thought_id: str, top_k: int = 20, min_confidence: float = 0.3) -> list[dict]:
        """Suggest new links for a thought based on GNN structural reasoning."""
        src_idx = self.id_to_idx.get(thought_id)
        if src_idx is None:
            return []

        # Score this node against all others
        src_indices = torch.full((self.num_nodes,), src_idx, dtype=torch.long, device=self.device)
        tgt_indices = torch.arange(self.num_nodes, dtype=torch.long, device=self.device)
        pair_ei = torch.stack([src_indices, tgt_indices])

        with torch.no_grad():
            # Link existence score
            link_logits = self.model.predict_link(self.node_emb, pair_ei)
            link_probs = torch.sigmoid(link_logits).cpu()

            # Link type prediction
            type_logits = self.model.predict_link_type(self.node_emb, pair_ei)
            type_probs = torch.softmax(type_logits, dim=-1).cpu()
            type_preds = type_logits.argmax(dim=-1).cpu()
            type_confidences = type_probs.max(dim=-1).values.cpu()

            # Contradiction score
            contra_logits = self.model.predict_contradiction(self.node_emb, pair_ei)
            contra_probs = torch.sigmoid(contra_logits).cpu()

        results = []
        for i in range(self.num_nodes):
            if i == src_idx:
                continue
            # Skip existing edges (both directions)
            if (src_idx, i) in self.existing_edges or (i, src_idx) in self.existing_edges:
                continue

            link_prob = link_probs[i].item()
            if link_prob < min_confidence:
                continue

            pred_type = RELATION_TYPES[type_preds[i].item()]
            type_conf = type_confidences[i].item()
            contra_prob = contra_probs[i].item()

            # Override type if contradiction probability is high
            if contra_prob > 0.5:
                pred_type = "contradicts"
                type_conf = contra_prob

            results.append({
                "target_id": self.node_ids[i],
                "link_probability": round(link_prob, 4),
                "predicted_relation": pred_type,
                "type_confidence": round(type_conf, 4),
                "contradiction_probability": round(contra_prob, 4),
            })

        # Sort by link probability
        results.sort(key=lambda x: x["link_probability"], reverse=True)
        return results[:top_k]

    def score_pair(self, source_id: str, target_id: str) -> dict | None:
        """Score a specific thought pair."""
        src_idx = self.id_to_idx.get(source_id)
        tgt_idx = self.id_to_idx.get(target_id)
        if src_idx is None or tgt_idx is None:
            return None

        pair_ei = torch.tensor([[src_idx], [tgt_idx]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            link_logit = self.model.predict_link(self.node_emb, pair_ei)
            link_prob = torch.sigmoid(link_logit).item()

            type_logits = self.model.predict_link_type(self.node_emb, pair_ei)
            type_probs = torch.softmax(type_logits, dim=-1).cpu().squeeze()
            pred_type = RELATION_TYPES[type_probs.argmax().item()]
            type_conf = type_probs.max().item()

            contra_logit = self.model.predict_contradiction(self.node_emb, pair_ei)
            contra_prob = torch.sigmoid(contra_logit).item()

        return {
            "source_id": source_id,
            "target_id": target_id,
            "link_probability": round(link_prob, 4),
            "predicted_relation": pred_type,
            "type_confidence": round(type_conf, 4),
            "contradiction_probability": round(contra_prob, 4),
            "type_distribution": {
                RELATION_TYPES[i]: round(type_probs[i].item(), 4)
                for i in range(len(RELATION_TYPES))
            },
        }

    def find_contradictions(self, top_k: int = 20, threshold: float = 0.3) -> list[dict]:
        """Find top contradiction candidates across all edges."""
        ei = self.data.edge_index.to(self.device)

        with torch.no_grad():
            contra_logits = self.model.predict_contradiction(self.node_emb, ei)
            contra_probs = torch.sigmoid(contra_logits).cpu()

        edge_labels = self.data.edge_labels.cpu() if hasattr(self.data, "edge_labels") else None

        sorted_indices = torch.argsort(contra_probs, descending=True)

        results = []
        for idx in sorted_indices:
            prob = contra_probs[idx].item()
            if prob < threshold:
                break
            if len(results) >= top_k:
                break

            src_idx = self.data.edge_index[0, idx].item()
            tgt_idx = self.data.edge_index[1, idx].item()
            current_type = RELATION_TYPES[edge_labels[idx].item()] if edge_labels is not None else "unknown"

            results.append({
                "source_id": self.node_ids[src_idx],
                "target_id": self.node_ids[tgt_idx],
                "contradiction_probability": round(prob, 4),
                "current_relation": current_type,
                "is_known_contradiction": current_type == "contradicts",
            })

        return results

    def stats(self) -> dict:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.data.edge_index.size(1),
            "hidden_dim": self.hidden_dim,
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "checkpoint_epoch": self.epoch,
            "checkpoint_val_loss": round(self.val_loss, 4),
            "device": str(self.device),
            "existing_edges": len(self.existing_edges),
        }


# ── HTTP Server ────────────────────────────────────────────────────────────


class GNNHandler(BaseHTTPRequestHandler):
    engine: "GNNEngine | None" = None  # Set by server setup

    def log_message(self, format, *args):
        # Custom log format
        print(f"  {C_DIM}{self.client_address[0]}{C_RESET} {C_TEXT}{args[0]}{C_RESET}")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            self._send_json({"status": "ok", "model": "SableGNN"})

        elif path == "/stats":
            self._send_json(self.engine.stats())

        else:
            self._send_json({"error": f"Unknown endpoint: {path}"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path

        try:
            body = self._read_body()
        except ValueError as e:  # JSONDecodeError is a subclass of ValueError
            self._send_json({"error": f"Invalid JSON: {e}"}, 400)
            return

        if path == "/suggest":
            thought_id = body.get("thought_id")
            if not thought_id:
                self._send_json({"error": "thought_id required"}, 400)
                return

            top_k = body.get("top_k", 20)
            min_confidence = body.get("min_confidence", 0.3)

            t0 = time.time()
            results = self.engine.suggest_links(thought_id, top_k, min_confidence)
            elapsed = time.time() - t0

            self._send_json({
                "thought_id": thought_id,
                "suggestions": results,
                "count": len(results),
                "elapsed_ms": round(elapsed * 1000, 1),
            })

        elif path == "/score":
            source_id = body.get("source_id")
            target_id = body.get("target_id")
            if not source_id or not target_id:
                self._send_json({"error": "source_id and target_id required"}, 400)
                return

            result = self.engine.score_pair(source_id, target_id)
            if result is None:
                self._send_json({"error": "One or both thought IDs not found in graph"}, 404)
                return

            self._send_json(result)

        elif path == "/contradictions":
            top_k = body.get("top_k", 20)
            threshold = body.get("threshold", 0.3)
            results = self.engine.find_contradictions(top_k, threshold)
            self._send_json({
                "contradictions": results,
                "count": len(results),
            })

        else:
            self._send_json({"error": f"Unknown endpoint: {path}"}, 404)


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="SableGNN Link Suggestion Server")
    parser.add_argument("--port", type=int, default=5070)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--data", type=str, default="cortex_graph.pt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — GNN Server{C_RESET}")
    print(f"  {C_DIM}{'═' * 40}{C_RESET}\n")

    print(f"  {C_INFO}Loading model...{C_RESET}")
    engine = GNNEngine(args.data, args.checkpoint, args.device)
    stats = engine.stats()
    print(f"  {C_TEXT}nodes:  {C_BRIGHT}{stats['num_nodes']}{C_RESET}")
    print(f"  {C_TEXT}edges:  {C_BRIGHT}{stats['num_edges']}{C_RESET}")
    print(f"  {C_TEXT}params: {C_BRIGHT}{stats['model_params']:,}{C_RESET}")
    print(f"  {C_TEXT}device: {C_BRIGHT}{stats['device']}{C_RESET}")

    GNNHandler.engine = engine

    server = HTTPServer((args.host, args.port), GNNHandler)
    print(f"\n  {C_SUCCESS}{C_BOLD}Listening on {args.host}:{args.port}{C_RESET}")
    print(f"  {C_DIM}POST /suggest        — Suggest links for a thought")
    print(f"  POST /score          — Score a specific pair")
    print(f"  POST /contradictions — Find contradictions")
    print(f"  GET  /health         — Health check")
    print(f"  GET  /stats          — Model stats{C_RESET}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n  {C_DIM}Shutting down...{C_RESET}")
        server.server_close()


if __name__ == "__main__":
    main()
