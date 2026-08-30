#!/usr/bin/env python3
"""
Project PARALLAX — Contradiction Finder
==========================================
Uses the trained GNN's contradiction head to surface candidate
contradictions in the CORTEX knowledge graph.

Two modes:
  1. Score existing edges — which linked thoughts might actually contradict?
  2. Score unlinked pairs — which unlinked thoughts might be in tension?

Usage:
    python find_contradictions.py                     # Score existing edges
    python find_contradictions.py --unlinked --top 50 # Score unlinked pairs
    python find_contradictions.py --threshold 0.3     # Lower threshold = more candidates

Requires: ml-env with torch, torch_geometric, httpx
"""

import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.utils import negative_sampling

sys.path.insert(0, str(Path(__file__).parent))
from cortex_gnn_model import SableGNN

# ── Configuration ──────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lqpvskwevanpgywdksqu.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

UNKNOWN_CONTENT = "(unknown)"

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


def fetch_thought_content() -> dict[str, str]:
    """Fetch thought ID → content mapping from Supabase (read-only)."""
    import httpx

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    thoughts = {}
    offset = 0
    with httpx.Client(timeout=60) as client:
        while True:
            resp = client.get(
                f"{SUPABASE_URL}/rest/v1/thoughts",
                headers=headers,
                params={
                    "select": "id,content,metadata",
                    "order": "created_at.asc",
                    "offset": offset,
                    "limit": 1000,
                },
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for t in batch:
                thoughts[t["id"]] = t["content"][:200]
            offset += len(batch)
            if len(batch) < 1000:
                break
    return thoughts


def main():
    parser = argparse.ArgumentParser(description="Find contradiction candidates in CORTEX")
    parser.add_argument("--data", type=str, default="cortex_graph.pt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--top", type=int, default=30, help="Number of top candidates to show")
    parser.add_argument("--threshold", type=float, default=0.2, help="Min contradiction probability")
    parser.add_argument("--unlinked", action="store_true", help="Also score unlinked pairs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\n{C_GOLD}{C_BOLD}  PARALLAX — Contradiction Finder{C_RESET}")
    print(f"  {C_DIM}{'═' * 45}{C_RESET}\n")

    # Load model + data
    data = torch.load(args.data, weights_only=False)
    ckpt = torch.load(args.checkpoint, weights_only=False)
    mc = ckpt["config"]

    model = SableGNN(
        in_dim=mc["in_dim"], hidden_dim=mc["hidden_dim"],
        edge_dim=mc["edge_dim"], num_layers=mc["num_layers"],
        heads=mc["heads"], dropout=mc["dropout"],
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    node_ids = data.node_ids
    device = args.device

    print(f"  {C_TEXT}Fetching thought content...{C_RESET}")
    content_map = fetch_thought_content()

    with torch.no_grad():
        # Encode all nodes
        node_emb = model.encode(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
        )

        # ── Score existing edges ──
        print(f"\n  {C_INFO}Scoring existing edges for contradictions...{C_RESET}")
        ei = data.edge_index.to(device)
        contradiction_logits = model.predict_contradiction(node_emb, ei)
        contradiction_probs = torch.sigmoid(contradiction_logits).cpu()

        # Also get link type predictions to compare
        link_type_logits = model.predict_link_type(node_emb, ei)
        link_type_preds = link_type_logits.argmax(dim=-1).cpu()

        # Get existing edge labels
        edge_labels = data.edge_labels.cpu() if hasattr(data, 'edge_labels') else None

        relation_types = ["supports", "contradicts", "elaborates", "depends_on", "caused_by", "related", "supersedes"]

        # Sort by contradiction probability
        sorted_indices = torch.argsort(contradiction_probs, descending=True)

        print(f"\n  {C_GOLD}{C_BOLD}  Top Contradiction Candidates (Existing Edges){C_RESET}")
        print(f"  {C_DIM}{'─' * 70}{C_RESET}")

        shown = 0
        for idx in sorted_indices:
            prob = contradiction_probs[idx].item()
            if prob < args.threshold:
                break
            if shown >= args.top:
                break

            src_idx = data.edge_index[0, idx].item()
            tgt_idx = data.edge_index[1, idx].item()
            src_id = node_ids[src_idx]
            tgt_id = node_ids[tgt_idx]

            current_label = relation_types[edge_labels[idx].item()] if edge_labels is not None else "?"
            predicted_type = relation_types[link_type_preds[idx].item()]

            src_content = content_map.get(src_id, UNKNOWN_CONTENT)[:120]
            tgt_content = content_map.get(tgt_id, UNKNOWN_CONTENT)[:120]

            is_already_contradiction = current_label == "contradicts"
            marker = f"{C_SUCCESS}[KNOWN]{C_RESET}" if is_already_contradiction else f"{C_DANGER}[NEW?]{C_RESET}"

            print(f"\n  {marker} {C_BRIGHT}p={prob:.3f}{C_RESET}  current: {C_DIM}{current_label}{C_RESET}  predicted: {C_DIM}{predicted_type}{C_RESET}")
            print(f"    {C_TEXT}SRC: {src_content}{C_RESET}")
            print(f"    {C_TEXT}TGT: {tgt_content}{C_RESET}")
            shown += 1

        if shown == 0:
            print(f"  {C_DIM}No candidates above threshold {args.threshold}{C_RESET}")

        # ── Score unlinked pairs ──
        if args.unlinked:
            print(f"\n\n  {C_INFO}Scoring unlinked pairs for contradictions...{C_RESET}")
            print(f"  {C_DIM}(sampling {min(args.top * 100, 10000)} random unlinked pairs){C_RESET}")

            num_neg = min(args.top * 100, 10000)
            neg_ei = negative_sampling(
                data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=num_neg,
            ).to(device)

            neg_contradiction_logits = model.predict_contradiction(node_emb, neg_ei)
            neg_probs = torch.sigmoid(neg_contradiction_logits).cpu()
            neg_ei_cpu = neg_ei.cpu()

            sorted_neg = torch.argsort(neg_probs, descending=True)

            print(f"\n  {C_GOLD}{C_BOLD}  Top Contradiction Candidates (Unlinked Pairs){C_RESET}")
            print(f"  {C_DIM}{'─' * 70}{C_RESET}")

            shown = 0
            for idx in sorted_neg:
                prob = neg_probs[idx].item()
                if prob < args.threshold:
                    break
                if shown >= args.top:
                    break

                src_idx = neg_ei_cpu[0, idx].item()
                tgt_idx = neg_ei_cpu[1, idx].item()
                src_id = node_ids[src_idx]
                tgt_id = node_ids[tgt_idx]

                src_content = content_map.get(src_id, UNKNOWN_CONTENT)[:120]
                tgt_content = content_map.get(tgt_id, UNKNOWN_CONTENT)[:120]

                print(f"\n  {C_DANGER}[UNLINKED]{C_RESET} {C_BRIGHT}p={prob:.3f}{C_RESET}")
                print(f"    {C_TEXT}SRC: {src_content}{C_RESET}")
                print(f"    {C_TEXT}TGT: {tgt_content}{C_RESET}")
                shown += 1

            if shown == 0:
                print(f"  {C_DIM}No unlinked candidates above threshold {args.threshold}{C_RESET}")

    print()


if __name__ == "__main__":
    main()
