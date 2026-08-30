#!/usr/bin/env python3
"""
Project PARALLAX — CORTEX Graph Repair
=========================================
Uses SABLE's GNN to fix structural gaps in the knowledge graph.

Three repair operations:
  1. LINK ISOLATED THOUGHTS — 672 thoughts with zero connections.
     GNN scores each against all connected thoughts, creates top links.
  2. CREATE HIDDEN CONNECTIONS — GNN-predicted missing links between
     already-connected thoughts (fills structural gaps).
  3. FLAG CONTRADICTIONS — surfaces belief tensions for manual review.

IMPORTANT: This WRITES to CORTEX. Run with --dry-run first.

Usage:
    python cortex_repair.py --dry-run            # Preview changes only
    python cortex_repair.py --execute            # Apply changes
    python cortex_repair.py --execute --limit 50 # Apply first 50 repairs
"""

import argparse
import os
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "pillar1"))
from cortex_gnn_model import SableGNN

# ── Config ─────────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lqpvskwevanpgywdksqu.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

RELATION_TYPES = ["supports", "contradicts", "elaborates", "depends_on", "caused_by", "related", "supersedes"]

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


# ── Data Pull ──────────────────────────────────────────────────────────────


def pull_cortex():
    """Pull thoughts and links from CORTEX."""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Prefer": "return=representation",
    }
    client = httpx.Client(timeout=60)

    thoughts = []
    offset = 0
    while True:
        resp = client.get(
            f"{SUPABASE_URL}/rest/v1/thoughts", headers=headers,
            params={"select": "id,content,embedding,metadata,created_at",
                    "order": "created_at.asc", "offset": offset, "limit": 1000},
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        thoughts.extend(batch)
        offset += len(batch)
        if len(batch) < 1000:
            break

    links = []
    offset = 0
    while True:
        resp = client.get(
            f"{SUPABASE_URL}/rest/v1/thought_links", headers=headers,
            params={"select": "source_id,target_id,relation_type",
                    "offset": offset, "limit": 1000},
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        links.extend(batch)
        offset += len(batch)
        if len(batch) < 1000:
            break

    client.close()
    return thoughts, links


def insert_link(source_id: str, target_id: str, relation_type: str,
                confidence: float, context: str) -> bool:
    """Insert a link into CORTEX."""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    resp = httpx.post(
        f"{SUPABASE_URL}/rest/v1/thought_links",
        headers=headers,
        json={
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "confidence": confidence,
            "context": context,
        },
        timeout=30,
    )
    return resp.status_code in (200, 201, 204)


# ── GNN Scoring ───────────────────────────────────────────────────────────


def load_gnn(checkpoint_path: str, data_path: str, device: str = "cuda"):
    """Load GNN model and CORTEX graph data."""
    data = torch.load(data_path, weights_only=False)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    mc = ckpt["config"]
    model = SableGNN(
        in_dim=mc["in_dim"], hidden_dim=mc["hidden_dim"],
        edge_dim=mc["edge_dim"], num_layers=mc["num_layers"],
        heads=mc["heads"], dropout=mc["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, data


def score_pairs(model, data, pairs: list[tuple[int, int]], device: str = "cuda"):
    """Score a batch of node pairs using the GNN."""
    if not pairs:
        return [], [], []

    with torch.no_grad():
        node_emb = model.encode(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
        )

        src = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        tgt = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        pair_ei = torch.stack([src, tgt]).to(device)

        link_probs = torch.sigmoid(model.predict_link(node_emb, pair_ei)).cpu().numpy()
        type_preds = model.predict_link_type(node_emb, pair_ei).argmax(dim=-1).cpu().numpy()
        contra_probs = torch.sigmoid(model.predict_contradiction(node_emb, pair_ei)).cpu().numpy()

    return link_probs, type_preds, contra_probs


# ── Repair Operations ─────────────────────────────────────────────────────


def find_repairs(thoughts, links, model, data, device="cuda",
                 link_threshold=0.85, max_links_per_isolated=3):
    """Identify all repairs needed."""

    node_ids = data.node_ids
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    content_map = {t["id"]: t["content"][:120] for t in thoughts}

    # Build adjacency
    existing_edges = set()
    connected_nodes = set()
    for link in links:
        existing_edges.add((link["source_id"], link["target_id"]))
        existing_edges.add((link["target_id"], link["source_id"]))
        connected_nodes.add(link["source_id"])
        connected_nodes.add(link["target_id"])

    # Find isolated thoughts that are in the GNN's graph
    isolated = [nid for nid in node_ids if nid not in connected_nodes]
    connected = [nid for nid in node_ids if nid in connected_nodes]

    print(f"  {C_TEXT}Nodes in GNN graph: {C_BRIGHT}{len(node_ids)}{C_RESET}")
    print(f"  {C_TEXT}Connected:          {C_BRIGHT}{len(connected)}{C_RESET}")
    print(f"  {C_TEXT}Isolated:           {C_BRIGHT}{len(isolated)}{C_RESET}")

    repairs = {
        "isolated_links": [],     # New links for isolated thoughts
        "hidden_links": [],       # Missing links between connected thoughts
        "contradictions": [],     # Belief tensions to flag
    }

    # ── 1. Link isolated thoughts ──
    print(f"\n  {C_INFO}Scoring isolated thoughts against connected graph...{C_RESET}")
    t0 = time.time()

    for i, iso_id in enumerate(isolated):
        iso_idx = id_to_idx.get(iso_id)
        if iso_idx is None:
            continue

        # Score against all connected nodes
        pairs = [(iso_idx, id_to_idx[cid]) for cid in connected if cid in id_to_idx]
        if not pairs:
            continue

        # Batch score
        link_probs, type_preds, contra_probs = score_pairs(model, data, pairs, device)

        # Take top matches above threshold
        top_indices = np.argsort(link_probs)[::-1][:max_links_per_isolated]
        for idx in top_indices:
            prob = link_probs[idx]
            if prob < link_threshold:
                continue
            target_id = connected[idx] if idx < len(connected) else None
            if target_id is None:
                continue
            pred_type = RELATION_TYPES[type_preds[idx]]

            # Skip if link already exists
            if (iso_id, target_id) in existing_edges:
                continue

            repairs["isolated_links"].append({
                "source_id": iso_id,
                "target_id": target_id,
                "relation_type": pred_type,
                "probability": round(float(prob), 4),
                "source_content": content_map.get(iso_id, "?"),
                "target_content": content_map.get(target_id, "?"),
                "context": f"GNN-predicted link (P={prob:.3f}), auto-repair of isolated thought",
            })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    {C_DIM}{i+1}/{len(isolated)} ({elapsed:.1f}s){C_RESET}")

    elapsed = time.time() - t0
    print(f"  {C_TEXT}Found {C_BRIGHT}{len(repairs['isolated_links'])}{C_TEXT} links for isolated thoughts ({elapsed:.1f}s){C_RESET}")

    # ── 2. Hidden connections between connected thoughts ──
    print(f"\n  {C_INFO}Finding hidden connections in connected graph...{C_RESET}")
    rng = np.random.default_rng(42)
    sample_size = min(30000, len(connected) * (len(connected) - 1) // 2)

    sampled_pairs = []
    sampled_seen = set()
    for _ in range(sample_size * 2):
        if len(sampled_pairs) >= sample_size:
            break
        i = rng.integers(0, len(connected))
        j = rng.integers(0, len(connected))
        cid_i = connected[i]
        cid_j = connected[j]
        if cid_i == cid_j or (cid_i, cid_j) in existing_edges or (cid_i, cid_j) in sampled_seen:
            continue
        idx_i = id_to_idx.get(cid_i)
        idx_j = id_to_idx.get(cid_j)
        if idx_i is None or idx_j is None:
            continue
        sampled_pairs.append((idx_i, idx_j))
        sampled_seen.add((cid_i, cid_j))

    if sampled_pairs:
        link_probs, type_preds, contra_probs = score_pairs(model, data, sampled_pairs, device)
        top_indices = np.argsort(link_probs)[::-1][:50]

        for idx in top_indices:
            prob = link_probs[idx]
            if prob < 0.95:  # Higher threshold for connected-to-connected
                continue
            src_gnn_idx, tgt_gnn_idx = sampled_pairs[idx]
            src_id = node_ids[src_gnn_idx]
            tgt_id = node_ids[tgt_gnn_idx]
            pred_type = RELATION_TYPES[type_preds[idx]]
            contra = contra_probs[idx]

            if contra > 0.5:
                pred_type = "contradicts"

            repairs["hidden_links"].append({
                "source_id": src_id,
                "target_id": tgt_id,
                "relation_type": pred_type,
                "probability": round(float(prob), 4),
                "contradiction_prob": round(float(contra), 4),
                "source_content": content_map.get(src_id, "?"),
                "target_content": content_map.get(tgt_id, "?"),
                "context": f"GNN-predicted hidden link (P={prob:.3f}), structural gap repair",
            })

    print(f"  {C_TEXT}Found {C_BRIGHT}{len(repairs['hidden_links'])}{C_TEXT} hidden connections{C_RESET}")

    # ── 3. Contradictions ──
    print(f"\n  {C_INFO}Scanning for contradictions...{C_RESET}")
    with torch.no_grad():
        node_emb = model.encode(
            data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device),
        )
        all_contra = torch.sigmoid(
            model.predict_contradiction(node_emb, data.edge_index.to(device))
        ).cpu().numpy()

    edge_labels = data.edge_labels.cpu().numpy() if hasattr(data, "edge_labels") else None

    top_contra = np.argsort(all_contra)[::-1][:20]
    for idx in top_contra:
        prob = all_contra[idx]
        if prob < 0.5:
            continue
        s_idx = data.edge_index[0, idx].item()
        t_idx = data.edge_index[1, idx].item()
        current_type = RELATION_TYPES[edge_labels[idx]] if edge_labels is not None else "?"
        is_known = current_type == "contradicts"

        if not is_known:  # Only flag NEW contradictions
            repairs["contradictions"].append({
                "source_id": node_ids[s_idx],
                "target_id": node_ids[t_idx],
                "contradiction_prob": round(float(prob), 4),
                "current_type": current_type,
                "source_content": content_map.get(node_ids[s_idx], "?"),
                "target_content": content_map.get(node_ids[t_idx], "?"),
            })

    print(f"  {C_TEXT}Found {C_BRIGHT}{len(repairs['contradictions'])}{C_TEXT} new contradictions to review{C_RESET}")

    return repairs


# ── Execution ──────────────────────────────────────────────────────────────


def preview_repairs(repairs: dict, limit: int = 0):
    """Show what would be changed."""
    total = len(repairs["isolated_links"]) + len(repairs["hidden_links"])

    print(f"\n  {C_GOLD}{C_BOLD}  Repair Plan{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")
    print(f"  {C_TEXT}Links for isolated thoughts: {C_BRIGHT}{len(repairs['isolated_links'])}{C_RESET}")
    print(f"  {C_TEXT}Hidden connections:          {C_BRIGHT}{len(repairs['hidden_links'])}{C_RESET}")
    print(f"  {C_TEXT}Contradictions to review:    {C_BRIGHT}{len(repairs['contradictions'])}{C_RESET}")
    print(f"  {C_TEXT}Total links to create:       {C_BRIGHT}{total}{C_RESET}")
    if limit > 0:
        print(f"  {C_TEXT}Limit:                      {C_BRIGHT}{limit}{C_RESET}")

    # Sample of isolated links
    print(f"\n  {C_INFO}Sample — Isolated Thought Links:{C_RESET}")
    for r in repairs["isolated_links"][:5]:
        print(f"    {C_SUCCESS}P={r['probability']}{C_RESET} {C_DIM}({r['relation_type']}){C_RESET}")
        print(f"      {C_TEXT}{r['source_content'][:80]}{C_RESET}")
        print(f"      {C_DIM}→{C_RESET} {C_TEXT}{r['target_content'][:80]}{C_RESET}")

    # Sample of hidden links
    print(f"\n  {C_INFO}Sample — Hidden Connections:{C_RESET}")
    for r in repairs["hidden_links"][:5]:
        print(f"    {C_SUCCESS}P={r['probability']}{C_RESET} {C_DIM}({r['relation_type']}){C_RESET}")
        print(f"      {C_TEXT}{r['source_content'][:80]}{C_RESET}")
        print(f"      {C_DIM}→{C_RESET} {C_TEXT}{r['target_content'][:80]}{C_RESET}")

    # Contradictions
    if repairs["contradictions"]:
        print(f"\n  {C_INFO}Contradictions (review only — not auto-fixed):{C_RESET}")
        for c in repairs["contradictions"][:3]:
            print(f"    {C_DANGER}P={c['contradiction_prob']}{C_RESET} {C_DIM}(currently: {c['current_type']}){C_RESET}")
            print(f"      {C_TEXT}{c['source_content'][:80]}{C_RESET}")
            print(f"      {C_TEXT}{c['target_content'][:80]}{C_RESET}")


def execute_repairs(repairs: dict, limit: int = 0):
    """Apply repairs to CORTEX."""
    all_links = repairs["isolated_links"] + repairs["hidden_links"]
    if limit > 0:
        all_links = all_links[:limit]

    print(f"\n  {C_GOLD}{C_BOLD}  Executing Repairs{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")
    print(f"  {C_TEXT}Creating {C_BRIGHT}{len(all_links)}{C_TEXT} links...{C_RESET}")

    created = 0
    failed = 0
    t0 = time.time()

    for i, link in enumerate(all_links):
        success = insert_link(
            source_id=link["source_id"],
            target_id=link["target_id"],
            relation_type=link["relation_type"],
            confidence=link["probability"],
            context=link["context"],
        )
        if success:
            created += 1
        else:
            failed += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    {C_DIM}{i+1}/{len(all_links)} ({rate:.0f}/s) — {created} created, {failed} failed{C_RESET}")

    elapsed = time.time() - t0
    print(f"\n  {C_SUCCESS}{C_BOLD}Done:{C_RESET} {C_TEXT}{created} links created, {failed} failed ({elapsed:.1f}s){C_RESET}")

    # Contradiction report (manual review needed)
    if repairs["contradictions"]:
        print(f"\n  {C_DANGER}{C_BOLD}  Contradictions Requiring Manual Review:{C_RESET}")
        for c in repairs["contradictions"]:
            print(f"\n    {C_DANGER}P={c['contradiction_prob']}{C_RESET} {C_DIM}(currently: {c['current_type']}){C_RESET}")
            print(f"      {C_TEXT}{c['source_content']}{C_RESET}")
            print(f"      {C_TEXT}{c['target_content']}{C_RESET}")
            print(f"      {C_DIM}source: {c['source_id']}{C_RESET}")
            print(f"      {C_DIM}target: {c['target_id']}{C_RESET}")

    return created, failed


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="CORTEX Graph Repair")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't write")
    parser.add_argument("--execute", action="store_true", help="Apply repairs to CORTEX")
    parser.add_argument("--limit", type=int, default=0, help="Max links to create (0=all)")
    parser.add_argument("--gnn-ckpt", type=str, default="../pillar1/checkpoints/best_model.pt")
    parser.add_argument("--gnn-data", type=str, default="../pillar1/cortex_graph.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--link-threshold", type=float, default=0.85)
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print(f"{C_DANGER}Specify --dry-run or --execute{C_RESET}")
        sys.exit(1)

    print(f"\n{C_GOLD}{C_BOLD}  SABLE — CORTEX Graph Repair{C_RESET}")
    print(f"  {C_DIM}{'═' * 45}{C_RESET}")
    mode = "DRY RUN" if args.dry_run else "EXECUTE"
    print(f"  {C_TEXT}Mode: {C_BRIGHT}{mode}{C_RESET}\n")

    # Pull data
    print(f"  {C_INFO}Pulling CORTEX graph...{C_RESET}")
    thoughts, links = pull_cortex()
    print(f"  {C_TEXT}Thoughts: {C_BRIGHT}{len(thoughts)}{C_RESET}")
    print(f"  {C_TEXT}Links:    {C_BRIGHT}{len(links)}{C_RESET}")

    # Load GNN
    print(f"\n  {C_INFO}Loading GNN...{C_RESET}")
    model, data = load_gnn(args.gnn_ckpt, args.gnn_data, args.device)

    # Find repairs
    print(f"\n  {C_INFO}Analyzing graph for repairs...{C_RESET}")
    repairs = find_repairs(
        thoughts, links, model, data, args.device,
        link_threshold=args.link_threshold,
    )

    # Preview
    preview_repairs(repairs, args.limit)

    # Execute if not dry run
    if args.execute:
        print(f"\n  {C_GOLD}Proceeding with execution...{C_RESET}")
        _, _ = execute_repairs(repairs, args.limit)

        print(f"\n  {C_GOLD}{C_BOLD}  Post-Repair Actions{C_RESET}")
        print(f"  {C_DIM}{'─' * 45}{C_RESET}")
        print(f"  {C_TEXT}1. Re-export GNN data: python cortex_gnn_export.py{C_RESET}")
        print(f"  {C_TEXT}2. Re-train GNN:       python cortex_gnn_model.py{C_RESET}")
        print(f"  {C_TEXT}3. Re-run diagnostic:  python cortex_diagnostic.py{C_RESET}")
        print(f"  {C_TEXT}4. Review contradictions manually{C_RESET}")
    else:
        print(f"\n  {C_DIM}Dry run complete. Use --execute to apply changes.{C_RESET}")

    print()


if __name__ == "__main__":
    main()
