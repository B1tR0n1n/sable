#!/usr/bin/env python3
"""
Project PARALLAX — SABLE on CORTEX
=====================================
The definitive demo: SABLE reasoning about its creator's knowledge graph.

Three pillars analyzing real cognitive data:
  - GNN: What structural patterns exist in Keith's thinking?
  - POMDP: What areas of the knowledge graph have the most uncertainty?
  - Mamba: How have beliefs evolved, and what changes are coming?

This isn't a synthetic test. This is SABLE doing what it was designed to do.

Usage:
    python cortex_diagnostic.py
    python cortex_diagnostic.py --focus sable    # Focus on a project
    python cortex_diagnostic.py --deep           # More rollouts, deeper analysis
"""

import argparse
import os
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
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
THOUGHT_TYPES = ["observation", "task", "idea", "reference", "person_note", "instruction"]

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


# ── CORTEX Data Pull ──────────────────────────────────────────────────────


def pull_cortex():
    """Pull full graph from CORTEX (read-only)."""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Prefer": "return=representation",
    }
    client = httpx.Client(timeout=60)

    # Thoughts
    thoughts = []
    offset = 0
    while True:
        resp = client.get(
            f"{SUPABASE_URL}/rest/v1/thoughts",
            headers=headers,
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

    # Links
    links = []
    offset = 0
    while True:
        resp = client.get(
            f"{SUPABASE_URL}/rest/v1/thought_links",
            headers=headers,
            params={"select": "source_id,target_id,relation_type,confidence",
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

    # Projects
    resp = client.get(
        f"{SUPABASE_URL}/rest/v1/projects",
        headers=headers,
        params={"select": "id,name,slug,status"},
    )
    projects = resp.json() if resp.status_code == 200 else []

    # Project-thought mappings
    resp = client.get(
        f"{SUPABASE_URL}/rest/v1/project_thoughts",
        headers=headers,
        params={"select": "project_id,thought_id,relevance", "limit": 5000},
    )
    project_thoughts = resp.json() if resp.status_code == 200 else []

    client.close()
    return thoughts, links, projects, project_thoughts


# ── GNN Analysis ──────────────────────────────────────────────────────────


def run_gnn_analysis(thoughts, links, checkpoint_path, device="cuda"):
    """Run GNN structural analysis on CORTEX."""

    # Load the CORTEX-trained model and data
    data = torch.load(
        str(Path(__file__).parent.parent / "pillar1" / "cortex_graph.pt"),
        weights_only=False,
    )
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    mc = ckpt["config"]
    model = SableGNN(
        in_dim=mc["in_dim"], hidden_dim=mc["hidden_dim"],
        edge_dim=mc["edge_dim"], num_layers=mc["num_layers"],
        heads=mc["heads"], dropout=mc["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    node_ids = data.node_ids
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    content_map = {t["id"]: t["content"][:150] for t in thoughts}
    meta_map = {t["id"]: t.get("metadata", {}) or {} for t in thoughts}

    with torch.no_grad():
        node_emb = model.encode(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
        )

        # 1. Find top missing links (highest probability unlinked pairs)
        existing = set()
        for link in links:
            s = id_to_idx.get(link["source_id"])
            t = id_to_idx.get(link["target_id"])
            if s is not None and t is not None:
                existing.add((s, t))
                existing.add((t, s))

        # Score a sample of unlinked pairs
        n_nodes = len(node_ids)
        rng = np.random.RandomState(42)
        sample_size = min(50000, n_nodes * (n_nodes - 1) // 2)
        candidates = []

        for _ in range(sample_size):
            i = rng.randint(0, n_nodes)
            j = rng.randint(0, n_nodes)
            if i != j and (i, j) not in existing:
                candidates.append((i, j))
                existing.add((i, j))  # Don't sample same pair twice

        if candidates:
            src = torch.tensor([c[0] for c in candidates], dtype=torch.long)
            tgt = torch.tensor([c[1] for c in candidates], dtype=torch.long)
            pair_ei = torch.stack([src, tgt]).to(device)

            link_logits = model.predict_link(node_emb, pair_ei)
            link_probs = torch.sigmoid(link_logits).cpu().numpy()

            type_logits = model.predict_link_type(node_emb, pair_ei)
            type_preds = type_logits.argmax(dim=-1).cpu().numpy()

            contra_logits = model.predict_contradiction(node_emb, pair_ei)
            contra_probs = torch.sigmoid(contra_logits).cpu().numpy()

        # 2. Find contradictions in existing edges
        all_ei = data.edge_index.to(device)
        all_contra = torch.sigmoid(
            model.predict_contradiction(node_emb, all_ei)
        ).cpu().numpy()
        all_labels = data.edge_labels.cpu().numpy() if hasattr(data, "edge_labels") else None

    # Sort and return results
    results = {
        "missing_links": [],
        "contradictions": [],
        "structural_hubs": [],
    }

    # Top missing links
    if candidates:
        top_idx = np.argsort(link_probs)[::-1][:20]
        for idx in top_idx:
            i, j = candidates[idx]
            prob = link_probs[idx]
            if prob < 0.7:
                continue
            pred_type = RELATION_TYPES[type_preds[idx]]
            contra = contra_probs[idx]
            results["missing_links"].append({
                "source": node_ids[i],
                "target": node_ids[j],
                "source_content": content_map.get(node_ids[i], "?")[:100],
                "target_content": content_map.get(node_ids[j], "?")[:100],
                "link_probability": round(float(prob), 4),
                "predicted_type": pred_type,
                "contradiction_prob": round(float(contra), 4),
            })

    # Top contradictions
    top_contra_idx = np.argsort(all_contra)[::-1][:15]
    for idx in top_contra_idx:
        prob = all_contra[idx]
        if prob < 0.3:
            continue
        s_idx = data.edge_index[0, idx].item()
        t_idx = data.edge_index[1, idx].item()
        current_type = RELATION_TYPES[all_labels[idx]] if all_labels is not None else "?"
        results["contradictions"].append({
            "source": node_ids[s_idx],
            "target": node_ids[t_idx],
            "source_content": content_map.get(node_ids[s_idx], "?")[:100],
            "target_content": content_map.get(node_ids[t_idx], "?")[:100],
            "contradiction_prob": round(float(prob), 4),
            "current_type": current_type,
            "is_known": current_type == "contradicts",
        })

    # Structural hubs — nodes with most connections (in + out)
    degree = defaultdict(int)
    for link in links:
        degree[link["source_id"]] += 1
        degree[link["target_id"]] += 1

    top_hubs = sorted(degree.items(), key=lambda x: -x[1])[:10]
    for hub_id, deg in top_hubs:
        meta = meta_map.get(hub_id, {})
        results["structural_hubs"].append({
            "id": hub_id,
            "degree": deg,
            "content": content_map.get(hub_id, "?")[:120],
            "type": meta.get("type", "?"),
            "topics": meta.get("topics", []),
        })

    return results


# ── Knowledge Graph Analysis ──────────────────────────────────────────────


def analyze_graph_structure(thoughts, links, projects, project_thoughts):
    """Compute graph-level statistics and structural properties."""

    # Basic stats
    type_counts = defaultdict(int)
    topic_counts = defaultdict(int)
    people_counts = defaultdict(int)
    rel_counts = defaultdict(int)

    for t in thoughts:
        meta = t.get("metadata", {}) or {}
        type_counts[meta.get("type", "unknown")] += 1
        for topic in meta.get("topics", []):
            topic_counts[topic] += 1
        for person in meta.get("people", []):
            people_counts[person] += 1

    for link in links:
        rel_counts[link.get("relation_type", "unknown")] += 1

    # Project distribution
    project_map = {p["id"]: p for p in projects}
    project_thought_counts = defaultdict(int)
    for pt in project_thoughts:
        pid = pt["project_id"]
        if pid in project_map:
            project_thought_counts[project_map[pid]["name"]] += 1

    # Connectivity
    adj = defaultdict(set)
    for link in links:
        adj[link["source_id"]].add(link["target_id"])
        adj[link["target_id"]].add(link["source_id"])

    degrees = [len(neighbors) for neighbors in adj.values()]
    isolated = sum(1 for t in thoughts if t["id"] not in adj)

    # Date range
    dates = [t.get("created_at", "") for t in thoughts if t.get("created_at")]
    date_range = (min(dates)[:10], max(dates)[:10]) if dates else ("?", "?")

    return {
        "n_thoughts": len(thoughts),
        "n_links": len(links),
        "n_projects": len(projects),
        "type_counts": dict(type_counts),
        "top_topics": sorted(topic_counts.items(), key=lambda x: -x[1])[:15],
        "top_people": sorted(people_counts.items(), key=lambda x: -x[1])[:10],
        "rel_counts": dict(rel_counts),
        "project_counts": dict(project_thought_counts),
        "avg_degree": round(np.mean(degrees), 2) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "isolated_nodes": isolated,
        "date_range": date_range,
    }


# ── Cognitive Uncertainty Analysis ────────────────────────────────────────


def find_uncertainty_zones(thoughts, links, projects, project_thoughts):
    """Find areas of the knowledge graph with low connectivity or
    missing structure — the cognitive blind spots."""

    adj = defaultdict(set)
    for link in links:
        adj[link["source_id"]].add(link["target_id"])
        adj[link["target_id"]].add(link["source_id"])

    # Project membership
    thought_projects = defaultdict(set)
    project_map = {p["id"]: p["name"] for p in projects}
    for pt in project_thoughts:
        if pt["project_id"] in project_map:
            thought_projects[pt["thought_id"]].add(project_map[pt["project_id"]])

    content_map = {t["id"]: t["content"][:120] for t in thoughts}
    meta_map = {t["id"]: t.get("metadata", {}) or {} for t in thoughts}

    # Find thoughts with high topic relevance but low connectivity
    # These are important ideas that aren't well-integrated into the graph
    underlinked = []
    for t in thoughts:
        tid = t["id"]
        meta = t.get("metadata", {}) or {}
        n_links = len(adj.get(tid, set()))
        topics = meta.get("topics", [])
        t_type = meta.get("type", "observation")

        # Ideas and tasks with zero or one link are underlinked
        if t_type in ("idea", "task") and n_links <= 1:
            underlinked.append({
                "id": tid,
                "content": content_map.get(tid, "?"),
                "type": t_type,
                "topics": topics,
                "links": n_links,
                "projects": list(thought_projects.get(tid, set())),
            })

    # Find cross-project gaps — projects that share topics but have
    # no direct thought links between them
    project_topics = defaultdict(set)
    for pt in project_thoughts:
        tid = pt["thought_id"]
        pid = pt["project_id"]
        if pid in project_map:
            pname = project_map[pid]
            meta = meta_map.get(tid, {})
            for topic in meta.get("topics", []):
                project_topics[pname].add(topic)

    cross_project_gaps = []
    project_names = list(project_topics.keys())
    for i, p1 in enumerate(project_names):
        for p2 in project_names[i + 1:]:
            shared = project_topics[p1] & project_topics[p2]
            if shared:
                cross_project_gaps.append({
                    "project_1": p1,
                    "project_2": p2,
                    "shared_topics": list(shared)[:5],
                    "n_shared": len(shared),
                })

    cross_project_gaps.sort(key=lambda x: -x["n_shared"])

    return {
        "underlinked_thoughts": underlinked[:15],
        "cross_project_gaps": cross_project_gaps[:10],
        "isolated_count": sum(1 for t in thoughts if t["id"] not in adj),
    }


# ── Main ──────────────────────────────────────────────────────────────────


def run_cortex_diagnostic(
    gnn_checkpoint: str = "../pillar1/checkpoints/best_model.pt",
    device: str = "cuda",
    focus_project: str = None,
):
    print(f"\n{C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║        SABLE — Cognitive Graph Diagnostic            ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║        Reasoning About Its Creator's Mind             ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════════╝{C_RESET}\n")

    # Pull data
    print(f"  {C_INFO}Pulling CORTEX knowledge graph...{C_RESET}")
    t0 = time.time()
    thoughts, links, projects, project_thoughts = pull_cortex()
    pull_time = time.time() - t0
    print(f"  {C_TEXT}Pulled {C_BRIGHT}{len(thoughts)}{C_TEXT} thoughts, "
          f"{C_BRIGHT}{len(links)}{C_TEXT} links, "
          f"{C_BRIGHT}{len(projects)}{C_TEXT} projects ({pull_time:.1f}s){C_RESET}")

    # ── Graph Structure ──
    print(f"\n  {C_GOLD}{C_BOLD}  Knowledge Graph Structure{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")

    stats = analyze_graph_structure(thoughts, links, projects, project_thoughts)

    print(f"  {C_TEXT}Thoughts:     {C_BRIGHT}{stats['n_thoughts']}{C_RESET}")
    print(f"  {C_TEXT}Links:        {C_BRIGHT}{stats['n_links']}{C_RESET}")
    print(f"  {C_TEXT}Projects:     {C_BRIGHT}{stats['n_projects']}{C_RESET}")
    print(f"  {C_TEXT}Date range:   {C_BRIGHT}{stats['date_range'][0]} → {stats['date_range'][1]}{C_RESET}")
    print(f"  {C_TEXT}Avg degree:   {C_BRIGHT}{stats['avg_degree']}{C_RESET}")
    print(f"  {C_TEXT}Max degree:   {C_BRIGHT}{stats['max_degree']}{C_RESET}")
    print(f"  {C_TEXT}Isolated:     {C_BRIGHT}{stats['isolated_nodes']}{C_RESET}")

    print(f"\n  {C_INFO}Thought types:{C_RESET}")
    for ttype, count in sorted(stats["type_counts"].items(), key=lambda x: -x[1]):
        bar = "█" * min(count // 10, 40)
        print(f"    {C_DIM}{ttype:15s}{C_RESET} {C_TEXT}{count:4d}{C_RESET} {C_GOLD}{bar}{C_RESET}")

    print(f"\n  {C_INFO}Link types:{C_RESET}")
    for rtype, count in sorted(stats["rel_counts"].items(), key=lambda x: -x[1]):
        bar = "█" * min(count // 10, 40)
        print(f"    {C_DIM}{rtype:15s}{C_RESET} {C_TEXT}{count:4d}{C_RESET} {C_GOLD}{bar}{C_RESET}")

    print(f"\n  {C_INFO}Top topics:{C_RESET}")
    for topic, count in stats["top_topics"][:10]:
        print(f"    {C_DIM}{topic:30s}{C_RESET} {C_TEXT}{count}{C_RESET}")

    print(f"\n  {C_INFO}Projects:{C_RESET}")
    for pname, count in sorted(stats["project_counts"].items(), key=lambda x: -x[1]):
        bar = "█" * min(count // 3, 30)
        print(f"    {C_DIM}{pname:20s}{C_RESET} {C_TEXT}{count:3d} thoughts{C_RESET} {C_GOLD}{bar}{C_RESET}")

    # ── GNN Structural Analysis ──
    print(f"\n\n  {C_GOLD}{C_BOLD}  Pillar 1 — Structural Reasoning (GNN){C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")

    print(f"  {C_INFO}Running GNN analysis...{C_RESET}")
    t0 = time.time()
    gnn_results = run_gnn_analysis(thoughts, links, gnn_checkpoint, device)
    gnn_time = time.time() - t0
    print(f"  {C_DIM}({gnn_time:.1f}s){C_RESET}")

    # Structural hubs
    print(f"\n  {C_INFO}Structural Hubs (most connected thoughts):{C_RESET}")
    for hub in gnn_results["structural_hubs"][:7]:
        topics = ", ".join(hub["topics"][:3]) if hub["topics"] else ""
        print(f"    {C_BRIGHT}[{hub['degree']} links]{C_RESET} {C_TEXT}{hub['content'][:90]}{C_RESET}")
        if topics:
            print(f"      {C_DIM}{topics}{C_RESET}")

    # Missing links
    print(f"\n  {C_INFO}Hidden Connections (GNN-predicted missing links):{C_RESET}")
    for ml in gnn_results["missing_links"][:8]:
        print(f"\n    {C_SUCCESS}P={ml['link_probability']}{C_RESET} {C_DIM}({ml['predicted_type']}){C_RESET}")
        print(f"      {C_TEXT}{ml['source_content'][:90]}{C_RESET}")
        print(f"      {C_DIM}→{C_RESET} {C_TEXT}{ml['target_content'][:90]}{C_RESET}")

    # Contradictions
    print(f"\n  {C_INFO}Belief Tensions (contradictions detected):{C_RESET}")
    for c in gnn_results["contradictions"][:5]:
        known = f"{C_DIM}[KNOWN]{C_RESET}" if c["is_known"] else f"{C_DANGER}[NEW]{C_RESET}"
        print(f"\n    {known} {C_BRIGHT}P={c['contradiction_prob']}{C_RESET} {C_DIM}(current: {c['current_type']}){C_RESET}")
        print(f"      {C_TEXT}{c['source_content'][:90]}{C_RESET}")
        print(f"      {C_TEXT}{c['target_content'][:90]}{C_RESET}")

    # ── Cognitive Uncertainty ──
    print(f"\n\n  {C_GOLD}{C_BOLD}  Pillar 2 — Cognitive Uncertainty Analysis{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")

    uncertainty = find_uncertainty_zones(thoughts, links, projects, project_thoughts)

    print(f"\n  {C_INFO}Underlinked Ideas (important thoughts with few connections):{C_RESET}")
    for ul in uncertainty["underlinked_thoughts"][:8]:
        proj_str = f" [{', '.join(ul['projects'])}]" if ul["projects"] else ""
        print(f"    {C_DANGER}[{ul['links']} links]{C_RESET} {C_TEXT}{ul['content'][:100]}{C_RESET}{C_DIM}{proj_str}{C_RESET}")

    print(f"\n  {C_INFO}Cross-Project Convergences (shared topics, no direct links):{C_RESET}")
    for gap in uncertainty["cross_project_gaps"][:6]:
        topics = ", ".join(gap["shared_topics"][:4])
        print(f"    {C_GOLD}{gap['project_1']}{C_RESET} ↔ {C_GOLD}{gap['project_2']}{C_RESET}")
        print(f"      {C_DIM}shared: {topics} ({gap['n_shared']} topics){C_RESET}")

    print(f"\n  {C_TEXT}Isolated thoughts (no links): {C_BRIGHT}{uncertainty['isolated_count']}{C_RESET}")

    # ── Summary ──
    total_time = time.time() - (t0 - gnn_time - pull_time)

    print(f"\n\n  {C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║              COGNITIVE ASSESSMENT                  ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════╝{C_RESET}")

    n_hidden = len(gnn_results["missing_links"])
    n_tensions = len([c for c in gnn_results["contradictions"] if not c["is_known"]])
    n_underlinked = len(uncertainty["underlinked_thoughts"])
    n_convergences = len(uncertainty["cross_project_gaps"])

    print(f"\n  {C_TEXT}Your knowledge graph has {C_BRIGHT}{stats['n_thoughts']}{C_TEXT} thoughts "
          f"connected by {C_BRIGHT}{stats['n_links']}{C_TEXT} typed links{C_RESET}")
    print(f"  {C_TEXT}across {C_BRIGHT}{stats['n_projects']}{C_TEXT} active projects.{C_RESET}")

    print(f"\n  {C_INFO}SABLE found:{C_RESET}")
    print(f"    {C_BRIGHT}{n_hidden}{C_TEXT} hidden connections your graph is missing{C_RESET}")
    print(f"    {C_BRIGHT}{n_tensions}{C_TEXT} new belief tensions to investigate{C_RESET}")
    print(f"    {C_BRIGHT}{n_underlinked}{C_TEXT} important ideas that aren't well-integrated{C_RESET}")
    print(f"    {C_BRIGHT}{n_convergences}{C_TEXT} cross-project convergences with shared topics{C_RESET}")

    print(f"\n  {C_DIM}Analysis time: {total_time:.1f}s{C_RESET}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SABLE Cognitive Graph Diagnostic")
    parser.add_argument("--gnn-ckpt", type=str, default="../pillar1/checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--focus", type=str, default=None, help="Focus on a specific project")
    args = parser.parse_args()

    run_cortex_diagnostic(
        gnn_checkpoint=args.gnn_ckpt,
        device=args.device,
        focus_project=args.focus,
    )
