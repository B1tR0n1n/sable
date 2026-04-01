#!/usr/bin/env python3
"""
SABLE Dashboard API
=====================
FastAPI backend bridging the React frontend to all SABLE services.

Endpoints:
  /api/graph          — CORTEX graph data for visualization
  /api/stats          — Graph statistics
  /api/gnn/suggest    — GNN link suggestions
  /api/gnn/contradictions — Contradiction detection
  /api/diagnostic     — Run orchestrator diagnostic
  /api/repair/preview — Preview graph repairs
  /api/repair/execute — Execute graph repairs

Usage:
    pip install fastapi uvicorn
    python api.py
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict

import httpx

# Add to PYTHONPATH before importing FastAPI
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar3"))
sys.path.insert(0, str(Path(__file__).parent.parent / "orchestrator"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lqpvskwevanpgywdksqu.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxxcHZza3dldmFucGd5d2Rrc3F1Iiwi"
    "cm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA0ODQ0MSwiZXhwIjoyMDg5"
    "NjI0NDQxfQ.2urXziUOfXC-lMwl1tz4RzsNuaMM_iL4A1I_hefwEJs"
))
GNN_SERVER = "http://localhost:5070"

app = FastAPI(title="SABLE Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Prefer": "return=representation",
    }


@app.get("/api/graph")
async def get_graph():
    """Pull CORTEX graph for visualization."""
    async with httpx.AsyncClient(timeout=60) as client:
        # Thoughts (exclude archived)
        thoughts = []
        offset = 0
        while True:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/thoughts",
                headers=supabase_headers(),
                params={"select": "id,content,metadata,created_at",
                        "archived": "eq.false",
                        "order": "created_at.asc", "offset": offset, "limit": 1000},
            )
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
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/thought_links",
                headers=supabase_headers(),
                params={"select": "source_id,target_id,relation_type,confidence",
                        "offset": offset, "limit": 1000},
            )
            batch = resp.json()
            if not batch:
                break
            links.extend(batch)
            offset += len(batch)
            if len(batch) < 1000:
                break

        # Projects
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/projects",
            headers=supabase_headers(),
            params={"select": "id,name,slug,status"},
        )
        projects = resp.json() if resp.status_code == 200 else []

        # Project-thought mappings
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/project_thoughts",
            headers=supabase_headers(),
            params={"select": "project_id,thought_id", "limit": 5000},
        )
        pt_map = resp.json() if resp.status_code == 200 else []

    # Build project lookup
    project_by_id = {p["id"]: p["name"] for p in projects}
    thought_project = {}
    for pt in pt_map:
        if pt["project_id"] in project_by_id:
            thought_project[pt["thought_id"]] = project_by_id[pt["project_id"]]

    # Build graph nodes
    nodes = []
    for t in thoughts:
        meta = t.get("metadata") or {}
        nodes.append({
            "id": t["id"],
            "content": t["content"][:150],
            "type": meta.get("type", "observation"),
            "topics": meta.get("topics", [])[:3],
            "project": thought_project.get(t["id"], ""),
            "created": t.get("created_at", "")[:10],
        })

    edges = [
        {
            "source": l["source_id"],
            "target": l["target_id"],
            "relation": l.get("relation_type", "related"),
            "confidence": l.get("confidence", 1.0),
        }
        for l in links
    ]

    return {"nodes": nodes, "edges": edges, "projects": projects}


@app.get("/api/stats")
async def get_stats():
    """Graph-level statistics."""
    graph = await get_graph()
    nodes = graph["nodes"]
    edges = graph["edges"]

    type_counts = defaultdict(int)
    rel_counts = defaultdict(int)
    project_counts = defaultdict(int)
    topic_counts = defaultdict(int)

    for n in nodes:
        type_counts[n["type"]] += 1
        if n["project"]:
            project_counts[n["project"]] += 1
        for t in n["topics"]:
            topic_counts[t] += 1

    for e in edges:
        rel_counts[e["relation"]] += 1

    # Degree distribution
    adj = defaultdict(int)
    for e in edges:
        adj[e["source"]] += 1
        adj[e["target"]] += 1
    degrees = list(adj.values()) if adj else [0]
    isolated = sum(1 for n in nodes if n["id"] not in adj)

    # Get archived count
    archived_count = 0
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/thoughts",
                headers={**supabase_headers(), "Prefer": "count=exact"},
                params={"select": "id", "archived": "eq.true", "limit": 0},
            )
            archived_count = int(resp.headers.get("content-range", "*/0").split("/")[-1])
        except Exception:
            pass

    return {
        "n_thoughts": len(nodes),
        "n_links": len(edges),
        "n_projects": len(graph["projects"]),
        "n_archived": archived_count,
        "isolated": isolated,
        "avg_degree": round(sum(degrees) / max(len(degrees), 1), 2),
        "max_degree": max(degrees),
        "type_counts": dict(type_counts),
        "rel_counts": dict(rel_counts),
        "project_counts": dict(project_counts),
        "top_topics": sorted(topic_counts.items(), key=lambda x: -x[1])[:15],
    }


class SuggestRequest(BaseModel):
    thought_id: str
    top_k: int = 10


@app.post("/api/gnn/suggest")
async def gnn_suggest(req: SuggestRequest):
    """Proxy to GNN server for link suggestions."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{GNN_SERVER}/suggest",
            json={"thought_id": req.thought_id, "top_k": req.top_k},
        )
        return resp.json()


@app.get("/api/gnn/contradictions")
async def gnn_contradictions():
    """Proxy to GNN server for contradiction detection."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{GNN_SERVER}/contradictions",
            json={"top_k": 20, "threshold": 0.3},
        )
        return resp.json()


@app.get("/api/gnn/stats")
async def gnn_stats():
    """Proxy to GNN server for model stats."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{GNN_SERVER}/stats")
        return resp.json()


@app.get("/api/health")
async def health():
    """Health check — test all services."""
    status = {"api": "ok", "cortex": "unknown", "gnn": "unknown"}

    async with httpx.AsyncClient(timeout=5) as client:
        try:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/thoughts",
                headers=supabase_headers(),
                params={"select": "id", "limit": "1"},
            )
            status["cortex"] = "ok" if resp.status_code == 200 else "error"
        except Exception:
            status["cortex"] = "error"

        try:
            resp = await client.get(f"{GNN_SERVER}/health")
            status["gnn"] = "ok" if resp.status_code == 200 else "error"
        except Exception:
            status["gnn"] = "offline"

    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3001)
