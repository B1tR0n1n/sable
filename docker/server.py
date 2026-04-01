#!/usr/bin/env -S python -u
"""
SABLE Engine — FastAPI Server
Serves the dashboard + WebSocket for live inference streaming.
"""

import asyncio
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import torch
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from sable_engine import SableEngine
from nemotron_bridge import NemotronBridge

app = FastAPI(title="SABLE Engine", version="1.0")

# Global engine + state
engine = SableEngine(device="cuda")
current_scenario = None
scenario_data = None
autoplay_task = None
autoplay_speed = 1.0
connected_clients: list[WebSocket] = []
state_lock = asyncio.Lock()  # Protects engine state from concurrent access
_scenario_cache: list[dict] | None = None  # Cached scenario metadata
_topo_nodes: list[dict] = []  # Topology node metadata, indexed by position
_topo_edges: list[dict] = []  # Topology edges
mc_dropout_samples: int = 0   # 0 = off, >0 = MC dropout enabled with N samples
nemotron = NemotronBridge()   # Nemotron LLM bridge for natural language reports

# Scenario name validation: alphanumeric, hyphens, underscores only
_SAFE_NAME = re.compile(r"^[a-zA-Z0-9_\-]+$")

SCENARIO_DIR = Path(__file__).parent / "scenarios"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"
TOPOLOGY_DIR = Path(__file__).parent.parent / "adapters" / "topologies"

BANNER = """
\033[38;2;201;162;39m
  ┌─────────────────────────────────────────┐
  │                                         │
  │   S A B L E    E N G I N E    v 1.0     │
  │                                         │
  │   Three-Pillar Cognitive Architecture   │
  │   Temporal Chain Active                 │
  │                                         │
  └─────────────────────────────────────────┘
\033[0m"""


def load_scenario(name: str) -> dict | None:
    """Load a pre-computed scenario. Validates name to prevent path traversal."""
    if not _SAFE_NAME.match(name):
        return None
    path = SCENARIO_DIR / f"{name}.pt"
    if not path.resolve().parent == SCENARIO_DIR.resolve():
        return None
    if not path.exists():
        return None
    return torch.load(path, weights_only=False)


def _load_topology_metadata():
    """Load topology YAML for node labeling."""
    global _topo_nodes, _topo_edges
    if not TOPOLOGY_DIR.exists():
        return
    for f in sorted(TOPOLOGY_DIR.glob("*.y*ml")):
        data = yaml.safe_load(f.read_text())
        _topo_nodes = data.get("nodes", [])
        _topo_edges = data.get("edges", [])
        print(f"  Topology loaded: {f.stem} ({len(_topo_nodes)} nodes, {len(_topo_edges)} edges)")
        return


def node_label(idx: int) -> str:
    """Get human-readable label for a node index."""
    if idx < len(_topo_nodes):
        t = _topo_nodes[idx]
        return t.get("label", t.get("id", f"Node {idx:02d}"))
    return f"Node {idx:02d}"


def node_id(idx: int) -> str:
    """Get topology ID for a node index."""
    if idx < len(_topo_nodes):
        return _topo_nodes[idx].get("id", f"node-{idx:02d}")
    return f"node-{idx:02d}"


def node_type(idx: int) -> str:
    """Get component type for a node index."""
    if idx < len(_topo_nodes):
        return _topo_nodes[idx].get("type", "UNKNOWN")
    return "UNKNOWN"


def enrich_tick(result: dict) -> dict:
    """Add topology context to a tick result."""
    if not _topo_nodes:
        return result
    for node in result.get("nodes", []):
        i = node["id"]
        node["label"] = node_label(i)
        node["topo_id"] = node_id(i)
        node["component_type"] = node_type(i)
    return result


def enrich_recommendations(recs: dict) -> dict:
    """Replace generic 'Node XX' with infrastructure labels in recommendations."""
    if not _topo_nodes:
        return recs

    for action in recs.get("actions", []):
        # Parse node index from target string like "Node 05"
        target = action.get("target", "")
        if target.startswith("Node "):
            try:
                idx = int(target.split()[-1])
                action["target"] = node_label(idx)
                action["target_id"] = node_id(idx)
                action["target_type"] = node_type(idx)
            except (ValueError, IndexError):
                pass

        # Also enrich reason text
        reason = action.get("reason", "")
        for i in range(len(_topo_nodes)):
            reason = reason.replace(f"Node {i:02d}", node_label(i))
        action["reason"] = reason

        rec = action.get("recommendation", "")
        for i in range(len(_topo_nodes)):
            rec = rec.replace(f"Node {i:02d}", node_label(i))
        action["recommendation"] = rec

    # Enrich summary
    summary = recs.get("summary", "")
    for i in range(len(_topo_nodes)):
        summary = summary.replace(f"Node {i:02d}", node_label(i))
    recs["summary"] = summary

    # Add root cause label
    if recs.get("root_cause") is not None:
        recs["root_cause_label"] = node_label(recs["root_cause"])
        recs["root_cause_id"] = node_id(recs["root_cause"])
        recs["root_cause_type"] = node_type(recs["root_cause"])

    return recs


@app.on_event("startup")
async def startup():
    print(BANNER)
    print(f"  Loading checkpoints from {CHECKPOINT_DIR}...", flush=True)
    engine.load_checkpoints(str(CHECKPOINT_DIR))
    print(f"  Engine ready on {engine.device}", flush=True)

    # Load topology for node labeling
    _load_topology_metadata()

    # Initialize feedback database
    _init_feedback_db()
    print(f"  Feedback DB: {FEEDBACK_DB}")

    # Pre-load default scenario (prefer real telemetry)
    global current_scenario, scenario_data
    default = "smd_incident_1"
    scenario_data = load_scenario(default)
    if scenario_data is None:
        scenario_data = load_scenario("monday_morning")
        default = "monday_morning"
    if scenario_data:
        current_scenario = default
        engine.reset_state(scenario_data["n_nodes"])
        source = scenario_data.get("source", "sable_sim")
        print(f"  Default scenario: {current_scenario} ({scenario_data['n_nodes']} nodes, "
              f"{scenario_data['n_ticks']} ticks, source={source})")

    print("\n  SABLE Engine running at http://localhost:8080\n", flush=True)


# ── REST API ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    if DASHBOARD_PATH.exists():
        return HTMLResponse(DASHBOARD_PATH.read_text())
    return HTMLResponse("<h1>SABLE Engine — Dashboard not found</h1>")


@app.get("/api/status")
async def status():
    return {
        "engine_loaded": engine.model is not None,
        "scenario": current_scenario,
        "scenario_description": scenario_data["description"] if scenario_data else None,
        "cycle": engine.cycle,
        "n_nodes": engine.n_nodes,
        "total_ticks": scenario_data["n_ticks"] if scenario_data else 0,
        "autoplay": autoplay_task is not None,
        "autoplay_speed": autoplay_speed,
        "lora_enabled": engine.lora_active,
        "mc_dropout": mc_dropout_samples > 0,
        "mc_samples": mc_dropout_samples,
        "nemotron": nemotron.is_available(),
        **engine.get_summary(),
    }


@app.get("/api/scenarios")
async def list_scenarios():
    global _scenario_cache
    if _scenario_cache is None:
        _scenario_cache = []
        for f in sorted(SCENARIO_DIR.glob("*.pt")):
            data = torch.load(f, weights_only=False)
            _scenario_cache.append({
                "name": f.stem,
                "description": data.get("description", ""),
                "n_nodes": data["n_nodes"],
                "n_ticks": data["n_ticks"],
                "source": data.get("source", "sable_sim"),
            })
    # Only show real telemetry scenarios in the demo
    return [s for s in _scenario_cache if s["source"] == "smd"]


@app.post("/api/scenario")
async def set_scenario(body: dict):
    global current_scenario, scenario_data, autoplay_task
    async with state_lock:
        name = body.get("name", "monday_morning")

        if autoplay_task:
            autoplay_task.cancel()
            autoplay_task = None

        scenario_data = load_scenario(name)
        if scenario_data is None:
            return JSONResponse({"error": f"Scenario '{name}' not found"}, status_code=404)

        current_scenario = name
        engine.reset_state(scenario_data["n_nodes"])
        return {"loaded": name, "n_nodes": scenario_data["n_nodes"], "n_ticks": scenario_data["n_ticks"]}


@app.post("/api/tick")
async def tick():
    async with state_lock:
        if scenario_data is None:
            return JSONResponse({"error": "No scenario loaded"}, status_code=400)
        if engine.cycle >= scenario_data["n_ticks"]:
            return JSONResponse({"error": "Scenario complete", "cycle": engine.cycle}, status_code=400)

        result = await asyncio.to_thread(run_tick)
        return result


@app.post("/api/autoplay")
async def autoplay(body: dict):
    global autoplay_task, autoplay_speed
    async with state_lock:
        speed = body.get("speed", 1.0)
        action = body.get("action", "toggle")

        if action == "stop" or (action == "toggle" and autoplay_task is not None):
            if autoplay_task:
                autoplay_task.cancel()
                autoplay_task = None
            return {"autoplay": False}

        autoplay_speed = max(0.1, min(10.0, speed))
        if autoplay_task is None:
            autoplay_task = asyncio.create_task(autoplay_loop())
        return {"autoplay": True, "speed": autoplay_speed}


@app.get("/api/node/{idx}")
async def get_node_detail(idx: int):
    report = engine.get_node_report(idx)
    report["label"] = node_label(idx)
    report["topo_id"] = node_id(idx)
    report["component_type"] = node_type(idx)
    return report


@app.get("/api/recommendations")
async def recommendations():
    return enrich_recommendations(engine.get_recommendations())


@app.post("/api/reset")
async def reset():
    global autoplay_task
    async with state_lock:
        if autoplay_task:
            autoplay_task.cancel()
            autoplay_task = None
        if scenario_data:
            engine.reset_state(scenario_data["n_nodes"])
        return {"reset": True, "cycle": 0}


# ── Topology ─────────────────────────────────────────────────────────────


@app.get("/api/topology")
async def get_topology():
    """Serve infrastructure topology for the dashboard overlay."""
    # Try to find a topology file
    if not TOPOLOGY_DIR.exists():
        return {"nodes": [], "edges": []}

    for f in sorted(TOPOLOGY_DIR.glob("*.y*ml")):
        data = yaml.safe_load(f.read_text())
        return {
            "name": f.stem,
            "nodes": data.get("nodes", []),
            "edges": data.get("edges", []),
        }
    return {"nodes": [], "edges": []}


# ── WebSocket ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    try:
        # Send initial status
        await ws.send_json({"type": "status", **await _status_dict()})
        while True:
            # Keep alive — client can send commands too
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "tick":
                async with state_lock:
                    result = await asyncio.to_thread(run_tick)
                await ws.send_json({"type": "tick", **result})
    except WebSocketDisconnect:
        pass
    finally:
        if ws in connected_clients:
            connected_clients.remove(ws)


async def _status_dict():
    return {
        "scenario": current_scenario,
        "cycle": engine.cycle,
        "n_nodes": engine.n_nodes,
        "total_ticks": scenario_data["n_ticks"] if scenario_data else 0,
    }


async def broadcast(data: dict):
    """Send to all connected WebSocket clients."""
    dead = []
    for ws in connected_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients.remove(ws)


async def autoplay_loop():
    """Advance ticks automatically and stream results."""
    global autoplay_task
    try:
        while scenario_data and engine.cycle < scenario_data["n_ticks"]:
            async with state_lock:
                result = await asyncio.to_thread(run_tick)
            await broadcast({"type": "tick", **result})
            await asyncio.sleep(1.0 / autoplay_speed)  # Intentional rate-limit delay for animation pacing
        # Scenario complete
        await broadcast({"type": "complete", "cycle": engine.cycle})
    except asyncio.CancelledError:
        raise
    finally:
        autoplay_task = None


# ── Core tick logic ───────────────────────────────────────────────────────

def run_tick() -> dict:
    """Execute one inference cycle on current scenario."""
    t = engine.cycle
    n = scenario_data["n_nodes"]

    t0 = time.time()
    gnn = scenario_data["gnn"][:, t, :n, :].to(engine.device)      # (1, N, GNN_DIM)
    pomdp = scenario_data["pomdp"][:, t, :n, :].to(engine.device)   # (1, N, POMDP_DIM)
    mamba = scenario_data["mamba"][:, t, :n, :].to(engine.device)    # (1, N, NODE_FEAT_DIM)
    gt = scenario_data["ground_truth"][t, :n]                         # (N,)

    result = engine.infer(gnn, pomdp, mamba, ground_truth=gt, mc_samples=mc_dropout_samples)
    result["inference_ms"] = round((time.time() - t0) * 1000, 2)
    return enrich_tick(result)


# ── Live Monitor Endpoint ────────────────────────────────────────────────

@app.post("/api/live_tick")
async def live_tick(body: dict):
    """Receive pre-encoded pillar inputs from the live monitor.

    The live monitor (live_monitor.py) polls Prometheus, encodes telemetry
    into pillar tensor formats, and POSTs them here. This endpoint runs
    inference and returns the same result format as /api/tick.

    Body: {
        gnn: [[...]]         - (N, 1044) node features as nested list
        pomdp: {node_id: [...]} - node_id to 8-dim belief vector
        mamba: [[[...]]]     - (1, 2, max_nodes*26) temporal input
        node_ids: [str]      - ordered node IDs
        n_nodes: int
        ground_truth: [int]  - optional, state indices from health scorer
    }
    """
    import numpy as np

    async with state_lock:
        n_nodes = body["n_nodes"]
        node_ids = body["node_ids"]

        # Convert lists back to tensors
        gnn_np = np.array(body["gnn"], dtype=np.float32)  # (N, 1044)
        gnn_t = torch.tensor(gnn_np, device=engine.device).unsqueeze(0)  # (1, N, 1044)

        # POMDP: dict of beliefs -> (1, N, 8) tensor
        pomdp_list = []
        for nid in node_ids:
            b = body["pomdp"].get(nid, [1, 0, 0, 0, 1, 0, 0, 0])
            pomdp_list.append(b)
        pomdp_t = torch.tensor(pomdp_list, dtype=torch.float32, device=engine.device).unsqueeze(0)

        # Mamba: already (1, 2, max_nodes*26), but engine expects (1, N, feat_dim)
        # Extract current tick features for each node
        mamba_np = np.array(body["mamba"], dtype=np.float32)  # (1, 2, max_nodes*26)
        # Reshape tick 1 (current) into per-node features
        from adapters.encode import MAMBA_NODE_FEAT_DIM
        tick_flat = mamba_np[0, 1, :]  # Current tick
        n = min(n_nodes, len(tick_flat) // MAMBA_NODE_FEAT_DIM)
        mamba_nodes = tick_flat[:n * MAMBA_NODE_FEAT_DIM].reshape(n, MAMBA_NODE_FEAT_DIM)
        mamba_t = torch.tensor(mamba_nodes, dtype=torch.float32, device=engine.device).unsqueeze(0)

        # Ground truth (optional)
        gt = None
        if "ground_truth" in body and body["ground_truth"]:
            gt = torch.tensor(body["ground_truth"][:n_nodes], dtype=torch.long)

        # Initialize engine state for live data if needed
        if engine.n_nodes != n_nodes:
            engine.reset_state(n_nodes)

        # Update topology labels from node_ids
        global _topo_nodes
        if not _topo_nodes or len(_topo_nodes) != n_nodes:
            _topo_nodes = [{"id": nid, "label": nid, "type": "UNKNOWN"} for nid in node_ids]
            # Try to enrich from snapshot data if we have component types
            # (the live monitor doesn't send these yet, but future-proof)

        result = engine.infer(gnn_t, pomdp_t, mamba_t, ground_truth=gt,
                              mc_samples=mc_dropout_samples)
        result["source"] = "live"
        result["inference_ms"] = result.get("inference_ms", 0)

        # Enrich with node labels
        for node in result.get("nodes", []):
            i = node["id"]
            if i < len(node_ids):
                node["label"] = node_ids[i]
                node["topo_id"] = node_ids[i]

        # Broadcast to WebSocket clients
        await broadcast({"type": "live_tick", **result})

        return result


@app.post("/api/lora")
async def toggle_lora(body: dict):
    """Toggle LoRA adapter on/off for before/after demo."""
    enabled = bool(body.get("enabled", True))
    engine.toggle_lora(enabled)
    return {"lora_enabled": engine.lora_active}


@app.post("/api/mc_dropout")
async def set_mc_dropout(body: dict):
    """Toggle MC dropout for honest confidence estimates."""
    global mc_dropout_samples
    mc_dropout_samples = max(0, min(20, int(body.get("samples", 0))))
    return {"mc_dropout": mc_dropout_samples > 0, "samples": mc_dropout_samples}


# ── Operator Feedback ────────────────────────────────────────────────────

FEEDBACK_DB = Path(__file__).parent / "feedback.db"


def _init_feedback_db():
    """Create feedback table if it doesn't exist."""
    with sqlite3.connect(str(FEEDBACK_DB)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_idx INTEGER NOT NULL,
                node_id TEXT,
                predicted_state TEXT NOT NULL,
                correct_state TEXT NOT NULL,
                cycle INTEGER,
                scenario TEXT,
                operator TEXT DEFAULT '',
                timestamp REAL NOT NULL
            )
        """)


@app.post("/api/feedback")
async def submit_feedback(body: dict):
    """Submit an operator correction for a node's predicted state.

    Body: {
        node_idx: int,
        correct_state: "healthy" | "degraded" | "failed" | "unreachable" | "oscillating",
        operator: str (optional)
    }
    """
    node_idx = int(body.get("node_idx", -1))
    correct_state = body.get("correct_state", "")
    operator = body.get("operator", "")

    valid_states = ["healthy", "degraded", "failed", "unreachable", "oscillating"]
    if correct_state not in valid_states:
        return JSONResponse({"error": f"Invalid state. Must be one of: {valid_states}"}, status_code=400)
    if node_idx < 0 or node_idx >= engine.n_nodes:
        return JSONResponse({"error": f"Invalid node_idx. Must be 0-{engine.n_nodes - 1}"}, status_code=400)

    # Get current prediction for this node
    predicted = "unknown"
    if engine.history:
        latest = engine.history[-1]
        from sable_sim.core.states import STATE_NAMES
        predicted = STATE_NAMES[latest["predictions"][node_idx]]

    with sqlite3.connect(str(FEEDBACK_DB)) as conn:
        conn.execute(
            "INSERT INTO corrections (node_idx, node_id, predicted_state, correct_state, "
            "cycle, scenario, operator, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (node_idx, node_id(node_idx), predicted, correct_state,
             engine.cycle, current_scenario, operator, time.time()),
        )
        total = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]

    return {
        "recorded": True,
        "node": node_label(node_idx),
        "predicted": predicted,
        "corrected_to": correct_state,
        "total_corrections": total,
    }


@app.get("/api/feedback/stats")
async def feedback_stats():
    """Get feedback statistics."""
    if not FEEDBACK_DB.exists():
        return {"total": 0, "corrections": []}

    with sqlite3.connect(str(FEEDBACK_DB)) as conn:
        total = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]

        rows = conn.execute(
            "SELECT node_id, predicted_state, correct_state, cycle, scenario, timestamp "
            "FROM corrections ORDER BY timestamp DESC LIMIT 20"
        ).fetchall()

        state_stats = conn.execute(
            "SELECT predicted_state, correct_state, COUNT(*) "
            "FROM corrections GROUP BY predicted_state, correct_state"
        ).fetchall()

    corrections = [
        {"node_id": r[0], "predicted": r[1], "corrected_to": r[2],
         "cycle": r[3], "scenario": r[4], "timestamp": r[5]}
        for r in rows
    ]

    confusion = {}
    for predicted, correct, count in state_stats:
        if predicted not in confusion:
            confusion[predicted] = {}
        confusion[predicted][correct] = count

    return {
        "total": total,
        "corrections": corrections,
        "confusion": confusion,
        "ready_for_finetune": total >= 100,
    }


# ── Main ──────────────────────────────────────────────────────────────────

# ---- Nemotron LLM Bridge ----


@app.get("/api/nemotron/status")
async def nemotron_status():
    """Check if Nemotron is available."""
    return {"available": nemotron.is_available(), "url": nemotron.llama_url}


@app.post("/api/nemotron/report")
async def nemotron_report():
    """Generate a natural language incident report from current state."""
    if not engine.history or len(engine.history) < 2:
        return {"error": "Need at least 2 ticks for a report"}

    recs = enrich_recommendations(engine.get_recommendations())
    narrative = await asyncio.to_thread(nemotron.explain_recommendations, recs)

    return {
        "narrative": narrative,
        "recommendations": recs,
        "nemotron_available": nemotron.is_available(),
    }


@app.post("/api/nemotron/chat")
async def nemotron_chat(body: dict):
    """Chat with Nemotron about SABLE's findings."""
    user_message = body.get("message", "")
    history = body.get("history", [])

    if not user_message:
        return {"error": "No message provided"}

    if not engine.history:
        return {"error": "Run at least one tick first"}

    recs = enrich_recommendations(engine.get_recommendations())
    reply = await asyncio.to_thread(
        nemotron.chat, user_message, recs, history
    )

    return {
        "reply": reply,
        "nemotron_available": nemotron.is_available(),
    }


@app.post("/api/nemotron/after_action")
async def nemotron_after_action():
    """Generate an after-action report from a completed or in-progress scenario."""
    if not engine.history:
        return {"error": "No data for report"}

    # Build tick history summaries
    tick_summaries = []
    for h in engine.history:
        tick_summaries.append({
            "cycle": h["cycle"],
            "class_counts": {},
            "accuracy": None,
        })

    recs = enrich_recommendations(engine.get_recommendations())
    narrative = await asyncio.to_thread(
        nemotron.explain_scenario_complete, tick_summaries, recs
    )

    return {
        "narrative": narrative,
        "recommendations": recs,
        "cycles_completed": len(engine.history),
        "nemotron_available": nemotron.is_available(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
