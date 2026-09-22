#!/usr/bin/env python3
"""A stand-in for SABLE's engine server, for when the trained weights are not
on this machine. It is NOT SABLE: it serves SABLE's API shapes so the whole
loop — lab, OVERLORD, planner, gate, verification, receipts — runs live, and
it says so on every response (`engine: "stub"`, device `"stub"`).

What is real in it: SABLE's own Prometheus adapter and health scorer
(adapters/prometheus.py, adapters/health_scorer.py — unchanged) turn the
lab's metrics into per-node states exactly as the live monitor would feed
the engine. What is replaced: the neural fusion. States come straight from
the scorer; confidence is a heuristic and labelled as such; the root cause
is SABLE's own rule — the earliest node to leave `healthy` — with one
documented widening: `unreachable` nodes qualify too (the engine only ever
names a `failed` node, and a stopped lab service is `unreachable`).

Serves, like docker/server.py: GET /api/status, /api/topology,
/api/recommendations, /api/node/{idx}, /api/nemotron/status,
POST /api/nemotron/chat (Claude, through docker/claude_bridge.py, when
ANTHROPIC_API_KEY is set), WebSocket /ws broadcasting {"type": "live_tick"}.

    python3 -m console.lab.sable_stub [--config console/lab/sable_prometheus.yaml]
Env: PROMETHEUS_URL, SABLE_BIND (0.0.0.0), SABLE_PORT (8080), SABLE_LLM=claude
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]              # the sable checkout
for p in (ROOT, ROOT / "docker"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

STATE_NAMES = ["healthy", "degraded", "failed", "unreachable", "oscillating"]
STATE_IDX = {s: i for i, s in enumerate(STATE_NAMES)}


class StubEngine:
    """Mirrors the parts of SableEngine the server and the Claude bridge read:
    history of tick dicts, n_nodes, get_recommendations, get_node_report."""

    def __init__(self, node_ids: list[str], node_types: dict[str, str]):
        self.node_ids = list(node_ids)                  # positional = SABLE's node index
        self.node_types = dict(node_types)
        self.n_nodes = len(node_ids)
        self.history: list[dict[str, Any]] = []
        self.cycle = 0
        self.first_bad: dict[int, int] = {}             # idx -> first tick it left healthy
        self.transitions: dict[int, int] = {}
        self.last_state: dict[int, str] = {}

    # ---------------------------------------------------------------- ticks

    def tick_from_states(self, states: dict[str, str], healths: dict[str, float]) -> dict[str, Any]:
        self.cycle += 1
        nodes, counts = [], {s: 0 for s in STATE_NAMES}
        for i, nid in enumerate(self.node_ids):
            st = states.get(nid, "healthy")
            if st not in STATE_IDX:
                st = "healthy"
            h = healths.get(nid)
            # heuristic confidence: how far the health score sits from the scorer's thresholds
            conf = 0.95 if h is None else max(0.55, min(0.98, abs(h - 0.55) / 0.45))
            probs = {s: (1 - conf) / 4 for s in STATE_NAMES}
            probs[st] = conf
            if self.last_state.get(i, "healthy") != st:
                self.transitions[i] = self.transitions.get(i, 0) + 1
                if st != "healthy" and i not in self.first_bad:
                    self.first_bad[i] = self.cycle
            if st == "healthy":
                self.first_bad.pop(i, None)
            self.last_state[i] = st
            nodes.append({"id": i, "state": st, "state_idx": STATE_IDX[st], "confidence": round(conf, 4),
                          "probs": {k: round(v, 4) for k, v in probs.items()},
                          "transition": {"improving": 0.0, "stable": 1.0, "deteriorating": 0.0},
                          "trend": "stable", "routing": {"GNN": 0.0, "POMDP": 0.0, "Mamba": 0.0, "Fusion": 1.0},
                          "label": nid, "topo_id": nid, "component_type": self.node_types.get(nid, "UNKNOWN")})
            counts[st] += 1
        confs = [n["confidence"] for n in nodes] or [0.0]
        tick = {"cycle": self.cycle, "source": "live", "engine": "stub", "nodes": nodes, "class_counts": counts,
                "avg_confidence": round(sum(confs) / len(confs), 4),
                "min_confidence": {"node": min(range(len(confs)), key=confs.__getitem__), "value": min(confs)},
                "max_confidence": {"node": max(range(len(confs)), key=confs.__getitem__), "value": max(confs)},
                "routing": {}, "inference_ms": 0, "timestamp": time.time()}
        self.history.append(tick)
        if len(self.history) > 1000:
            del self.history[0]
        return tick

    # ---------------------------------------------------------------- SABLE's read API

    def get_recommendations(self) -> dict[str, Any]:
        if len(self.history) < 2:
            return {"actions": [], "summary": "Insufficient data — need at least 2 inference cycles.", "root_cause": None,
                    "total_affected": 0}
        last = self.history[-1]
        bad = [n for n in last["nodes"] if n["state"] != "healthy"]
        candidates = [n for n in bad if n["state"] in ("failed", "unreachable")]
        candidates.sort(key=lambda n: (self.first_bad.get(n["id"], self.cycle), n["id"]))
        root = candidates[0] if candidates else None
        actions = []
        if root is not None:
            actions.append({"priority": 1, "action": "INVESTIGATE ROOT CAUSE", "target": root["label"], "target_id": root["label"],
                            "target_type": root["component_type"],
                            "reason": f"{root['label']} was the first node to leave healthy (at tick {self.first_bad.get(root['id'], self.cycle)})",
                            "recommendation": "restart/recover the service and re-check its dependents"})
        for n in bad:
            if root is not None and n["id"] == root["id"]:
                continue
            actions.append({"priority": 2, "action": "MONITOR/INVESTIGATE", "target": n["label"], "target_id": n["label"],
                            "target_type": n["component_type"],
                            "reason": f"{n['label']} is {n['state']} (since tick {self.first_bad.get(n['id'], self.cycle)})",
                            "recommendation": "expected to recover once the root cause is healed"})
        summary = ("All nodes healthy." if not bad else
                   f"{len(bad)}/{self.n_nodes} nodes affected; root cause {root['label']} ({root['state']})" if root
                   else f"{len(bad)}/{self.n_nodes} nodes degraded; no failed or unreachable node yet")
        out = {"summary": summary, "total_affected": len(bad), "actions": actions,
               "root_cause": root["id"] if root else None, "engine": "stub"}
        if root is not None:
            out.update(root_cause_label=root["label"], root_cause_id=root["label"], root_cause_type=root["component_type"])
        return out

    def get_node_report(self, idx: int) -> dict[str, Any]:
        if not self.history:
            return {"error": "No inference cycles yet"}
        if not (0 <= idx < self.n_nodes):
            return {"error": f"node index {idx} out of range 0..{self.n_nodes - 1}"}
        traj = [{"cycle": t["cycle"], "prediction": t["nodes"][idx]["state"]} for t in self.history[-8:]]
        return {"node_id": idx, "current_state": self.history[-1]["nodes"][idx]["state"], "trajectory": traj,
                "n_state_changes": self.transitions.get(idx, 0), "label": self.node_ids[idx], "topo_id": self.node_ids[idx],
                "component_type": self.node_types.get(self.node_ids[idx], "UNKNOWN")}

    def get_summary(self) -> dict[str, Any]:
        return {"n_nodes": self.n_nodes, "cycle": self.cycle, "n_states": len(STATE_NAMES), "state_names": STATE_NAMES,
                "device": "stub", "model_loaded": False, "temporal_active": False, "engine": "stub"}


# ---------------------------------------------------------------- the server


def build_app(config_path: str, poll_interval: float = 5.0):
    import yaml
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    from console.lab.live_monitor_lab import build_prometheus_config, load_lab_config
    from adapters.health_scorer import HealthScorer
    from adapters.prometheus import PrometheusAdapter
    import live_monitor

    cfg = load_lab_config(config_path)
    if os.environ.get("PROMETHEUS_URL"):
        cfg["prometheus_url"] = os.environ["PROMETHEUS_URL"]
        cfg.setdefault("prometheus", {})["url"] = os.environ["PROMETHEUS_URL"]
    prom_cfg = build_prometheus_config(cfg)
    adapter = PrometheusAdapter(prom_cfg)
    scorer = HealthScorer(live_monitor.build_health_config(cfg.get("health_overrides")))
    topo_path = Path(config_path).resolve().parent / str(cfg.get("topology", "topology.yaml"))
    topo = yaml.safe_load(topo_path.read_text()) or {}
    nodes, edges = topo.get("nodes") or [], topo.get("edges") or []
    node_ids = [str(n["id"]) for n in nodes]
    engine = StubEngine(node_ids, {str(n["id"]): str(n.get("type", "UNKNOWN")) for n in nodes})
    clients: set[asyncio.Queue] = set()

    # the same bridge SABLE's server uses; Claude when SABLE_LLM=claude + a key
    from claude_bridge import ClaudeBridge, bridge_info, make_tools, select_bridge
    bridge = select_bridge()
    if isinstance(bridge, ClaudeBridge):
        bridge.register_tools(**make_tools(engine, lambda r: r, lambda i: node_ids[i], lambda i: node_ids[i],
                                           lambda i: engine.node_types.get(node_ids[i], "UNKNOWN"), nodes, edges))

    app = FastAPI(title="SABLE (stub)", version="stub")

    async def poll_loop():
        while True:
            try:
                snap = await asyncio.to_thread(adapter.poll_with_health, scorer)
                states = {nid: (n.state or "healthy") for nid, n in snap.nodes.items()}
                healths = {nid: n.health for nid, n in snap.nodes.items() if n.health is not None}
                tick = engine.tick_from_states(states, healths)
                msg = {"type": "live_tick", "tick": tick, **{k: v for k, v in tick.items() if k != "nodes"}, "nodes": tick["nodes"]}
                for q in list(clients):
                    if q.qsize() < 100:
                        q.put_nowait(msg)
            except Exception as e:                    # noqa: BLE001 — a bad poll is one missed tick
                print(f"stub: poll failed: {e}", file=sys.stderr, flush=True)
            await asyncio.sleep(poll_interval)

    @app.on_event("startup")
    async def _start():
        asyncio.create_task(poll_loop())
        print(f"  SABLE STUB (no trained weights): {len(node_ids)} nodes from {topo_path.name}, "
              f"prometheus {prom_cfg.url}, analyst {bridge_info(bridge).get('provider')}", flush=True)

    @app.get("/api/status")
    async def status():
        return {**engine.get_summary(), "scenario": None, "mode": "live", "mc_dropout": False, "mc_samples": 0}

    @app.get("/api/topology")
    async def topology():
        return {"name": topo_path.stem, "nodes": copy.deepcopy(nodes), "edges": copy.deepcopy(edges)}

    @app.get("/api/recommendations")
    async def recommendations():
        return engine.get_recommendations()

    @app.get("/api/node/{idx}")
    async def node(idx: int):
        return engine.get_node_report(idx)

    @app.get("/api/nemotron/status")
    async def nemotron_status():
        return {"available": bridge.is_available(), "url": bridge.llama_url, **bridge_info(bridge)}

    @app.post("/api/nemotron/chat")
    async def nemotron_chat(body: dict):
        recs = engine.get_recommendations()
        reply = await asyncio.to_thread(bridge.chat, str(body.get("message") or ""), recs, list(body.get("history") or []))
        return {"reply": reply, "nemotron_available": bridge.is_available()}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue()
        clients.add(q)
        try:
            await sock.send_json({"type": "status", **engine.get_summary()})
            while True:
                await sock.send_json(await q.get())
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(q)

    return app


def main(argv=None) -> int:
    import uvicorn
    ap = argparse.ArgumentParser(prog="sable_stub")
    ap.add_argument("--config", default=str(Path(__file__).parent / "sable_prometheus.yaml"))
    ap.add_argument("--interval", type=float, default=float(os.environ.get("SABLE_STUB_INTERVAL", "5")))
    ap.add_argument("--bind", default=os.environ.get("SABLE_BIND", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("SABLE_PORT", "8080")))
    args = ap.parse_args(argv)
    uvicorn.run(build_app(args.config, args.interval), host=args.bind, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
