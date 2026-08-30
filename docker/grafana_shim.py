"""
SABLE Engine — Grafana shim.

Adapts engine state into the JSON shapes the Grafana dashboard panels expect.
Mounted at /grafana/* by server.py. The marcusolsson-json-datasource queries
these paths via the datasource proxy.
"""

import time
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from sable_engine import STATE_NAMES

UNCERTAIN_THRESHOLD = 0.60
SERIES_LIMIT = 200
_SEVERITY_RANK = {"HARD": 3, "SOFT": 2, "LOOSE": 1, "NONE": 0}


def _ts_ms(entry: dict) -> int:
    return int(entry.get("timestamp", time.time()) * 1000)


def _confidences(entry: dict) -> list[float]:
    return entry.get("confidences", []) or []


def _safe_min(xs: list[float]) -> float:
    return min(xs) if xs else 0.0


def _safe_max(xs: list[float]) -> float:
    return max(xs) if xs else 0.0


def _safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _classify_health(mean_conf: float, uncertain: int, n_nodes: int) -> str:
    if n_nodes == 0:
        return "IDLE"
    if uncertain >= max(1, n_nodes // 3) or mean_conf < 0.5:
        return "CRITICAL"
    if uncertain > 0 or mean_conf < 0.7:
        return "DEGRADED"
    return "HEALTHY"


def _structural_stats(topo_nodes: list, topo_edges: list) -> list[dict]:
    stats = [
        {"connections": 0, "hard_deps": 0, "criticality": "NONE"}
        for _ in topo_nodes
    ]
    id_to_idx = {n.get("id"): i for i, n in enumerate(topo_nodes) if n.get("id")}
    for e in topo_edges:
        crit = e.get("criticality", "NONE")
        for end in (id_to_idx.get(e.get("source")), id_to_idx.get(e.get("target"))):
            if end is None:
                continue
            stats[end]["connections"] += 1
            if crit == "HARD":
                stats[end]["hard_deps"] += 1
            if _SEVERITY_RANK.get(crit, 0) > _SEVERITY_RANK.get(stats[end]["criticality"], 0):
                stats[end]["criticality"] = crit
    return stats


def _latest_node_state(engine, idx: int) -> tuple[str, float]:
    if not engine.history:
        return "unknown", 0.0
    latest = engine.history[-1]
    state_idx = latest["predictions"][idx] if idx < len(latest["predictions"]) else 0
    conf = latest["confidences"][idx] if idx < len(latest["confidences"]) else 0.0
    return STATE_NAMES[state_idx], round(float(conf), 4)


def _series_tail(history: list[dict]) -> list[dict]:
    return history[-SERIES_LIMIT:] if len(history) > SERIES_LIMIT else history


def make_router(srv) -> APIRouter:
    """Build the /grafana/* router. `srv` is the live server module reference
    (pass `sys.modules[__name__]` from server.py) so the shim reads the actual
    running engine, topology, and route handlers instead of re-importing them.
    """
    router = APIRouter()

    @router.get("/scenario_list")
    async def scenario_list():
        scenarios = await srv.list_scenarios()
        return {"scenarios": [s["name"] for s in scenarios]}

    @router.post("/control")
    async def control(body: dict):
        action = body.get("action", "")
        if action == "play":
            res = await srv.autoplay({"action": "toggle", "speed": float(body.get("speed", 1.0))})
            return {"ok": True, **res}
        if action == "stop":
            res = await srv.autoplay({"action": "stop"})
            return {"ok": True, **res}
        if action == "reset":
            res = await srv.reset()
            return {"ok": True, **res}
        if action == "scenario":
            name = body.get("name", "")
            if not name:
                return JSONResponse({"error": "missing scenario name"}, status_code=400)
            res = await srv.set_scenario({"name": name})
            return {"ok": True, **res} if "error" not in res else res
        if action == "lora":
            res = await srv.toggle_lora({"enabled": bool(body.get("enabled", True))})
            return {"ok": True, "lora": res["lora_enabled"]}
        return JSONResponse({"error": f"unknown action: {action}"}, status_code=400)

    @router.get("/hero")
    async def hero():
        eng = srv.engine
        n = eng.n_nodes
        if not eng.history:
            return {
                "health": "IDLE",
                "cycle": eng.cycle,
                "nodes": n,
                "min_confidence": 0,
                "uncertain_nodes": 0,
                "inference": 0,
                "accuracy": None,
                "lora": "ON" if eng.lora_active else "OFF",
                "confidence": 0,
                "spread": 0,
            }
        latest = eng.history[-1]
        confs = _confidences(latest)
        mn = _safe_min(confs)
        mx = _safe_max(confs)
        mean = _safe_mean(confs)
        uncertain = sum(1 for c in confs if c < UNCERTAIN_THRESHOLD)
        return {
            "health": _classify_health(mean, uncertain, n),
            "cycle": eng.cycle,
            "nodes": n,
            "min_confidence": round(mn, 3),
            "uncertain_nodes": uncertain,
            "inference": latest.get("inference_ms", 0),
            "accuracy": round(latest["accuracy"], 4) if latest.get("accuracy") is not None else None,
            "lora": "ON" if eng.lora_active else "OFF",
            "confidence": round(mean, 3),
            "spread": round(mx - mn, 3),
        }

    @router.get("/accuracy")
    async def accuracy_series():
        points = [
            {"time": _ts_ms(h), "value": round(float(h["accuracy"]), 4)}
            for h in _series_tail(srv.engine.history)
            if h.get("accuracy") is not None
        ]
        return {"points": points}

    @router.get("/latency")
    async def latency_series():
        points = [
            {"time": _ts_ms(h), "value": float(h.get("inference_ms", 0))}
            for h in _series_tail(srv.engine.history)
        ]
        return {"points": points}

    @router.get("/confidence")
    async def confidence_series():
        mn_pts, mean_pts, mx_pts, unc_pts = [], [], [], []
        for h in _series_tail(srv.engine.history):
            t = _ts_ms(h)
            confs = _confidences(h)
            mn_pts.append({"time": t, "value": round(_safe_min(confs), 4)})
            mean_pts.append({"time": t, "value": round(_safe_mean(confs), 4)})
            mx_pts.append({"time": t, "value": round(_safe_max(confs), 4)})
            unc_pts.append({"time": t, "value": sum(1 for c in confs if c < UNCERTAIN_THRESHOLD)})
        return {"min": mn_pts, "mean": mean_pts, "max": mx_pts, "uncertain": unc_pts}

    @router.get("/classes")
    async def classes():
        eng = srv.engine
        if not eng.history:
            return {"states": list(STATE_NAMES), "counts": [0] * len(STATE_NAMES)}
        preds = eng.history[-1]["predictions"]
        counts = [0] * len(STATE_NAMES)
        for p in preds:
            if 0 <= p < len(counts):
                counts[p] += 1
        return {"states": list(STATE_NAMES), "counts": counts}

    @router.get("/nodes")
    async def nodes():
        eng = srv.engine
        out = []
        for i in range(eng.n_nodes):
            state, conf = _latest_node_state(eng, i)
            out.append({
                "index": i,
                "name": srv.node_label(i),
                "type": srv.node_type(i),
                "state": state,
                "confidence": conf,
            })
        return {"nodes": out}

    @router.get("/topology")
    async def topology_view():
        eng = srv.engine
        topo = srv._topo_nodes
        n_render = max(eng.n_nodes, len(topo))
        out = []
        for i in range(n_render):
            state, conf = _latest_node_state(eng, i)
            entry = {
                "index": i,
                "id": srv.node_id(i),
                "label": srv.node_label(i),
                "type": srv.node_type(i),
                "tier": topo[i].get("tier", "unknown") if i < len(topo) else "unknown",
                "state": state,
                "confidence": conf,
            }
            out.append(entry)
        return {"nodes": out}

    @router.get("/gnn")
    async def gnn_structure():
        topo = srv._topo_nodes
        edges = srv._topo_edges
        stats = _structural_stats(topo, edges)
        out = []
        for i, node in enumerate(topo):
            s = stats[i] if i < len(stats) else {"connections": 0, "hard_deps": 0, "criticality": "NONE"}
            out.append({
                "name": node.get("label", node.get("id", f"Node {i:02d}")),
                "type": node.get("type", "UNKNOWN"),
                "tier": node.get("tier", "unknown"),
                "connections": s["connections"],
                "hard_deps": s["hard_deps"],
                "criticality": s["criticality"],
            })
        return {"nodes": out}

    @router.get("/pomdp")
    async def pomdp_view():
        eng = srv.engine
        if not eng.history:
            return {"nodes": []}
        views = (eng.history[-1].get("pillar_feed") or {}).get("pillar_views", [])
        out = []
        for i, pv in enumerate(views):
            belief = pv.get("pomdp_belief", {}) or {}
            out.append({
                "name": srv.node_label(i),
                "p_healthy": round(float(belief.get("healthy", 0.0)), 4),
                "p_degraded": round(float(belief.get("degraded", 0.0)), 4),
                "p_failed": round(float(belief.get("failed", 0.0)), 4),
                "confidence": round(float(pv.get("pomdp_confidence", 0.0)), 4),
                "obs_age": round(float(pv.get("pomdp_obs_age", 0.0)), 4),
            })
        return {"nodes": out}

    @router.get("/temporal")
    async def temporal_view():
        eng = srv.engine
        if not eng.history:
            return {"nodes": []}
        views = (eng.history[-1].get("pillar_feed") or {}).get("pillar_views", [])
        out = []
        for i, pv in enumerate(views):
            probs = pv.get("trend_probs", {}) or {}
            out.append({
                "name": srv.node_label(i),
                "trend": pv.get("trend", "stable"),
                "p_improving": round(float(probs.get("improving", 0.0)), 4),
                "p_stable": round(float(probs.get("stable", 0.0)), 4),
                "p_deteriorating": round(float(probs.get("deteriorating", 0.0)), 4),
            })
        return {"nodes": out}

    @router.get("/recommendations")
    async def recommendations_view():
        recs = srv.enrich_recommendations(srv.engine.get_recommendations())
        actions = []
        for a in recs.get("actions", []):
            actions.append({
                "priority": a.get("priority", 0),
                "target": a.get("target", ""),
                "action": a.get("action", ""),
                "reason": a.get("reason", ""),
                "fix": a.get("recommendation", ""),
            })
        return {"actions": actions, "summary": recs.get("summary", "")}

    return router
