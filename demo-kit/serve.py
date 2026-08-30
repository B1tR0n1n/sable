#!/usr/bin/env python3
"""
SABLE — live interactive demo server.

Loads the honest (de-leaked) model once and diagnoses topologies on demand.
Pick a scenario, click nodes to set what monitoring observes, hit Diagnose,
and the real model infers the hidden failures through the dependency graph.

Run:   python3 serve.py      →   open http://localhost:8760
"""
import json
import os
import re
import sys
import threading
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ── Security config ──
# Bind localhost by default; set SABLE_HOST=0.0.0.0 to expose on the LAN.
# When exposed, set SABLE_TOKEN to require an X-SABLE-Token header on every
# mutating endpoint (diagnose / scan). Names must match this pattern — no path
# traversal into arbitrary .json files.
HOST = os.environ.get("SABLE_HOST", "127.0.0.1")
PORT = int(os.environ.get("SABLE_PORT", "8760"))
TOKEN = os.environ.get("SABLE_TOKEN")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_scan_lock = threading.Lock()


def _require_token(x_sable_token: str | None):
    """Enforce the shared token when one is configured."""
    if TOKEN and x_sable_token != TOKEN:
        raise HTTPException(status_code=401, detail="invalid or missing X-SABLE-Token")


def _safe_topology_path(name: str) -> Path:
    """Resolve a topology name to a path inside TOPO_DIR, or reject it."""
    if not _SAFE_NAME.match(name or ""):
        raise HTTPException(status_code=400, detail="invalid topology name")
    path = (TOPO_DIR / f"{name}.json").resolve()
    if path.parent != TOPO_DIR.resolve() or not path.exists():
        raise HTTPException(status_code=404, detail="topology not found")
    return path

SABLE = Path(__file__).resolve().parent.parent
for p in [SABLE / "adapters", SABLE / "fusion", SABLE / "pillar1", SABLE / "pillar3",
          SABLE / "pillar1/archive/cortex-moved-2026-04-08"]:
    sys.path.insert(0, str(p))

import topology_ingest as TI
from staged_fusion_v3 import SharpRoutedFusion
from cortex_gnn_model import SableGNN
from generate_temporal_data import NODE_FEAT_DIM
from sable_sim.core.states import STATE_NAMES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TIER = {
    "INTERNET_GATEWAY": "external", "WAN_LINK": "external",
    "FIREWALL": "core", "CORE_SWITCH": "core",
    "ACCESS_SWITCH": "access", "LOAD_BALANCER": "access",
    "HYPERVISOR": "compute", "SERVER_PHYSICAL": "compute",
    "SERVER_VIRTUAL": "compute", "VDI_HOST": "compute",
    "STORAGE_ARRAY": "storage", "STORAGE_TARGET": "storage",
    "DNS_SERVER": "services", "DHCP_SERVER": "services", "DOMAIN_CONTROLLER": "services",
    "CERTIFICATE_AUTHORITY": "services", "MONITORING_SERVER": "services", "VDI_BROKER": "services",
    "APPLICATION_SERVICE": "application",
}
LABEL = {
    "INTERNET_GATEWAY": "Internet GW", "FIREWALL": "Firewall", "CORE_SWITCH": "Core Switch",
    "ACCESS_SWITCH": "Access Switch", "HYPERVISOR": "Hypervisor", "SERVER_VIRTUAL": "VM",
    "STORAGE_ARRAY": "Storage Array", "STORAGE_TARGET": "Storage Tgt", "DNS_SERVER": "DNS",
    "DOMAIN_CONTROLLER": "Domain Ctrl", "APPLICATION_SERVICE": "App Service",
    "MONITORING_SERVER": "Monitoring", "INTERNET_GATEWAY ": "Internet GW",
}

# ── load models once ──
print(f"  Loading honest SABLE model on {DEVICE}...")
_gck = torch.load(SABLE / "pillar1/checkpoints/best_model.pt", weights_only=False, map_location=DEVICE)
_mc = _gck["config"]
GNN = SableGNN(in_dim=_mc["in_dim"], hidden_dim=_mc["hidden_dim"], edge_dim=_mc["edge_dim"],
               num_layers=_mc["num_layers"], heads=_mc["heads"], dropout=_mc["dropout"]).to(DEVICE)
_gsd = GNN.state_dict()
GNN.load_state_dict({k: v for k, v in _gck["model_state_dict"].items()
                     if k in _gsd and _gsd[k].shape == v.shape}, strict=False)
GNN.eval()
FUSION = SharpRoutedFusion(mamba_dim=NODE_FEAT_DIM).to(DEVICE)
FUSION.load_state_dict(torch.load(SABLE / "fusion/checkpoints/staged_fusion_v3.pt",
                                  weights_only=False, map_location=DEVICE)["model_state_dict"])
FUSION.eval()
print("  Model ready.")

app = FastAPI()
TOPO_DIR = SABLE / "adapters"


def list_topologies():
    return sorted(p.stem for p in TOPO_DIR.glob("*.json"))


def diagnose(name: str, observations: dict | None):
    path = _safe_topology_path(name)
    graph, default_obs, disp = TI.load_topology(path)
    obs = observations if observations is not None else default_obs
    if not isinstance(obs, dict):
        raise HTTPException(status_code=400, detail="observations must be an object")
    obs = {k: v for k, v in obs.items() if v and v != "unobserved"}
    bad = {v for v in obs.values() if v not in TI._VALID_STATES}
    if bad:
        raise HTTPException(status_code=400,
                            detail=f"invalid observation state(s): {sorted(bad)}")

    gnn_t, pomdp_t, mamba_t, cids, observed = TI.encode(graph, obs, GNN, DEVICE)
    with torch.no_grad():
        probs = torch.softmax(FUSION(gnn_t, pomdp_t, mamba_t)["logits"][0], dim=-1)
        preds, conf = probs.argmax(-1), probs.max(-1).values

    comps = {c.id: c for c in graph.get_all_components()}
    nodes = []
    for i, cid in enumerate(cids):
        ctype = str(comps[cid].type)
        st = STATE_NAMES[preds[i].item()]
        is_obs = cid in observed
        nodes.append({
            "id": cid, "type": ctype,
            "label": LABEL.get(ctype, ctype.replace("_", " ").title()),
            "tier": TIER.get(ctype, "compute"),
            "observed": is_obs, "observed_state": obs.get(cid) if is_obs else None,
            "state": st, "confidence": round(conf[i].item(), 3),
            "inferred_problem": (not is_obs) and st != "healthy",
        })
    edges = [{"source": d.source_id, "target": d.target_id, "type": str(d.type)}
             for d in graph.get_all_dependencies()]
    return {
        "name": disp, "nodes": nodes, "edges": edges,
        "n_observed": len(observed), "n_hidden": len(nodes) - len(observed),
        "n_inferred_problems": sum(1 for n in nodes if n["inferred_problem"]),
        "default_observations": default_obs,
    }


class DiagReq(BaseModel):
    topology: str
    observations: dict | None = None


@app.get("/api/topologies")
def topologies():
    return JSONResponse(list_topologies())


@app.post("/api/diagnose")
def api_diagnose(req: DiagReq, x_sable_token: str | None = Header(default=None)):
    _require_token(x_sable_token)
    return JSONResponse(diagnose(req.topology, req.observations))


class ScanReq(BaseModel):
    community: str | None = None   # SNMP community for L2 (LLDP/CDP); None = skip


@app.post("/api/scan")
def api_scan(req: ScanReq, x_sable_token: str | None = Header(default=None)):
    """Live-scan the network, build a SABLE topology, save it, and diagnose."""
    _require_token(x_sable_token)
    if req.community is not None and not re.match(r"^[\x20-\x7e]{1,64}$", req.community):
        raise HTTPException(status_code=400, detail="invalid SNMP community")
    # One scan at a time — never let requests stack overlapping nmap/snmp trees.
    if not _scan_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="a scan is already running")
    try:
        import network_discovery as ND
        topo = ND.discover(community=req.community)
        clean = {
            "name": topo["name"],
            "components": [{"id": c["id"], "type": c["type"]} for c in topo["components"]],
            "dependencies": [{k: e[k] for k in ("source", "target", "type") if k in e}
                             for e in topo["dependencies"]],
            "observations": topo["observations"],
        }
        (TOPO_DIR / "discovered_topology.json").write_text(json.dumps(clean))
        result = diagnose("discovered_topology", None)
        result["scanned"] = topo["meta"]
        return JSONResponse(result)
    finally:
        _scan_lock.release()


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "live.html").read_text()


if __name__ == "__main__":
    print(f"\n  SABLE live demo →  http://{HOST}:{PORT}\n")
    if HOST != "127.0.0.1" and not TOKEN:
        print("  [!] WARNING: bound to a non-local address with no SABLE_TOKEN set — "
              "scan/diagnose endpoints are UNAUTHENTICATED on the LAN.\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
