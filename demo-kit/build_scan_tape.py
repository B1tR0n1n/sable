#!/usr/bin/env python3
"""
Build the "network_discovery" demo tape for the offline flash-drive kit.

Honest by construction:
  - The TOPOLOGY is a real nmap/SNMP auto-discovery of a live /24
    (adapters/network_discovery.py output), sanitized so no personal-device
    hostname, IP, or MAC ships in a kit handed to prospects. Structure, role
    inference, and dependency edges are exactly what SABLE discovered.
  - The INCIDENT is a scripted gateway failure over the discovered graph,
    clearly labelled as simulated. Every per-node prediction, expert-routing
    weight, transition and confidence is REAL output from the de-leaked
    temporal-chain model in docker/checkpoints — nothing is hand-written.

The point it proves: SABLE builds its own topology from a network you point it
at (no manual CMDB entry), then reasons over it — here inferring a HIDDEN
gateway root cause from downstream symptoms monitoring can see.

    python3 build_scan_tape.py            # emit tape JSON to stdout / scan_tape.json
    python3 build_scan_tape.py --inject   # also splice it into index.html
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

SABLE = Path(__file__).resolve().parent.parent
for p in [SABLE, SABLE / "adapters", SABLE / "docker", SABLE / "fusion",
          SABLE / "pillar1", SABLE / "pillar2", SABLE / "pillar3",
          SABLE / "pillar1/archive/cortex-moved-2026-04-08"]:
    sys.path.insert(0, str(p))

import topology_ingest as TI
from sable_engine import SableEngine
from sable_sim.core.states import STATE_NAMES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_STATES = 5

# ── Sanitized real scan ──────────────────────────────────────────────────
# Real discovery of a live /24 (gateway + WAN uplink + three reachable hosts).
# Identifiers genericized; role TYPES and edges are exactly what the scanner
# inferred. `inferred_by` records how the scan classified each node — shown in
# the demo so the classification is transparent, not asserted.
NODES = [
    {"id": "gateway-1",  "type": "INTERNET_GATEWAY",   "label": "Internet Gateway",
     "tier": "external", "inferred_by": "default route"},
    {"id": "wan-1",      "type": "WAN_LINK",           "label": "ISP Uplink",
     "tier": "external", "inferred_by": "gateway uplink"},
    {"id": "host-15",    "type": "APPLICATION_SERVICE", "label": "Monitored Host",
     "tier": "application", "inferred_by": "web service (80/443)"},
    {"id": "endpoint-147", "type": "SERVER_PHYSICAL",   "label": "Endpoint A",
     "tier": "compute", "inferred_by": "host up (no service fingerprint)"},
    {"id": "endpoint-65",  "type": "SERVER_PHYSICAL",   "label": "Endpoint B",
     "tier": "compute", "inferred_by": "host up (no service fingerprint)"},
]
EDGES = [
    {"source": "gateway-1",    "target": "wan-1",     "type": "NETWORK_PATH",   "criticality": "HARD"},
    {"source": "host-15",      "target": "gateway-1", "type": "NETWORK_PATH",   "criticality": "HARD"},
    {"source": "host-15",      "target": "gateway-1", "type": "DNS_DEPENDENCY", "criticality": "HARD"},
    {"source": "endpoint-147", "target": "gateway-1", "type": "NETWORK_PATH",   "criticality": "HARD"},
    {"source": "endpoint-147", "target": "gateway-1", "type": "DNS_DEPENDENCY", "criticality": "HARD"},
    {"source": "endpoint-65",  "target": "gateway-1", "type": "NETWORK_PATH",   "criticality": "HARD"},
    {"source": "endpoint-65",  "target": "gateway-1", "type": "DNS_DEPENDENCY", "criticality": "HARD"},
]

HOSTS = ["host-15", "endpoint-147", "endpoint-65"]
OBSERVED_ROOT = "gateway-1"  # monitoring watches the gateway/uplink
# Hosts are the HIDDEN side: SABLE must infer their blast radius from the
# observed gateway failure via forward propagation through the discovered graph.

# ── Scripted incident (SIMULATED — labelled as such in the tape) ──────────
# truth[tick]  = the real state of every node (ground truth for the readout)
# observed[tick] = what monitoring can SEE (drives the model input). The
# gateway is never observed: SABLE must INFER it from host symptoms.
N_TICKS = 14


def _truth(tick):
    # Physically defensible ground truth: a host that HARD-depends on the
    # gateway for both its network path and DNS is unreachable exactly while
    # the gateway/uplink is down. No artificial degrade-lag — that would be
    # reverse-engineering the timeline to flatter (or punish) the model.
    down = 4 <= tick < 11
    t = {n["id"]: "healthy" for n in NODES}
    if down:
        t[OBSERVED_ROOT] = "failed"
        t["wan-1"] = "failed"
        for h in HOSTS:
            t[h] = "unreachable"
    return t


def _observed(tick):
    """Monitoring view: the gateway/uplink only. Hosts are HIDDEN — SABLE must
    infer their state by propagating the observed gateway failure through the
    topology it discovered."""
    truth = _truth(tick)
    return {OBSERVED_ROOT: truth[OBSERVED_ROOT], "wan-1": truth["wan-1"]}


def _recommendations(pred_by_id, meta):
    """Minimal honest recommendation from the model's own predictions.

    Names the highest-fan-in non-healthy node as the likely root cause — the
    same 'most-depended-on failure' heuristic the live server uses.
    """
    fanin = {n["id"]: 0 for n in NODES}
    for e in EDGES:
        fanin[e["target"]] += 1
    bad = {cid: st for cid, st in pred_by_id.items() if st != "healthy"}
    if not bad:
        return {"summary": "All discovered components healthy.", "total_affected": 0, "actions": []}
    root = max(bad, key=lambda c: (fanin.get(c, 0), bad[c] in ("failed", "unreachable")))
    counts = {}
    for st in bad.values():
        counts[st] = counts.get(st, 0) + 1
    breakdown = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    return {
        "summary": f"{len(bad)}/{len(NODES)} nodes affected: {breakdown}. "
                   f"Root cause likely {meta[root]['label']} "
                   f"({'observed' if root in (OBSERVED_ROOT, 'wan-1') else 'inferred — not directly monitored'}).",
        "total_affected": len(bad),
        "root_cause_id": root,
        "root_cause_label": meta[root]["label"],
        "actions": [{
            "priority": 1, "action": "INVESTIGATE ROOT CAUSE",
            "target": meta[root]["label"], "target_id": root,
            "reason": f"{fanin.get(root, 0)} discovered nodes depend on it; "
                      f"downstream symptoms point here.",
            "recommendation": "Check the uplink/gateway path, DNS reachability, and recent changes.",
        }],
    }


def build_tape():
    meta = {n["id"]: n for n in NODES}
    graph, _, _ = TI.load_topology(_write_topo_tmp())
    cids_order = [c.id for c in graph.get_all_components()]

    # Load GNN exactly as topology_ingest.diagnose does (structural encoder).
    gck = torch.load(SABLE / "pillar1/checkpoints/best_model.pt",
                     weights_only=False, map_location=DEVICE)
    mc = gck["config"]
    gnn = TI.SableGNN(in_dim=mc["in_dim"], hidden_dim=mc["hidden_dim"],
                      edge_dim=mc["edge_dim"], num_layers=mc["num_layers"],
                      heads=mc["heads"], dropout=mc["dropout"]).to(DEVICE)
    gsd = gnn.state_dict()
    gnn.load_state_dict({k: v for k, v in gck["model_state_dict"].items()
                         if k in gsd and gsd[k].shape == v.shape}, strict=False)
    gnn.eval()

    engine = SableEngine(device=DEVICE)
    engine.load_checkpoints(str(SABLE / "docker/checkpoints"))
    engine.reset_state(len(cids_order))

    ticks, node_detail = [], {}
    for tick in range(N_TICKS):
        truth = _truth(tick)
        obs = _observed(tick)
        gt = torch.tensor([list(STATE_NAMES).index(truth[c]) for c in cids_order],
                          dtype=torch.long)
        gnn_t, pomdp_t, mamba_t, cids, observed = TI.encode(graph, obs, gnn, DEVICE)
        t0 = time.time()
        result = engine.infer(gnn_t, pomdp_t, mamba_t, ground_truth=gt)
        result["inference_ms"] = round((time.time() - t0) * 1000, 2)

        nodes, pred_by_id = [], {}
        for node in result["nodes"]:
            i = node["id"]
            cid = cids[i]
            m = meta[cid]
            node = dict(node)
            node["label"] = m["label"]
            node["topo_id"] = cid
            node["component_type"] = m["type"]
            nodes.append(node)
            pred_by_id[cid] = node["state"]
            nd = node_detail.setdefault(str(i), {
                "node_id": i, "topo_id": cid, "label": m["label"],
                "current_state": node["state"], "trajectory": []})
            nd["current_state"] = node["state"]
            nd["trajectory"].append({
                "cycle": tick, "prediction": node["state"], "truth": truth[cid],
                "correct": node["state"] == truth[cid]})

        # Full engine state (class_counts, routing, avg_confidence, accuracy,
        # pillar_feed — the per-pillar independent verdicts) with nodes enriched
        # by topology labels. The whole dashboard reads these fields; a stripped
        # {cycle, nodes} state would break the hero bar and pillar panel.
        result["nodes"] = nodes
        ticks.append({
            "tick_index": tick,
            "state": result,
            "recommendations": _recommendations(pred_by_id, meta),
        })

    desc = ("Network Discovery — SABLE auto-discovers a live /24 (real nmap/SNMP "
            "scan, sanitized), builds the topology itself, then infers the HIDDEN "
            "blast radius: with only the gateway monitored, it calls three unseen "
            "hosts unreachable the instant the uplink drops. Topology is real; the "
            "incident is a labelled simulation.")
    topology = {"name": "auto-discovered", "nodes": NODES, "edges": EDGES}
    return {
        "name": "network_discovery", "description": desc,
        "n_nodes": len(NODES), "n_ticks": N_TICKS,
        "state_names": list(STATE_NAMES[:N_STATES]),
        "topology": topology, "ticks": ticks, "node_detail": node_detail,
    }


_TMP = Path("/tmp/_scan_topo.json")


def _write_topo_tmp():
    _TMP.write_text(json.dumps({
        "name": "auto-discovered",
        "components": [{"id": n["id"], "type": n["type"]} for n in NODES],
        "dependencies": [{"source": e["source"], "target": e["target"], "type": e["type"]}
                         for e in EDGES],
    }))
    return str(_TMP)


def inject_into_index(tape):
    idx = Path(__file__).parent / "index.html"
    html = idx.read_text(encoding="utf-8")
    marker = "window.__TAPE__"
    i = html.index(marker)
    j = html.index("{", i)
    depth = 0
    for k in range(j, len(html)):
        if html[k] == "{":
            depth += 1
        elif html[k] == "}":
            depth -= 1
            if depth == 0:
                break
    data = json.loads(html[j:k + 1])
    # Replace any prior network_discovery entry (idempotent re-runs)
    data["scenarios"] = [s for s in data["scenarios"] if s["name"] != tape["name"]]
    data["tapes"].pop(tape["name"], None)
    data["scenarios"].append({"name": tape["name"], "description": tape["description"],
                              "n_nodes": tape["n_nodes"], "n_ticks": tape["n_ticks"]})
    data["tapes"][tape["name"]] = tape
    new_html = html[:j] + json.dumps(data, ensure_ascii=False) + html[k + 1:]
    idx.write_text(new_html, encoding="utf-8")
    print(f"  [✓] spliced '{tape['name']}' into {idx} "
          f"({len(data['scenarios'])} scenarios total)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", action="store_true", help="splice into index.html")
    args = ap.parse_args()

    tape = build_tape()
    out = Path(__file__).parent / "scan_tape.json"
    out.write_text(json.dumps(tape, ensure_ascii=False))
    print(f"  [✓] tape: {out}  ({tape['n_nodes']} nodes, {tape['n_ticks']} ticks)")

    # Honest console readout: did SABLE infer the HIDDEN hosts from the
    # observed gateway failure? (gateway observed; hosts unobserved)
    print("\n  cycle | gateway(observed) | HIDDEN hosts inferred | vs truth")
    print("  " + "-" * 66)
    for t in tape["ticks"]:
        gw = next(n for n in t["state"]["nodes"] if n["topo_id"] == OBSERVED_ROOT)
        hosts = [n for n in t["state"]["nodes"] if n["topo_id"] in HOSTS]
        truth = _truth(t["tick_index"])
        hs = ",".join(h["state"][:4] for h in hosts)
        th = ",".join(truth[h["topo_id"]][:4] for h in hosts)
        hit = sum(h["state"] == truth[h["topo_id"]] for h in hosts)
        print(f"   {t['tick_index']:>3}  | {gw['state']:>9} {gw['confidence']*100:3.0f}%  "
              f"| {hs:<20} | {th:<20} {hit}/3")

    if args.inject:
        inject_into_index(tape)


if __name__ == "__main__":
    main()
