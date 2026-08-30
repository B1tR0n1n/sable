#!/usr/bin/env python3
"""
Real-topology ingest for SABLE.

Turns a client's real dependency graph (JSON, e.g. exported from a CMDB /
discovery tool) plus their current monitoring observations into a live
diagnosis — inferring the state of the nodes monitoring CAN'T see by
propagating the observed failures through the topology (the GNN's job).

JSON schema:
{
  "name": "acme-dc1",
  "components":   [{"id": "core-sw-1", "type": "CORE_SWITCH"}, ...],
  "dependencies": [{"source": "access-sw-1", "target": "core-sw-1",
                    "type": "NETWORK_PATH", "criticality": "HARD"}, ...],
  "observations": [{"id": "app-portal", "state": "failed"}, ...]   # optional
}

Component/dependency `type` strings must match the ComponentType /
DependencyType enums. `state` is one of healthy|degraded|failed|unreachable.
Nodes with no observation are treated as UNOBSERVED (hidden from all channels).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

SABLE = Path(__file__).resolve().parent.parent
for p in [SABLE, SABLE / "fusion", SABLE / "pillar1", SABLE / "pillar2",
          SABLE / "pillar3", SABLE / "pillar1/archive/cortex-moved-2026-04-08"]:
    sys.path.insert(0, str(p))

from sable_sim.core.component import Component, ComponentType, ComponentState, DEFAULT_PROPERTIES
from sable_sim.core.dependency import Dependency, DependencyType, Criticality
from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.states import STATE_NAMES
from pomcp import BeliefState
from generate_temporal_data import encode_system_state, NODE_FEAT_DIM, N_STATES
from shared_latent_space import GNN_DIM, POMDP_DIM
from staged_fusion_v3 import SharpRoutedFusion
from cortex_gnn_model import SableGNN

# Same taxonomies the (fixed) encoders use.
from build_infra_dataset import NODE_TYPES, N_NODE_TYPES, EDGE_TYPES, N_EDGE_TYPES, EDGE_FEAT_DIM as INFRA_EDGE_FEAT_DIM
INFRA_NODE_FEAT_DIM = N_NODE_TYPES + 2

COMP_TYPE_MAP = {
    "CORE_SWITCH": NODE_TYPES["router"], "ACCESS_SWITCH": NODE_TYPES["switch"],
    "FIREWALL": NODE_TYPES["router"], "ROUTER": NODE_TYPES["router"],
    "LOAD_BALANCER": NODE_TYPES["switch"], "SERVER_PHYSICAL": NODE_TYPES["server"],
    "SERVER_VIRTUAL": NODE_TYPES["server"], "HYPERVISOR": NODE_TYPES["server"],
    "STORAGE_ARRAY": NODE_TYPES["storage"], "STORAGE_TARGET": NODE_TYPES["storage"],
    "VDI_BROKER": NODE_TYPES["service"], "VDI_HOST": NODE_TYPES["server"],
    "DNS_SERVER": NODE_TYPES["service"], "DHCP_SERVER": NODE_TYPES["service"],
    "DOMAIN_CONTROLLER": NODE_TYPES["service"], "CERTIFICATE_AUTHORITY": NODE_TYPES["service"],
    "MONITORING_SERVER": NODE_TYPES["service"], "WAN_LINK": NODE_TYPES["gateway"],
    "INTERNET_GATEWAY": NODE_TYPES["gateway"], "APPLICATION_SERVICE": NODE_TYPES["server"],
}
DEP_TYPE_MAP = {
    "NETWORK_PATH": EDGE_TYPES["backbone"], "HOSTING_DEPENDENCY": EDGE_TYPES["access"],
    "STORAGE_DEPENDENCY": EDGE_TYPES["storage_dep"], "SERVICE_DEPENDENCY": EDGE_TYPES["service_dep"],
    "DNS_DEPENDENCY": EDGE_TYPES["service_dep"], "AUTHENTICATION_DEPENDENCY": EDGE_TYPES["service_dep"],
    "REPLICATION_DEPENDENCY": EDGE_TYPES["service_dep"], "MONITORING_DEPENDENCY": EDGE_TYPES["management"],
}
OBS_HEALTH = {"healthy": 1.0, "degraded": 0.4, "failed": 0.0, "unreachable": 0.0}


_VALID_STATES = set(STATE_NAMES[:N_STATES]) | {"unobserved"}


def _enum(enum_cls, value, field: str):
    """Look up an enum member by name, raising a clear ValueError on a bad key.

    Topology JSON is untrusted file content (a CMDB/discovery export). A bad
    type string must surface as an actionable error, not a raw KeyError 500.
    """
    try:
        return enum_cls[value]
    except (KeyError, TypeError):
        raise ValueError(
            f"invalid {field} '{value}'; expected one of: "
            f"{', '.join(m.name for m in enum_cls)}"
        )


def load_topology(path: str) -> tuple[InfrastructureGraph, dict]:
    """Parse a topology JSON into an InfrastructureGraph + observations dict."""
    spec = json.loads(Path(path).read_text())
    if not isinstance(spec.get("components"), list):
        raise ValueError("topology JSON missing a 'components' list")
    graph = InfrastructureGraph()
    for c in spec["components"]:
        ctype = _enum(ComponentType, c["type"], "component type")
        props = dict(DEFAULT_PROPERTIES.get(ctype, {}))
        graph.add_component(Component(id=str(c["id"]), type=ctype, properties=props))
    for d in spec.get("dependencies", []):
        graph.add_dependency(Dependency(
            source_id=str(d["source"]), target_id=str(d["target"]),
            type=_enum(DependencyType, d.get("type", "NETWORK_PATH"), "dependency type"),
            criticality=_enum(Criticality, d.get("criticality", "HARD"), "criticality"),
        ))
    observations = {}
    for o in spec.get("observations", []):
        state = o["state"]
        if state not in _VALID_STATES:
            raise ValueError(
                f"invalid observation state '{state}' for '{o.get('id')}'; "
                f"expected one of: {', '.join(sorted(_VALID_STATES))}"
            )
        observations[str(o["id"])] = state
    return graph, observations, spec.get("name", "topology")


def encode(graph: InfrastructureGraph, observations: dict, gnn: SableGNN, device="cuda"):
    """Encode a topology + partial observations into (gnn, pomdp, mamba) tensors.

    Observed nodes reveal their state; unobserved nodes are hidden from every
    channel — the model must infer them from observed neighbours via the graph.
    """
    components = graph.get_all_components()
    component_ids = [c.id for c in components]
    n = len(component_ids)
    cid_to_idx = {cid: i for i, cid in enumerate(component_ids)}
    observed_ids = set(observations)

    # Belief from the given observations (fog is real here — the monitoring)
    belief = BeliefState(component_ids)
    for cid, state in observations.items():
        belief.update_from_observation(cid, state, 0.85)
    belief.propagate_beliefs(graph)

    # GNN node features — observation-gated health + typed nodes
    degrees = {c.id: len(c.dependencies_in) + len(c.dependencies_out) for c in components}
    max_deg = max(degrees.values()) if degrees else 1
    nf = np.zeros((n, INFRA_NODE_FEAT_DIM), dtype=np.float32)
    for i, comp in enumerate(components):
        health = OBS_HEALTH.get(observations.get(comp.id), 0.5)  # unobserved -> 0.5
        nf[i, COMP_TYPE_MAP.get(str(comp.type), NODE_TYPES["unknown"])] = 1.0
        nf[i, N_NODE_TYPES] = degrees[comp.id] / max(max_deg, 1)
        nf[i, N_NODE_TYPES + 1] = health
    x = torch.tensor(nf, device=device)

    sources, targets, efeat = [], [], []
    for comp in components:
        si = cid_to_idx[comp.id]
        for dep_id in comp.dependencies_in:
            ti = cid_to_idx.get(dep_id)
            if ti is None:
                continue
            dep = graph.get_dependency(dep_id, comp.id)
            if not dep:
                continue
            f = np.zeros(INFRA_EDGE_FEAT_DIM, dtype=np.float32)
            f[DEP_TYPE_MAP.get(str(dep.type), EDGE_TYPES["unknown"])] = 1.0
            f[N_EDGE_TYPES] = 1.0
            sources.append(si); targets.append(ti); efeat.append(f)

    gnn_out = torch.zeros(n, GNN_DIM)
    if sources:
        ei = torch.tensor([sources, targets], dtype=torch.long, device=device)
        ea = torch.tensor(np.array(efeat), device=device)
        with torch.no_grad():
            emb = gnn.encode(x, ei, ea).cpu()
        gnn_out[:, :min(emb.size(1), GNN_DIM)] = emb[:, :GNN_DIM]

    # POMDP belief vector (contradiction flag left 0 — no temporal instability signal here)
    pomdp_out = torch.zeros(n, POMDP_DIM)
    for i, cid in enumerate(component_ids):
        b = belief.beliefs[cid]
        conf = 1.0 - belief.entropy(cid) / 2.0
        hub = min(len(graph.get_dependents(cid)) / 10.0, 1.0)
        obs_age = 0.0 if cid in observed_ids else 0.5
        pomdp_out[i] = torch.tensor([b[0], b[1], b[2], b[3], conf, obs_age, 0.0, hub])

    # Mamba — observation-based (belief), same as training
    raw = encode_system_state(graph, component_ids, belief=belief)
    mamba_out = torch.tensor(raw, dtype=torch.float32).reshape(n, -1)[:, :NODE_FEAT_DIM]

    return (gnn_out.unsqueeze(0).to(device), pomdp_out.unsqueeze(0).to(device),
            mamba_out.unsqueeze(0).to(device), component_ids, observed_ids)


def diagnose(topology_path: str, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    graph, observations, name = load_topology(topology_path)

    # Load the trained GNN (structural) + retrained fusion (honest, de-leaked)
    gck = torch.load(SABLE / "pillar1/checkpoints/best_model.pt", weights_only=False, map_location=device)
    mc = gck["config"]
    gnn = SableGNN(in_dim=mc["in_dim"], hidden_dim=mc["hidden_dim"], edge_dim=mc["edge_dim"],
                   num_layers=mc["num_layers"], heads=mc["heads"], dropout=mc["dropout"]).to(device)
    gsd = gnn.state_dict()
    gnn.load_state_dict({k: v for k, v in gck["model_state_dict"].items()
                         if k in gsd and gsd[k].shape == v.shape}, strict=False)
    gnn.eval()

    fusion = SharpRoutedFusion(mamba_dim=NODE_FEAT_DIM).to(device)
    fusion.load_state_dict(torch.load(SABLE / "fusion/checkpoints/staged_fusion_v3.pt",
                                      weights_only=False, map_location=device)["model_state_dict"])
    fusion.eval()

    gnn_t, pomdp_t, mamba_t, cids, observed = encode(graph, observations, gnn, device)
    with torch.no_grad():
        out = fusion(gnn_t, pomdp_t, mamba_t)
        probs = torch.softmax(out["logits"][0], dim=-1)
        preds = probs.argmax(-1)
        conf = probs.max(-1).values

    print(f"\n  SABLE — diagnosis of '{name}'  ({len(cids)} components, "
          f"{len(observed)} observed / {len(cids) - len(observed)} hidden)\n")
    flagged = []
    for i, cid in enumerate(cids):
        st = STATE_NAMES[preds[i].item()]
        if st != "healthy":
            seen = "observed" if cid in observed else "INFERRED (hidden)"
            flagged.append((cid, st, conf[i].item(), seen))
    if not flagged:
        print("    All components healthy.")
    else:
        flagged.sort(key=lambda x: (x[3] != "INFERRED (hidden)", -x[2]))
        print(f"    {'component':22} {'state':12} {'conf':>6}  source")
        print("    " + "-" * 60)
        for cid, st, c, seen in flagged:
            print(f"    {cid:22} {st:12} {c*100:5.0f}%  {seen}")
        hidden_flags = [f for f in flagged if f[3].startswith('INFERRED')]
        print(f"\n    {len(hidden_flags)} problem(s) inferred in nodes monitoring couldn't see.")
    return flagged


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(SABLE / "adapters/sample_topology.json")
    diagnose(path)
