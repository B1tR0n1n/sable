#!/usr/bin/env -S python -u
"""
Pre-compute demo scenarios using the EXACT same feature pipeline as training.
Uses generate_fusion_data's encoding path for feature alignment.
Controlled cascade injection with hub-node failures + manual state evolution
for scenarios that need specific patterns (whiplash, slow poison).
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

SABLE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SABLE_ROOT))
sys.path.insert(0, str(SABLE_ROOT / "fusion"))
sys.path.insert(0, str(SABLE_ROOT / "pillar1"))
sys.path.insert(0, str(SABLE_ROOT / "pillar2"))
sys.path.insert(0, str(SABLE_ROOT / "pillar3"))

from sable_sim.core.states import N_STATES, STATE_NAMES, STATE_MAP, OSCILLATING_CLASS
from sable_sim.core.component import ComponentState
from sable_sim.core.state import SystemState, StateChange
from sable_sim.simulation.propagation import PropagationEngine
from sable_sim.simulation.failure_injection import FailureInjector
from sable_sim.simulation.fog import FogOfWar
from sable_sim.utils.random import SeededRandom
from shared_latent_space import GNN_DIM, POMDP_DIM
from generate_temporal_data import build_random_topology, encode_system_state, NODE_FEAT_DIM
from build_infra_dataset import NODE_TYPES, EDGE_TYPES, N_NODE_TYPES, N_EDGE_TYPES
from build_infra_dataset import NODE_FEAT_DIM as INFRA_NODE_FEAT_DIM, EDGE_FEAT_DIM as INFRA_EDGE_FEAT_DIM
from cortex_gnn_model import SableGNN
from pomcp import BeliefState

C = "\033[38;2;201;162;39m"
R = "\033[0m"

# Same maps as generate_fusion_data in shared_latent_space.py
COMP_TYPE_MAP = {
    "ComponentType.CORE_SWITCH": NODE_TYPES["router"],
    "ComponentType.ACCESS_SWITCH": NODE_TYPES["switch"],
    "ComponentType.FIREWALL": NODE_TYPES["router"],
    "ComponentType.ROUTER": NODE_TYPES["router"],
    "ComponentType.LOAD_BALANCER": NODE_TYPES["switch"],
    "ComponentType.SERVER_PHYSICAL": NODE_TYPES["server"],
    "ComponentType.SERVER_VIRTUAL": NODE_TYPES["server"],
    "ComponentType.HYPERVISOR": NODE_TYPES["server"],
    "ComponentType.STORAGE_ARRAY": NODE_TYPES["storage"],
    "ComponentType.STORAGE_TARGET": NODE_TYPES["storage"],
    "ComponentType.VDI_BROKER": NODE_TYPES["service"],
    "ComponentType.VDI_HOST": NODE_TYPES["server"],
    "ComponentType.DNS_SERVER": NODE_TYPES["service"],
    "ComponentType.DHCP_SERVER": NODE_TYPES["service"],
    "ComponentType.DOMAIN_CONTROLLER": NODE_TYPES["service"],
    "ComponentType.CERTIFICATE_AUTHORITY": NODE_TYPES["service"],
    "ComponentType.MONITORING_SERVER": NODE_TYPES["service"],
    "ComponentType.WAN_LINK": NODE_TYPES["gateway"],
    "ComponentType.INTERNET_GATEWAY": NODE_TYPES["gateway"],
    "ComponentType.APPLICATION_SERVICE": NODE_TYPES["server"],
}
DEP_TYPE_MAP = {
    "DependencyType.HARD": EDGE_TYPES["backbone"],
    "DependencyType.SOFT": EDGE_TYPES["access"],
    "DependencyType.SERVICE": EDGE_TYPES["service_dep"],
    "DependencyType.RESOURCE": EDGE_TYPES["storage_dep"],
}


def load_gnn(device="cpu"):
    gnn_path = SABLE_ROOT / "pillar1" / "checkpoints" / "best_model.pt"
    ckpt = torch.load(gnn_path, weights_only=False, map_location=device)
    mc = ckpt["config"]
    gnn = SableGNN(
        in_dim=mc["in_dim"], hidden_dim=mc["hidden_dim"],
        edge_dim=mc["edge_dim"], num_layers=mc["num_layers"],
        heads=mc["heads"], dropout=mc["dropout"],
    ).to(device)
    state_dict = gnn.state_dict()
    for k, v in ckpt["model_state_dict"].items():
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k] = v
    gnn.load_state_dict(state_dict)
    gnn.eval()
    return gnn


def encode_tick(graph, components, component_ids, gnn, device, rng, belief):
    """Encode one tick using the EXACT same feature pipeline as training.
    Returns (gnn_feat, pomdp_feat, mamba_feat, gt_states) all as tensors.
    """
    n = len(component_ids)
    cid_to_idx = {cid: i for i, cid in enumerate(component_ids)}

    # --- GNN: same path as generate_fusion_data lines 712-758 ---
    degrees = {comp.id: len(comp.dependencies_in) + len(comp.dependencies_out) for comp in components}
    max_deg = max(degrees.values()) if degrees else 1

    node_features = np.zeros((n, INFRA_NODE_FEAT_DIM), dtype=np.float32)
    for i, comp in enumerate(components):
        ntype = COMP_TYPE_MAP.get(str(comp.type), NODE_TYPES["unknown"])
        node_features[i, ntype] = 1.0
        node_features[i, N_NODE_TYPES] = degrees[comp.id] / max(max_deg, 1)
        node_features[i, N_NODE_TYPES + 1] = comp.health
    x = torch.tensor(node_features, dtype=torch.float32).to(device)

    sources, targets, edge_feats = [], [], []
    for comp in components:
        si = cid_to_idx[comp.id]
        for dep_id in comp.dependencies_in:
            ti = cid_to_idx.get(dep_id)
            if ti is not None:
                dep = graph.get_dependency(comp.id, dep_id)
                if dep:
                    etype = DEP_TYPE_MAP.get(str(dep.type), EDGE_TYPES["unknown"])
                    feat = np.zeros(INFRA_EDGE_FEAT_DIM, dtype=np.float32)
                    feat[etype] = 1.0
                    feat[N_EDGE_TYPES] = 1.0
                    sources.append(si); targets.append(ti); edge_feats.append(feat)

    gnn_out = torch.zeros(n, GNN_DIM)
    if sources:
        ei = torch.tensor([sources, targets], dtype=torch.long).to(device)
        ea = torch.tensor(np.array(edge_feats), dtype=torch.float32).to(device)
        with torch.no_grad():
            emb = gnn.encode(x, ei, ea).cpu()
        if emb.size(1) < GNN_DIM:
            gnn_out[:, :emb.size(1)] = emb
        else:
            gnn_out = emb[:, :GNN_DIM]

    # --- POMDP: same path as generate_fusion_data lines 760-781 ---
    fog = FogOfWar(monitoring_coverage=0.5, rng=SeededRandom(rng.randint(0, 2**31)))
    state_obj = SystemState(graph)
    operator_view = fog.generate_operator_view(state_obj)

    for obs in operator_view.get("observations", []):
        belief.update_from_observation(obs["component_id"], obs["observed_state"], 0.85)
    belief.propagate_beliefs(graph)

    pomdp_out = torch.zeros(n, POMDP_DIM)
    for i, cid in enumerate(component_ids):
        b = belief.beliefs[cid]
        conf = 1.0 - belief.entropy(cid) / 2.0
        obs_age = float(belief.observation_age.get(cid, 5))
        hub = min(len(graph.get_dependents(cid)) / 10.0, 1.0)
        pomdp_out[i] = torch.tensor([b[0], b[1], b[2], b[3], conf, obs_age / 10.0, 0.0, hub])

    # --- Mamba: same path as generate_fusion_data line 783-786 ---
    raw = encode_system_state(graph, component_ids)
    mamba_out = torch.tensor(raw, dtype=torch.float32).reshape(n, -1)[:, :NODE_FEAT_DIM]

    # --- Ground truth ---
    gt = torch.tensor([STATE_MAP.get(graph.get_component(cid).state, 0) for cid in component_ids], dtype=torch.long)

    return gnn_out, pomdp_out, mamba_out, gt


def generate_scenario(name, desc, seed, n_ticks, inject_fn, gnn, device="cpu"):
    """Generate a multi-tick scenario with training-aligned features.

    inject_fn(graph, state, components, component_ids, rng, tick) -> bool
        Called each tick. Mutates graph component states.
        Returns True if state was modified this tick.
    """
    rng = SeededRandom(seed)
    graph = build_random_topology(rng, 20, 40)
    components = graph.get_all_components()
    component_ids = [c.id for c in components]
    n = len(component_ids)

    state = SystemState(graph)
    belief = BeliefState(component_ids)

    all_gnn = torch.zeros(n_ticks, n, GNN_DIM)
    all_pomdp = torch.zeros(n_ticks, n, POMDP_DIM)
    all_mamba = torch.zeros(n_ticks, n, NODE_FEAT_DIM)
    all_gt = torch.zeros(n_ticks, n, dtype=torch.long)

    for tick in range(n_ticks):
        # Let the inject function modify state this tick
        inject_fn(graph, state, components, component_ids, rng, tick)

        # Encode with training-aligned pipeline
        gnn_f, pomdp_f, mamba_f, gt = encode_tick(
            graph, components, component_ids, gnn, device, rng, belief
        )

        all_gnn[tick, :n] = gnn_f
        all_pomdp[tick, :n] = pomdp_f
        all_mamba[tick, :n] = mamba_f
        all_gt[tick, :n] = gt

        belief.age_observations()

    return {
        "name": name, "description": desc,
        "n_nodes": n, "n_ticks": n_ticks,
        "gnn": all_gnn.unsqueeze(0),
        "pomdp": all_pomdp.unsqueeze(0),
        "mamba": all_mamba.unsqueeze(0),
        "ground_truth": all_gt,
    }


def _set_component_states(graph, cids, target_state, health):
    """Helper: set state and health on a list of component IDs."""
    for cid in cids:
        c = graph.get_component(cid)
        if c:
            c.state = target_state
            c.health = health


def _set_healthy_only(graph, cids, target_state, health):
    """Helper: set state/health only on components currently HEALTHY."""
    for cid in cids:
        c = graph.get_component(cid)
        if c and c.state == ComponentState.HEALTHY:
            c.state = target_state
            c.health = health


def _get_hub_ranked_ids(graph, comps):
    """Return component IDs sorted by dependent count (descending)."""
    degrees = {c.id: len(graph.get_dependents(c.id)) for c in comps}
    return sorted(degrees, key=degrees.get, reverse=True)


def _inject_monday_morning(graph, state, comps, cids, rng, tick):
    """Monday Morning Meltdown — progressive cascade injection."""
    hubs = _get_hub_ranked_ids(graph, comps)
    if tick == 3:
        _set_component_states(graph, hubs[5:7], ComponentState.DEGRADED, 0.4)
    elif tick == 6:
        _set_component_states(graph, hubs[5:7], ComponentState.FAILED, 0.0)
        _set_healthy_only(graph, hubs[2:5], ComponentState.DEGRADED, 0.5)
    elif tick == 10:
        _set_component_states(graph, hubs[2:5], ComponentState.FAILED, 0.0)
        _set_component_states(graph, hubs[7:10], ComponentState.UNREACHABLE, 0.0)
    elif tick == 16:
        _set_component_states(graph, hubs[5:7], ComponentState.DEGRADED, 0.5)
    elif tick == 20:
        _set_component_states(graph, hubs[5:7], ComponentState.HEALTHY, 1.0)
        _set_component_states(graph, hubs[2:4], ComponentState.HEALTHY, 1.0)


def _inject_silent_killer(graph, state, comps, cids, rng, tick):
    """Silent Killer — hub fails, downstream cascades without direct observation."""
    degrees = {c.id: len(graph.get_dependents(c.id)) for c in comps}
    hub = max(degrees, key=degrees.get)
    deps = [c.id for c in graph.get_dependents(hub)]
    if tick == 3:
        _set_component_states(graph, [hub], ComponentState.FAILED, 0.0)
    elif tick == 7:
        _set_healthy_only(graph, deps[:3], ComponentState.DEGRADED, 0.4)
    elif tick == 11:
        _set_component_states(graph, deps[:2], ComponentState.FAILED, 0.0)
        _set_healthy_only(graph, deps[3:6], ComponentState.UNREACHABLE, 0.0)


def _inject_cascade_whiplash(graph, state, comps, cids, rng, tick):
    """Cascade Whiplash — root recovers but dependents stay dead."""
    root = cids[0]
    deps = cids[1:6]
    if tick == 2:
        _set_component_states(graph, [root], ComponentState.FAILED, 0.0)
        _set_component_states(graph, deps, ComponentState.DEGRADED, 0.4)
    elif tick == 5:
        _set_component_states(graph, deps[:3], ComponentState.FAILED, 0.0)
    elif tick == 8:
        _set_component_states(graph, [root], ComponentState.HEALTHY, 1.0)
    elif tick == 11:
        _set_component_states(graph, deps[3:], ComponentState.HEALTHY, 1.0)


def _inject_slow_poison(graph, state, comps, cids, rng, tick):
    """Slow Poison — gradual health decay, no sudden failures."""
    for cid in cids[:8]:
        c = graph.get_component(cid)
        if c:
            h = max(0.0, 1.0 - tick * 0.05)
            c.health = h
            if h < 0.3:
                c.state = ComponentState.FAILED
            elif h < 0.6:
                c.state = ComponentState.DEGRADED
            else:
                c.state = ComponentState.HEALTHY


def _inject_random_chaos(graph, state, comps, cids, rng, tick):
    """Random Chaos — multiple staged failures with partial recovery."""
    if tick == 3:
        _set_component_states(graph, cids[:3], ComponentState.FAILED, 0.0)
    if tick == 8:
        _set_component_states(graph, cids[5:9], ComponentState.DEGRADED, 0.3)
    if tick == 12:
        _set_component_states(graph, cids[10:14], ComponentState.UNREACHABLE, 0.0)
    if tick == 15:
        _set_component_states(graph, cids[:2], ComponentState.HEALTHY, 1.0)


def _save_and_report(scenarios, out_dir):
    """Save scenario tensors and print summary."""
    for s in scenarios:
        path = out_dir / f"{s['name']}.pt"
        torch.save(s, path)
        gt = s["ground_truth"]
        aff = (gt != 0).sum().item()
        tot = gt.numel()
        print(f"  {s['name']:<25s} {s['n_nodes']:2d}n {s['n_ticks']:2d}t "
              f"{aff}/{tot} ({aff/tot*100:.0f}%) {path.stat().st_size//1024}KB", flush=True)
    print(f"\n  Saved {len(scenarios)} scenarios to {out_dir}/\n", flush=True)


def main():
    print(f"\n  {C}SABLE — Pre-computing Demo Scenarios{R}\n", flush=True)
    gnn = load_gnn()
    out_dir = Path(__file__).parent / "scenarios"
    out_dir.mkdir(exist_ok=True)

    scenario_defs = [
        ("monday_morning",
         "Monday Morning Meltdown — DB degrades at tick 3, app cascade at 6, "
         "services unreachable by 10, partial recovery at 16.",
         400, 25, _inject_monday_morning),
        ("silent_killer",
         "Silent Killer — Hub fails at tick 3. No direct observation. "
         "Downstream degrades at 7, failures at 11.",
         999, 20, _inject_silent_killer),
        ("cascade_whiplash",
         "Cascade Whiplash — Root fails at 2, cascade at 5, root recovers at 8. "
         "Dependents stay dead.",
         777, 15, _inject_cascade_whiplash),
        ("slow_poison",
         "Slow Poison — 8 nodes lose 5% health per tick. No sudden failures. "
         "Only trajectory tracking catches the trend.",
         5555, 20, _inject_slow_poison),
        ("random_chaos",
         "Random Chaos — Failures at 3, degradation at 8, unreachable at 12, "
         "partial recovery at 15.",
         1234, 20, _inject_random_chaos),
    ]

    scenarios = [
        generate_scenario(name, desc, seed, n_ticks, inject_fn, gnn)
        for name, desc, seed, n_ticks, inject_fn in scenario_defs
    ]
    _save_and_report(scenarios, out_dir)


if __name__ == "__main__":
    main()
