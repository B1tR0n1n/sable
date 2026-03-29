#!/usr/bin/env -S python -u
"""
Project PARALLAX — Generate Temporal Sequence Training Data
==============================================================
Runs sable_sim cascades and captures per-tick pillar features
for training the temporal chain.

Each sequence = one cascade scenario with T ticks.
At each tick: (gnn_features, pomdp_features, mamba_features, ground_truth_states)

Output: temporal_sequences.pt

Usage:
    python generate_temporal_sequences.py --count 2000 --ticks 12
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar3"))

from sable_sim.core.states import N_STATES, STATE_NAMES, STATE_MAP, OSCILLATING_CLASS
from shared_latent_space import GNN_DIM, POMDP_DIM, Z_DIM
from generate_temporal_data import (
    build_random_topology, encode_system_state, NODE_FEAT_DIM,
)
from build_infra_dataset import (
    NODE_TYPES, EDGE_TYPES, N_NODE_TYPES, N_EDGE_TYPES,
    NODE_FEAT_DIM as INFRA_NODE_FEAT_DIM, EDGE_FEAT_DIM as INFRA_EDGE_FEAT_DIM,
)
from cortex_gnn_model import SableGNN
from pomcp import BeliefState
from sable_sim.core.component import ComponentState
from sable_sim.core.state import SystemState
from sable_sim.simulation.propagation import PropagationEngine
from sable_sim.simulation.failure_injection import FailureInjector
from sable_sim.simulation.fog import FogOfWar
from sable_sim.utils.random import SeededRandom

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"

# Same component type mapping as shared_latent_space fusion data generator
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

MAX_NODES = 40


def encode_gnn_features(graph, components, component_ids, gnn, device, rng=None):
    """Encode infrastructure topology through trained GNN encoder.

    Health values are noisy — GNN sees approximate health, not ground truth.
    """
    n = len(components)
    cid_to_idx = {cid: i for i, cid in enumerate(component_ids)}

    degrees = {}
    for comp in components:
        degrees[comp.id] = len(comp.dependencies_in) + len(comp.dependencies_out)
    max_deg = max(degrees.values()) if degrees else 1

    # Node features (infra-native 10-dim) — with noisy health
    node_features = np.zeros((n, INFRA_NODE_FEAT_DIM), dtype=np.float32)
    for i, comp in enumerate(components):
        ntype = COMP_TYPE_MAP.get(str(comp.type), NODE_TYPES["unknown"])
        node_features[i, ntype] = 1.0
        node_features[i, N_NODE_TYPES] = degrees[comp.id] / max(max_deg, 1)
        # Noisy health: add noise, 25% of nodes get masked to 0.5
        health = comp.health
        if rng is not None:
            health = max(0.0, min(1.0, health + rng.uniform(-0.15, 0.15)))
            if rng.random() < 0.25:
                health = 0.5  # uncertain
        node_features[i, N_NODE_TYPES + 1] = health

    # Edge features
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
                    sources.append(si)
                    targets.append(ti)
                    edge_feats.append(feat)

    if not sources:
        return torch.zeros(n, GNN_DIM)

    x = torch.tensor(node_features, dtype=torch.float32).to(device)
    ei = torch.tensor([sources, targets], dtype=torch.long).to(device)
    ea = torch.tensor(np.array(edge_feats), dtype=torch.float32).to(device)

    with torch.no_grad():
        emb = gnn.encode(x, ei, ea).cpu()  # (n, hidden_dim)

    # Pad to GNN_DIM if needed
    if emb.size(1) < GNN_DIM:
        padded = torch.zeros(n, GNN_DIM)
        padded[:, :emb.size(1)] = emb
        return padded
    return emb[:, :GNN_DIM]


def encode_pomdp_features(belief_state, component_ids):
    """Extract POMDP belief vectors as tensor features."""
    n = len(component_ids)
    features = np.zeros((n, POMDP_DIM), dtype=np.float32)
    for i, cid in enumerate(component_ids):
        b = belief_state.beliefs.get(cid, np.array([1, 0, 0, 0], dtype=np.float32))
        features[i, :4] = b
        features[i, 4] = 1.0 - min(belief_state.observation_age.get(cid, 0) / 5.0, 1.0)
        features[i, 5] = belief_state.observation_age.get(cid, 0) / 10.0
    return torch.tensor(features, dtype=torch.float32)


def _load_gnn(device):
    """Load trained GNN model for encoding."""
    gnn_path = Path(__file__).parent.parent / "pillar1" / "checkpoints" / "best_model.pt"
    ckpt = torch.load(gnn_path, weights_only=False, map_location=device)
    mc = ckpt["config"]
    gnn = SableGNN(
        in_dim=mc["in_dim"], hidden_dim=mc["hidden_dim"],
        edge_dim=mc["edge_dim"], num_layers=mc["num_layers"],
        heads=mc["heads"], dropout=mc["dropout"],
    ).to(device)
    model_state = gnn.state_dict()
    for k, v in ckpt["model_state_dict"].items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
    gnn.load_state_dict(model_state)
    gnn.eval()
    return gnn, mc


def _setup_scenario(rng, max_nodes):
    """Build topology and inject failures. Returns (graph, components, ids, use_deg, osc_nodes) or None."""
    graph = build_random_topology(rng, 15, max_nodes)
    components = graph.get_all_components()
    if len(components) < 15:
        return None
    component_ids = [c.id for c in components]
    n = len(component_ids)

    state = SystemState(graph)
    skip_injection = rng.random() < 0.12
    use_deg = False

    if not skip_injection:
        injector = FailureInjector(rng)
        try:
            injections = injector.generate_failures(
                state, difficulty=rng.weighted_choice(["easy", "medium", "hard"], [0.2, 0.5, 0.3])
            )
        except Exception:
            return None
        if not injections:
            return None

        use_deg = rng.random() < 0.4
        for inj in injections:
            injector.inject(state, inj)
            if use_deg:
                comp = graph.get_component(inj.component_id)
                if comp and comp.state == ComponentState.FAILED:
                    comp.state = ComponentState.DEGRADED
                    comp.health = rng.uniform(0.25, 0.45)

    # 30% of injected scenarios: oscillating nodes
    osc_nodes = set()
    if not skip_injection and rng.random() < 0.30:
        n_osc = rng.choice([3, 4, 5, 6])
        osc_candidates = list(range(min(n, 15)))
        if len(osc_candidates) >= n_osc:
            osc_nodes = set(np.random.default_rng(rng.randint(0, 2**31)).choice(
                osc_candidates, size=n_osc, replace=False).tolist())

    return graph, components, component_ids, state, use_deg, osc_nodes


def _flip_oscillating_nodes(graph, component_ids, osc_nodes, tick, rng):
    """Flip oscillating node states at varied cadences."""
    for oi in osc_nodes:
        comp = graph.get_component(component_ids[oi])
        if comp and tick > 0 and tick % rng.choice([1, 1, 2, 2, 3]) == 0:
            if comp.state == ComponentState.HEALTHY:
                comp.state = ComponentState.FAILED
                comp.health = 0.0
            elif comp.state in (ComponentState.FAILED, ComponentState.DEGRADED):
                comp.state = ComponentState.HEALTHY
                comp.health = 1.0


def _detect_oscillation_and_store_history(graph, component_ids, n, tick, generated,
                                           ds_gt_history, gt_states):
    """Detect oscillating nodes and store ground truth history for delayed obs."""
    if tick >= 2:
        for ci in range(n):
            flips = 0
            for dt in range(1, min(4, tick + 1)):
                prev_key = (generated, tick - dt, component_ids[ci])
                prev_s = ds_gt_history.get(prev_key)
                if dt == 1:
                    curr_s = str(graph.get_component(component_ids[ci]).state).split(".")[-1].lower()
                else:
                    curr_s = ds_gt_history.get((generated, tick - dt + 1, component_ids[ci]))
                if prev_s and curr_s and prev_s != curr_s:
                    flips += 1
            if flips >= 2:
                gt_states[ci] = OSCILLATING_CLASS

    for ci, cid in enumerate(component_ids):
        state_name = str(graph.get_component(cid).state).split(".")[-1].lower()
        ds_gt_history[(generated, tick, cid)] = state_name

    return gt_states


def _corrupt_mamba_features(mamba_raw, n):
    """Add noise and masking to Mamba features for training robustness."""
    mamba_tensor = torch.tensor(mamba_raw, dtype=torch.float32).reshape(n, -1)[:, :NODE_FEAT_DIM]
    health_noise = torch.randn(n) * 0.15
    mamba_tensor[:, 0] = (mamba_tensor[:, 0] + health_noise).clamp(0, 1)
    mask_nodes = torch.rand(n) < 0.20
    mamba_tensor[mask_nodes, 0] = 0.5
    return mamba_tensor


def _update_belief_fog(belief, graph, component_ids, rng, tick, generated, ds_gt_history):
    """Update belief state with hard fog-of-war: 30% observable, 15% noise, 2-tick delay."""
    for i, cid in enumerate(component_ids):
        comp = graph.get_component(cid)
        if rng.random() < 0.30:
            state_name = str(comp.state).split(".")[-1].lower()
            if rng.random() < 0.15:
                state_name = rng.choice(STATE_NAMES[:4])
            if tick >= 2:
                delayed_gt = ds_gt_history.get((generated, tick - 2, cid), state_name)
                state_name = delayed_gt
            belief.update_from_observation(cid, state_name)
    belief.age_observations()


def _propagate_one_tick(state, engine):
    """Propagate one tick of cascade and return whether changes occurred."""
    current_tick = state.tick
    changes = []
    changed_ids = set()
    prev_changes = state.get_changes_at_tick(current_tick)
    changed_components = {
        c.component_id: state.graph.get_component(c.component_id)
        for c in prev_changes
    }
    for cid, comp in changed_components.items():
        if comp is None:
            continue
        for dependent in state.graph.get_dependents(cid):
            if dependent.id in changed_ids:
                continue
            dep = state.graph.get_dependency(dependent.id, cid)
            if dep is None:
                continue
            sc = engine._evaluate_impact(state, comp, dependent, dep, current_tick)
            if sc is not None:
                changes.append(sc)
                changed_ids.add(dependent.id)

    new_tick = state.advance_tick()
    for sc in changes:
        sc.tick = new_tick
        state.record_change(sc)

    return len(changes) > 0 or not state.has_reached_steady_state(lookback=2)


def _save_and_print_stats(all_gnn, all_pomdp, all_mamba, all_states, all_mask,
                           all_seq_len, all_n_nodes, max_ticks):
    """Save dataset and print statistics."""
    dataset = {
        "gnn": all_gnn,
        "pomdp": all_pomdp,
        "mamba": all_mamba,
        "states": all_states,
        "mask": all_mask,
        "seq_len": all_seq_len,
        "n_nodes": all_n_nodes,
        "max_ticks": max_ticks,
        "max_nodes": MAX_NODES,
    }

    out_path = Path(__file__).parent / "temporal_sequences.pt"
    torch.save(dataset, str(out_path))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  {C_SUCCESS}{C_BOLD}Saved:{C_RESET} {C_TEXT}{out_path} ({size_mb:.1f} MB){C_RESET}")

    avg_len = all_seq_len.float().mean().item()
    print(f"  {C_TEXT}Avg sequence length: {C_BRIGHT}{avg_len:.1f} ticks{C_RESET}")
    valid_states = all_states[all_mask > 0].long()
    print(f"  {C_INFO}State distribution across all ticks:{C_RESET}")
    for s in range(N_STATES):
        n_s = (valid_states == s).sum().item()
        print(f"    {C_DIM}{STATE_NAMES[s]:12s}: {n_s:8d} ({n_s/valid_states.size(0)*100:.1f}%){C_RESET}")
    print()


def generate_temporal_sequences(count: int, max_ticks: int = 12,
                                device: str = "cpu", seed: int = 42):
    """Generate temporal sequences for training the chain."""
    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Temporal Sequence Generation{C_RESET}")
    print(f"  {C_DIM}{'═' * 50}{C_RESET}\n")

    gnn, mc = _load_gnn(device)
    print(f"  {C_TEXT}GNN loaded: hidden_dim={mc['hidden_dim']}{C_RESET}", flush=True)

    rng = SeededRandom(seed)

    all_gnn = torch.zeros(count, max_ticks, MAX_NODES, GNN_DIM)
    all_pomdp = torch.zeros(count, max_ticks, MAX_NODES, POMDP_DIM)
    all_mamba = torch.zeros(count, max_ticks, MAX_NODES, NODE_FEAT_DIM)
    all_states = torch.zeros(count, max_ticks, MAX_NODES, dtype=torch.long)
    all_mask = torch.zeros(count, max_ticks, MAX_NODES)
    all_seq_len = torch.zeros(count, dtype=torch.long)
    all_n_nodes = torch.zeros(count, dtype=torch.long)

    ds_gt_history = {}
    generated = 0
    t0 = time.time()

    while generated < count:
        result = _setup_scenario(rng, MAX_NODES)
        if result is None:
            continue
        graph, components, component_ids, state, use_deg, osc_nodes = result
        n = len(component_ids)

        belief = BeliefState(component_ids)
        engine = PropagationEngine(max_ticks=max_ticks, soft_impact_factor=0.4 if use_deg else 0.3)

        tick_count = 0
        for tick in range(max_ticks):
            _flip_oscillating_nodes(graph, component_ids, osc_nodes, tick, rng)

            gnn_feat = encode_gnn_features(graph, components, component_ids, gnn, device, rng=rng)
            pomdp_feat = encode_pomdp_features(belief, component_ids)
            mamba_raw = encode_system_state(graph, component_ids)

            gt_states = [STATE_MAP.get(graph.get_component(cid).state, 0) for cid in component_ids]
            gt_states = _detect_oscillation_and_store_history(
                graph, component_ids, n, tick, generated, ds_gt_history, gt_states)

            mamba_tensor = _corrupt_mamba_features(mamba_raw, n)

            all_gnn[generated, tick, :n] = gnn_feat
            all_pomdp[generated, tick, :n] = pomdp_feat
            all_mamba[generated, tick, :n] = mamba_tensor
            all_states[generated, tick, :n] = torch.tensor(gt_states, dtype=torch.long)
            all_mask[generated, tick, :n] = 1.0

            _update_belief_fog(belief, graph, component_ids, rng, tick, generated, ds_gt_history)
            tick_count += 1

            if not _propagate_one_tick(state, engine):
                break

        all_seq_len[generated] = tick_count
        all_n_nodes[generated] = n
        ds_gt_history.clear()
        generated += 1

        if generated % 200 == 0:
            rate = generated / (time.time() - t0)
            print(f"    {C_DIM}{generated:5d}/{count} ({rate:.0f}/s){C_RESET}", flush=True)

    elapsed = time.time() - t0
    print(f"  {C_TEXT}Generated {generated} sequences in {elapsed:.1f}s{C_RESET}", flush=True)

    _save_and_print_stats(all_gnn, all_pomdp, all_mamba, all_states, all_mask,
                          all_seq_len, all_n_nodes, max_ticks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--ticks", type=int, default=12)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_temporal_sequences(args.count, args.ticks, args.device, args.seed)
