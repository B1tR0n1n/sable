#!/usr/bin/env -S python -u
"""
Pre-compute SABLE scenarios from real SMD (Server Machine Dataset) telemetry.

Pipes 28 real server machines through the telemetry adapter -> health scorer
-> pillar encoder -> GNN inference -> scenario tensors. No sable_sim in the
loop. This is real data hitting the model.
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

from sable_sim.core.states import N_STATES, STATE_NAMES, STATE_MAP
from shared_latent_space import GNN_DIM, POMDP_DIM
from generate_temporal_data import NODE_FEAT_DIM
from cortex_gnn_model import SableGNN
from build_infra_dataset import (
    NODE_TYPES, N_NODE_TYPES, EDGE_TYPES, N_EDGE_TYPES,
    NODE_FEAT_DIM as INFRA_NODE_FEAT_DIM, EDGE_FEAT_DIM as INFRA_EDGE_FEAT_DIM,
)
from adapters.smd import SMDAdapter, FEATURE_GROUPS
from adapters.base import COMPONENT_TYPE_INDEX

C = "\033[38;2;201;162;39m"
D = "\033[38;2;138;130;114m"
R = "\033[0m"

# Map SABLE component types to infra GNN node types
COMP_TO_NODE_TYPE = {
    "CORE_SWITCH": NODE_TYPES["switch"],
    "ACCESS_SWITCH": NODE_TYPES["switch"],
    "FIREWALL": NODE_TYPES["router"],
    "ROUTER": NODE_TYPES["router"],
    "LOAD_BALANCER": NODE_TYPES["switch"],
    "SERVER_PHYSICAL": NODE_TYPES["server"],
    "SERVER_VIRTUAL": NODE_TYPES["server"],
    "HYPERVISOR": NODE_TYPES["server"],
    "STORAGE_ARRAY": NODE_TYPES["storage"],
    "STORAGE_TARGET": NODE_TYPES["storage"],
    "DNS_SERVER": NODE_TYPES["service"],
    "DOMAIN_CONTROLLER": NODE_TYPES["service"],
    "MONITORING_SERVER": NODE_TYPES["service"],
    "APPLICATION_SERVICE": NODE_TYPES["server"],
}

# Map dependency types to edge types
DEP_TO_EDGE_TYPE = {
    "NETWORK_PATH": EDGE_TYPES["access"],
    "HOSTING_DEPENDENCY": EDGE_TYPES["backbone"],
    "SERVICE_DEPENDENCY": EDGE_TYPES["service_dep"],
    "DNS_DEPENDENCY": EDGE_TYPES["service_dep"],
    "STORAGE_DEPENDENCY": EDGE_TYPES["storage_dep"],
    "AUTHENTICATION_DEPENDENCY": EDGE_TYPES["service_dep"],
    "MONITORING_DEPENDENCY": EDGE_TYPES["management"],
    "REPLICATION_DEPENDENCY": EDGE_TYPES["backbone"],
}

STATE_NAME_TO_IDX = {name: i for i, name in enumerate(STATE_NAMES)}


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


def encode_smd_tick(snapshot, gnn, device):
    """Convert an SMD SystemSnapshot into pillar tensors.

    This is the real data path - no sable_sim encoding.
    GNN features come from the infrastructure adapter.
    POMDP features come from the health scorer output.
    Mamba features use the same 26-dim format as training.
    """
    node_ids = sorted(snapshot.node_ids)
    n = len(node_ids)
    nid_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # --- GNN: infrastructure node features (10-dim) ---
    node_features = np.zeros((n, INFRA_NODE_FEAT_DIM), dtype=np.float32)
    degrees = {}
    for edge in snapshot.edges:
        degrees[edge.source_id] = degrees.get(edge.source_id, 0) + 1
        degrees[edge.target_id] = degrees.get(edge.target_id, 0) + 1
    max_deg = max(degrees.values()) if degrees else 1

    for i, nid in enumerate(node_ids):
        node = snapshot.nodes[nid]
        ntype = COMP_TO_NODE_TYPE.get(node.component_type, NODE_TYPES["server"])
        node_features[i, ntype] = 1.0
        node_features[i, N_NODE_TYPES] = degrees.get(nid, 0) / max(max_deg, 1)
        node_features[i, N_NODE_TYPES + 1] = node.health if node.health is not None else 1.0

    x = torch.tensor(node_features, dtype=torch.float32).to(device)

    # Build edges
    sources, targets, edge_feats = [], [], []
    for edge in snapshot.edges:
        si = nid_to_idx.get(edge.source_id)
        ti = nid_to_idx.get(edge.target_id)
        if si is None or ti is None:
            continue
        etype = DEP_TO_EDGE_TYPE.get(edge.dep_type, EDGE_TYPES["unknown"])
        feat = np.zeros(INFRA_EDGE_FEAT_DIM, dtype=np.float32)
        feat[etype] = 1.0
        feat[N_EDGE_TYPES] = 1.0
        sources.append(si)
        targets.append(ti)
        edge_feats.append(feat)

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

    # --- POMDP: belief vectors from health scorer output ---
    pomdp_out = torch.zeros(n, POMDP_DIM)
    for i, nid in enumerate(node_ids):
        node = snapshot.nodes[nid]
        health = node.health if node.health is not None else 1.0
        state = node.state or "healthy"

        # State probability distribution based on health
        probs = np.zeros(4, dtype=np.float32)
        si = STATE_NAME_TO_IDX.get(state, 0)
        if si < 4:
            probs[si] = 0.7 + 0.2 * health
            remaining = 1.0 - probs[si]
            for j in range(4):
                if j != si:
                    probs[j] = remaining / 3.0
        else:
            probs[0] = 0.3
            probs[1] = 0.5
            probs[2] = 0.2

        is_anomaly = node.labels.get("anomaly", "False") == "True"
        confidence = 0.6 if is_anomaly else (0.8 + 0.2 * health)
        obs_age = 0.1 if not is_anomaly else 0.3
        hub = degrees.get(nid, 0) / max(max_deg, 1)

        pomdp_out[i] = torch.tensor([probs[0], probs[1], probs[2], probs[3],
                                      confidence, obs_age, 0.0, hub])

    # --- Mamba: 26-dim per node (health + state_onehot + type_onehot) ---
    mamba_out = torch.zeros(n, NODE_FEAT_DIM)
    for i, nid in enumerate(node_ids):
        node = snapshot.nodes[nid]
        health = node.health if node.health is not None else 1.0
        mamba_out[i, 0] = health

        # State one-hot (5 states)
        si = STATE_NAME_TO_IDX.get(node.state or "healthy", 0)
        mamba_out[i, 1 + si] = 1.0

        # Type one-hot (20 types)
        ti = COMPONENT_TYPE_INDEX.get(node.component_type, 0)
        mamba_out[i, 6 + ti] = 1.0

    # --- Ground truth: derive from anomaly labels + health ---
    gt = torch.zeros(n, dtype=torch.long)
    for i, nid in enumerate(node_ids):
        node = snapshot.nodes[nid]
        si = STATE_NAME_TO_IDX.get(node.state or "healthy", 0)
        gt[i] = si

    return gnn_out, pomdp_out, mamba_out, gt, node_ids


def generate_smd_scenario(name, desc, adapter, start_tick, n_ticks, gnn, device="cpu"):
    """Generate a SABLE scenario from a window of SMD data."""
    first_snap = adapter.get_tick(start_tick)
    node_ids = sorted(first_snap.node_ids)
    n = len(node_ids)

    all_gnn = torch.zeros(n_ticks, n, GNN_DIM)
    all_pomdp = torch.zeros(n_ticks, n, POMDP_DIM)
    all_mamba = torch.zeros(n_ticks, n, NODE_FEAT_DIM)
    all_gt = torch.zeros(n_ticks, n, dtype=torch.long)

    anomaly_ticks = 0
    for t in range(n_ticks):
        tick = start_tick + t
        if tick >= adapter.n_ticks:
            break

        snapshot = adapter.get_tick(tick)
        gnn_f, pomdp_f, mamba_f, gt, _ = encode_smd_tick(snapshot, gnn, device)

        all_gnn[t, :n] = gnn_f
        all_pomdp[t, :n] = pomdp_f
        all_mamba[t, :n] = mamba_f
        all_gt[t, :n] = gt

        # Track anomaly presence
        gt_labels = adapter.get_ground_truth(tick)
        if any(gt_labels.values()):
            anomaly_ticks += 1

    return {
        "name": name,
        "description": desc,
        "n_nodes": n,
        "n_ticks": n_ticks,
        "gnn": all_gnn.unsqueeze(0),
        "pomdp": all_pomdp.unsqueeze(0),
        "mamba": all_mamba.unsqueeze(0),
        "ground_truth": all_gt,
        "source": "smd",
        "anomaly_ticks": anomaly_ticks,
    }


def find_anomaly_windows(adapter, min_length=15):
    """Find windows in the SMD test data that contain anomalies."""
    windows = []
    in_anomaly = False
    start = 0

    for tick in range(adapter.n_ticks):
        gt = adapter.get_ground_truth(tick)
        has_anomaly = any(gt.values())

        if has_anomaly and not in_anomaly:
            start = max(0, tick - 5)  # Start 5 ticks before anomaly
            in_anomaly = True
        elif not has_anomaly and in_anomaly:
            length = tick - start + 5  # End 5 ticks after
            if length >= min_length:
                windows.append((start, min(length, 30)))
            in_anomaly = False

    return windows


def main():
    print(f"\n  {C}SABLE - Pre-computing SMD Scenarios (Real Telemetry){R}\n")

    smd_path = SABLE_ROOT / "data" / "smd" / "OmniAnomaly" / "ServerMachineDataset"
    if not smd_path.exists():
        print(f"  SMD data not found at {smd_path}")
        return

    adapter = SMDAdapter(smd_path)
    adapter.load(split="test", max_ticks=2000)  # First 2000 ticks
    print(f"  Loaded {len(adapter.machines)} machines, {adapter.n_ticks} ticks")
    print(f"  Topology: {len(adapter._edges)} edges")

    gnn = load_gnn()
    out_dir = Path(__file__).parent / "scenarios"
    out_dir.mkdir(exist_ok=True)

    # Find interesting anomaly windows
    print(f"  Scanning for anomaly windows...", flush=True)
    windows = find_anomaly_windows(adapter, min_length=15)
    print(f"  Found {len(windows)} anomaly windows")

    scenarios = []

    # Scenario 1: Normal operation (first 25 ticks, likely clean)
    print(f"\n  Generating scenarios...", flush=True)
    s = generate_smd_scenario(
        "smd_normal", "Real server telemetry - normal operation (28 machines, 38 metrics)",
        adapter, 0, 25, gnn,
    )
    scenarios.append(s)
    print(f"    smd_normal: {s['anomaly_ticks']}/{s['n_ticks']} anomaly ticks")

    # Scenario 2-4: Anomaly windows
    for i, (start, length) in enumerate(windows[:3]):
        name = f"smd_incident_{i+1}"
        desc = f"Real server incident - anomaly window starting at tick {start} (28 machines)"
        s = generate_smd_scenario(name, desc, adapter, start, min(length, 25), gnn)
        scenarios.append(s)
        print(f"    {name}: {s['anomaly_ticks']}/{s['n_ticks']} anomaly ticks (start={start})")

    # Scenario 5: Extended window covering multiple incidents
    if len(windows) >= 2:
        start = windows[0][0]
        end = min(windows[1][0] + windows[1][1], adapter.n_ticks)
        length = min(end - start, 30)
        s = generate_smd_scenario(
            "smd_cascade", f"Real multi-incident cascade starting at tick {start}",
            adapter, start, length, gnn,
        )
        scenarios.append(s)
        print(f"    smd_cascade: {s['anomaly_ticks']}/{s['n_ticks']} anomaly ticks")

    # Save
    for s in scenarios:
        path = out_dir / f"{s['name']}.pt"
        torch.save(s, path)
        gt = s["ground_truth"]
        aff = (gt != 0).sum().item()
        tot = gt.numel()
        size = path.stat().st_size // 1024
        print(f"\n  {s['name']:<25s} {s['n_nodes']:2d}n {s['n_ticks']:2d}t "
              f"{aff}/{tot} ({aff / tot * 100:.0f}%) {size}KB "
              f"[{s['anomaly_ticks']} anomaly ticks]")

    print(f"\n  Saved {len(scenarios)} SMD scenarios to {out_dir}/\n")


if __name__ == "__main__":
    main()
