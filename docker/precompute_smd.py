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

    # --- POMDP: belief vectors from RAW METRICS (not health-scored state) ---
    # The model should predict state from observations, not be told the state.
    # POMDP beliefs represent what monitoring sees: noisy metric signals.
    pomdp_out = torch.zeros(n, POMDP_DIM)
    for i, nid in enumerate(node_ids):
        node = snapshot.nodes[nid]

        # Raw metric averages as observation signals (NOT the classified state)
        cpu = node.metrics.get("cpu_utilization", 0) / 100.0
        mem = node.metrics.get("mem_used_pct", 0) / 100.0
        disk = node.metrics.get("disk_io_util", 0) / 100.0
        net = node.metrics.get("net_bandwidth_util", 0) / 100.0

        # Soft state estimate from raw metrics (independent of health scorer)
        # High utilization -> probably degraded/failed, low -> probably healthy
        avg_util = (cpu * 1.5 + mem * 1.3 + disk * 0.8 + net * 0.7) / 4.3
        p_healthy = max(0.0, 1.0 - avg_util * 2.0)
        p_degraded = min(1.0, avg_util * 1.5) * (1.0 - p_healthy)
        p_failed = max(0.0, avg_util - 0.5) * 0.5
        p_unreachable = 0.02  # Small baseline
        total = p_healthy + p_degraded + p_failed + p_unreachable
        if total > 0:
            p_healthy /= total
            p_degraded /= total
            p_failed /= total
            p_unreachable /= total

        # Confidence from metric consistency (jittery metrics = lower confidence)
        confidence = max(0.3, 1.0 - avg_util * 0.5)
        obs_age = 0.1
        hub = degrees.get(nid, 0) / max(max_deg, 1)

        pomdp_out[i] = torch.tensor([p_healthy, p_degraded, p_failed, p_unreachable,
                                      confidence, obs_age, 0.0, hub])

    # --- Mamba: 26-dim per node (health + state_onehot + type_onehot) ---
    # Health comes from raw metric composite, NOT the health scorer.
    # State one-hot uses a NOISY estimate, not the ground truth classification.
    mamba_out = torch.zeros(n, NODE_FEAT_DIM)
    for i, nid in enumerate(node_ids):
        node = snapshot.nodes[nid]

        # Raw health from metrics (different formula than health scorer)
        cpu = node.metrics.get("cpu_utilization", 0) / 100.0
        mem = node.metrics.get("mem_used_pct", 0) / 100.0
        disk = node.metrics.get("disk_io_util", 0) / 100.0
        raw_health = max(0.0, min(1.0, 1.0 - (cpu + mem + disk) / 3.0 * 1.8))
        mamba_out[i, 0] = raw_health

        # Noisy state estimate from raw health (NOT the health scorer output)
        # This intentionally disagrees with ground truth sometimes
        if raw_health > 0.65:
            noisy_state = 0  # healthy
        elif raw_health > 0.25:
            noisy_state = 1  # degraded
        else:
            noisy_state = 2  # failed
        mamba_out[i, 1 + noisy_state] = 1.0

        # Type one-hot (20 types)
        ti = COMPONENT_TYPE_INDEX.get(node.component_type, 0)
        mamba_out[i, 6 + ti] = 1.0

    # --- Ground truth: from the HEALTH SCORER classification ---
    # This is what the model should predict - the definitive state assessment.
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
    adapter.load(split="test", max_ticks=5000)  # Full test set
    print(f"  Loaded {len(adapter.machines)} machines, {adapter.n_ticks} ticks")
    print(f"  Topology: {len(adapter._edges)} edges")

    gnn = load_gnn()
    out_dir = Path(__file__).parent / "scenarios"
    out_dir.mkdir(exist_ok=True)

    # Find ALL anomaly windows
    print(f"  Scanning for anomaly windows...", flush=True)
    windows = find_anomaly_windows(adapter, min_length=10)
    print(f"  Found {len(windows)} anomaly windows")

    scenarios = []

    # Normal operation windows (spread across the dataset)
    normal_starts = [0, 50, 500, 1000, 1500, 2000, 3000, 4000]
    for i, start in enumerate(normal_starts):
        if start + 25 > adapter.n_ticks:
            break
        name = f"smd_normal_{i+1}"
        s = generate_smd_scenario(
            name, f"Real server telemetry - normal operation window {i+1} (tick {start})",
            adapter, start, 25, gnn,
        )
        scenarios.append(s)
        print(f"    {name}: {s['anomaly_ticks']}/{s['n_ticks']} anomaly ticks")

    # ALL anomaly windows
    for i, (start, length) in enumerate(windows):
        name = f"smd_incident_{i+1}"
        n_ticks = min(length, 30)
        if start + n_ticks > adapter.n_ticks:
            continue
        s = generate_smd_scenario(
            name, f"Real server incident {i+1} - anomaly at tick {start} (28 machines)",
            adapter, start, n_ticks, gnn,
        )
        scenarios.append(s)
        print(f"    {name}: {s['anomaly_ticks']}/{s['n_ticks']} anomaly ticks (start={start})")

    # Extended cascades spanning multiple windows
    for i in range(0, len(windows) - 1, 3):
        start = windows[i][0]
        end_window = min(i + 2, len(windows) - 1)
        end = min(windows[end_window][0] + windows[end_window][1], adapter.n_ticks)
        length = min(end - start, 30)
        if length < 15:
            continue
        name = f"smd_cascade_{i//3 + 1}"
        s = generate_smd_scenario(
            name, f"Real multi-incident cascade {i//3 + 1} (ticks {start}-{start+length})",
            adapter, start, length, gnn,
        )
        scenarios.append(s)
        print(f"    {name}: {s['anomaly_ticks']}/{s['n_ticks']} anomaly ticks")

    # Save
    print(f"\n  Saving {len(scenarios)} scenarios...")
    for s in scenarios:
        path = out_dir / f"{s['name']}.pt"
        torch.save(s, path)
        gt = s["ground_truth"]
        aff = (gt != 0).sum().item()
        tot = gt.numel()
        size = path.stat().st_size // 1024
        print(f"  {s['name']:<25s} {s['n_nodes']:2d}n {s['n_ticks']:2d}t "
              f"{aff}/{tot} ({aff / tot * 100:.0f}%) {size}KB "
              f"[{s['anomaly_ticks']} anomaly ticks]")

    print(f"\n  Saved {len(scenarios)} SMD scenarios to {out_dir}/\n")


if __name__ == "__main__":
    main()
