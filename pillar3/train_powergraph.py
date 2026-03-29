#!/usr/bin/env python3
"""
Project PARALLAX — Pillar 3: Mamba Pre-training on PowerGraph
================================================================
Real power grid cascading failure dataset (NeurIPS 2024).
IEEE39: 39 buses, 46 lines, 28,000 cascade scenarios.
IEEE118: 118 buses, 186 lines, 122,500 cascade scenarios.

Teaches Mamba cascade dynamics from real physics simulations —
the same capability needed for infrastructure cascade prediction.

Data mapping:
  - Tick 0: Pre-cascade bus state + edge aggregation (normal operation)
  - Tick 1: Post-cascade bus state + edge aggregation (lines tripped)
  - Target: Per-node affected/state/health + graph-level severity

Usage:
    python train_powergraph.py                              # IEEE39 default
    python train_powergraph.py --grid ieee118 --epochs 100  # IEEE118
    python train_powergraph.py --grid all                   # All grids combined

Requires: ml-env with torch, h5py, numpy
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent))
from sable_mamba import SelectiveSSM, MambaBlock
from sable_mamba_final import SableMambaFinal, focal_loss

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


# ── Data Loading ──────────────────────────────────────────────────────────


def load_powergraph_grid(data_dir: str, grid_name: str = "ieee39") -> dict:
    """Load a single PowerGraph grid dataset from .mat files.

    Returns dict with raw data arrays.
    """
    raw_dir = Path(data_dir) / "dataset_cascades" / grid_name / grid_name / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"PowerGraph data not found at {raw_dir}")

    print(f"  {C_INFO}Loading {grid_name} from {raw_dir}...{C_RESET}")

    # Topology (fixed per grid)
    with h5py.File(raw_dir / "blist.mat", "r") as f:
        blist = np.array(f["bList"]).astype(np.int64) - 1  # 0-indexed

    num_nodes = int(blist.max()) + 1
    num_edges = blist.shape[1]

    # Build adjacency: node → list of edge indices
    node_edges = {n: [] for n in range(num_nodes)}
    for e in range(num_edges):
        u, v = blist[0, e], blist[1, e]
        node_edges[u].append(e)
        node_edges[v].append(e)

    # Load scenario data via h5py (MATLAB v7.3 HDF5 format)
    with h5py.File(raw_dir / "Bf.mat", "r") as f:
        bf_refs = f["B_f_tot"][0]
        num_scenarios = len(bf_refs)
        # Load all node features
        node_features = []
        for i in range(num_scenarios):
            nf = np.array(f[bf_refs[i]], dtype=np.float32)  # (3, num_nodes)
            node_features.append(nf)

    with h5py.File(raw_dir / "Ef_nc.mat", "r") as f:
        ef_nc_refs = f["E_f_kenza"][0]
        edge_features_pre = []
        for i in range(num_scenarios):
            ef = np.array(f[ef_nc_refs[i]], dtype=np.float32)  # (4, num_edges)
            edge_features_pre.append(ef)

    with h5py.File(raw_dir / "Ef.mat", "r") as f:
        ef_refs = f["E_f_post"][0]
        edge_features_post = []
        for i in range(num_scenarios):
            ef = np.array(f[ef_refs[i]], dtype=np.float32)  # (4, num_edges)
            edge_features_post.append(ef)

    # Labels
    with h5py.File(raw_dir / "of_bi.mat", "r") as f:
        bi_refs = f["output_features"][0]
        labels_binary = np.array([
            np.array(f[bi_refs[i]]).flatten()[0] for i in range(num_scenarios)
        ], dtype=np.float32)

    with h5py.File(raw_dir / "of_mc.mat", "r") as f:
        mc_refs = f["category"][0]
        labels_mc = np.array([
            np.argmax(np.array(f[mc_refs[i]]).flatten()) for i in range(num_scenarios)
        ], dtype=np.int64)

    with h5py.File(raw_dir / "of_reg.mat", "r") as f:
        labels_reg = np.array(f["dns_MW"]).flatten().astype(np.float32)

    print(f"    {C_TEXT}scenarios: {C_BRIGHT}{num_scenarios:,}{C_RESET}")
    print(f"    {C_TEXT}nodes:     {C_BRIGHT}{num_nodes}{C_RESET}")
    print(f"    {C_TEXT}edges:     {C_BRIGHT}{num_edges}{C_RESET}")

    # Class distribution
    for c in range(4):
        n = (labels_mc == c).sum()
        print(f"    {C_DIM}class {c}: {n:6d} ({n/num_scenarios*100:.1f}%){C_RESET}")

    return {
        "grid_name": grid_name,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_scenarios": num_scenarios,
        "blist": blist,
        "node_edges": node_edges,
        "node_features": node_features,
        "edge_features_pre": edge_features_pre,
        "edge_features_post": edge_features_post,
        "labels_binary": labels_binary,
        "labels_mc": labels_mc,
        "labels_reg": labels_reg,
    }


# ── Data Conversion ───────────────────────────────────────────────────────

# Per-node feature layout:
#   [0:3]  Bus features (active power, apparent power, voltage)
#   [3:7]  Mean adjacent edge features (4 features)
#   [7]    Edge health (fraction of adjacent edges intact)
NODE_FEAT_DIM = 8


def aggregate_edge_features(
    edge_features: np.ndarray,  # (4, num_edges)
    node_edges: dict,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate edge features per node and compute edge health.

    Returns:
        edge_agg: (num_nodes, 4) — mean adjacent edge features
        edge_health: (num_nodes,) — fraction of adjacent edges intact (non-zero)
    """
    edge_agg = np.zeros((num_nodes, 4), dtype=np.float32)
    edge_health = np.ones(num_nodes, dtype=np.float32)

    for n in range(num_nodes):
        adj_edges = node_edges[n]
        if not adj_edges:
            continue

        adj_feats = edge_features[:, adj_edges]  # (4, degree)

        # Which edges are intact (not all-zero)?
        intact = ~np.all(adj_feats == 0, axis=0)
        n_intact = intact.sum()
        n_total = len(adj_edges)

        edge_health[n] = n_intact / n_total

        if n_intact > 0:
            edge_agg[n] = adj_feats[:, intact].mean(axis=1)

    return edge_agg, edge_health


def convert_to_mamba_format(raw: dict) -> dict:
    """Convert PowerGraph raw data to SableMambaFinal training format.

    Returns dict matching temporal_v3.pt schema.
    """
    num_scenarios = raw["num_scenarios"]
    num_nodes = raw["num_nodes"]
    node_edges = raw["node_edges"]

    feat_dim = NODE_FEAT_DIM
    state_dim = num_nodes * feat_dim
    input_ticks = 2

    X_input = np.zeros((num_scenarios, input_ticks, state_dim), dtype=np.float32)
    Y_nodes = np.zeros((num_scenarios, num_nodes, 2), dtype=np.float32)  # health, state_class
    Y_severity = raw["labels_reg"].copy()
    N_nodes = np.full(num_scenarios, num_nodes, dtype=np.int64)

    # Normalize severity to [0, 1]
    sev_max = Y_severity.max()
    if sev_max > 0:
        Y_severity = Y_severity / sev_max

    t0 = time.time()
    for i in range(num_scenarios):
        bus_feats = raw["node_features"][i].T  # (num_nodes, 3)
        ef_pre = raw["edge_features_pre"][i]    # (4, num_edges)
        ef_post = raw["edge_features_post"][i]  # (4, num_edges)

        # Tick 0: pre-cascade (normal operation)
        edge_agg_pre, edge_health_pre = aggregate_edge_features(ef_pre, node_edges, num_nodes)
        tick0 = np.concatenate([bus_feats, edge_agg_pre, edge_health_pre[:, None]], axis=1)  # (N, 8)

        # Tick 1: post-cascade (lines may have tripped)
        edge_agg_post, edge_health_post = aggregate_edge_features(ef_post, node_edges, num_nodes)
        tick1 = np.concatenate([bus_feats, edge_agg_post, edge_health_post[:, None]], axis=1)  # (N, 8)

        X_input[i, 0] = tick0.flatten()
        X_input[i, 1] = tick1.flatten()

        # Per-node targets from edge health change
        for n in range(num_nodes):
            health = edge_health_post[n]
            if health >= 1.0:
                state_class = 0  # healthy
            elif health > 0.5:
                state_class = 1  # degraded
            elif health > 0.0:
                state_class = 2  # failed
            else:
                state_class = 3  # unreachable (all edges gone)
            Y_nodes[i, n, 0] = health
            Y_nodes[i, n, 1] = state_class

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    {C_DIM}{i+1:6d}/{num_scenarios} ({rate:.0f}/s){C_RESET}")

    elapsed = time.time() - t0
    print(f"    {C_TEXT}Converted {num_scenarios:,} scenarios in {elapsed:.1f}s{C_RESET}")

    # Train/val/test split (80/10/10)
    perm = np.random.default_rng(42).permutation(num_scenarios)
    n_train = int(num_scenarios * 0.8)
    n_val = int(num_scenarios * 0.1)

    dataset = {
        "X_input": torch.tensor(X_input),
        "Y_nodes": torch.tensor(Y_nodes),
        "Y_severity": torch.tensor(Y_severity),
        "N_nodes": torch.tensor(N_nodes),
        "train_idx": torch.tensor(perm[:n_train]),
        "val_idx": torch.tensor(perm[n_train:n_train + n_val]),
        "test_idx": torch.tensor(perm[n_train + n_val:]),
        "max_nodes": num_nodes,
        "input_ticks": input_ticks,
        "feat_dim": feat_dim,
        "state_dim": state_dim,
        "grid_name": raw["grid_name"],
        "labels_mc": torch.tensor(raw["labels_mc"]),
        "labels_binary": torch.tensor(raw["labels_binary"]),
    }

    # Stats
    Y_state = Y_nodes[:, :, 1].astype(np.int64)
    state_names = ["healthy", "degraded", "failed", "unreachable"]
    print(f"\n    {C_INFO}Per-node state distribution:{C_RESET}")
    for s in range(4):
        n = (Y_state == s).sum()
        total = Y_state.size
        print(f"      {C_DIM}{state_names[s]:12s}: {n:8d} ({n/total*100:.1f}%){C_RESET}")

    affected = (Y_state != 0).sum()
    print(f"    {C_DIM}affected total: {affected:8d} ({affected/Y_state.size*100:.1f}%){C_RESET}")
    print(f"    {C_TEXT}severity range: [{Y_severity.min():.4f}, {Y_severity.max():.4f}]{C_RESET}")

    return dataset


# ── Training ──────────────────────────────────────────────────────────────


@dataclass
class PGConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.0005
    weight_decay: float = 1e-3
    patience: int = 25
    d_model: int = 256
    n_layers: int = 4
    d_state: int = 16
    dropout: float = 0.2
    focal_gamma: float = 2.0
    affected_weight: float = 1.0
    state_weight: float = 1.0
    health_weight: float = 0.5
    severity_weight: float = 0.3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train(dataset: dict, config: PGConfig | None = None):
    if config is None:
        config = PGConfig()

    grid_name = dataset.get("grid_name", "unknown")
    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Pillar 3: PowerGraph {grid_name.upper()} Training{C_RESET}")
    print(f"  {C_DIM}{'═' * 60}{C_RESET}\n")

    X_input = dataset["X_input"]
    Y_nodes = dataset["Y_nodes"]
    Y_severity = dataset["Y_severity"]
    N_nodes = dataset["N_nodes"]
    train_idx = dataset["train_idx"]
    val_idx = dataset["val_idx"]
    test_idx = dataset["test_idx"]
    max_nodes = dataset["max_nodes"]
    state_dim = dataset["state_dim"]
    feat_dim = dataset["feat_dim"]
    input_ticks = dataset["input_ticks"]

    Y_health = Y_nodes[:, :, 0]
    Y_state = Y_nodes[:, :, 1].long()
    Y_affected = (Y_state != 0).float()

    print(f"  {C_TEXT}scenarios:   {C_BRIGHT}{X_input.size(0):,}{C_RESET}")
    print(f"  {C_TEXT}nodes/grid:  {C_BRIGHT}{max_nodes}{C_RESET}")
    print(f"  {C_TEXT}state dim:   {C_BRIGHT}{state_dim}{C_RESET}")
    print(f"  {C_TEXT}feat dim:    {C_BRIGHT}{feat_dim}{C_RESET}")
    print(f"  {C_TEXT}device:      {C_BRIGHT}{config.device}{C_RESET}")

    # Class balance
    n_affected = Y_affected[train_idx].sum().item()
    n_healthy = (Y_affected[train_idx] == 0).sum().item()
    affected_pos_weight = torch.tensor(
        [n_healthy / max(n_affected, 1)], device=config.device
    ).clamp(max=20.0)

    state_counts = torch.bincount(Y_state[train_idx].flatten(), minlength=4).float().clamp(min=1)
    state_weights = (Y_state[train_idx].numel() / (4 * state_counts)).to(config.device)
    state_weights = state_weights.clamp(max=15.0)

    print(f"\n  {C_INFO}Class balance:{C_RESET}")
    print(f"    {C_DIM}healthy nodes:  {int(n_healthy):,} ({n_healthy/(n_healthy+n_affected)*100:.1f}%){C_RESET}")
    print(f"    {C_DIM}affected nodes: {int(n_affected):,} ({n_affected/(n_healthy+n_affected)*100:.1f}%){C_RESET}")
    print(f"    {C_DIM}affected pos_weight: {affected_pos_weight.item():.1f}x{C_RESET}")
    state_names = ["healthy", "degraded", "failed", "unreachable"]
    for i, name in enumerate(state_names):
        print(f"    {C_DIM}{name}: weight={state_weights[i]:.2f}{C_RESET}")

    # Weighted sampler: oversample cascade scenarios
    sample_weights = torch.ones(len(train_idx))
    for i, idx in enumerate(train_idx):
        sev = Y_severity[idx].item()
        sample_weights[i] = 1.0 + 9.0 * sev  # Up to 10x for severe cascades

    sampler = WeightedRandomSampler(sample_weights, len(train_idx), replacement=True)

    # Model
    model = SableMambaFinal(
        state_dim=state_dim,
        max_nodes=max_nodes,
        node_feat_dim=feat_dim,
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_state=config.d_state,
        dropout=config.dropout,
        input_ticks=input_ticks,
    ).to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {C_TEXT}params:      {C_BRIGHT}{n_params:,}{C_RESET}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    # DataLoaders
    train_ds = TensorDataset(
        X_input[train_idx], Y_health[train_idx], Y_state[train_idx],
        Y_affected[train_idx], Y_severity[train_idx], N_nodes[train_idx],
    )
    val_ds = TensorDataset(
        X_input[val_idx], Y_health[val_idx], Y_state[val_idx],
        Y_affected[val_idx], Y_severity[val_idx], N_nodes[val_idx],
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    checkpoint_path = Path(__file__).parent / f"checkpoints_powergraph_{grid_name}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_val_metric = -1.0
    patience_counter = 0

    print(f"\n  {C_DIM}{'ep':>4s} {'loss':>7s} {'aff_f1':>7s} {'v_aff':>7s} {'v_st':>7s} {'v_fail':>7s} {'v_sev':>7s}{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")

    t_start = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_aff_tp = epoch_aff_fp = epoch_aff_fn = 0
        epoch_n = 0

        for bx, bh, bs, ba, bsev, bn in train_loader:
            bx = bx.to(config.device)
            bh = bh.to(config.device)
            bs = bs.to(config.device)
            ba = ba.to(config.device)
            bsev = bsev.to(config.device)

            out = model(bx)

            # Node mask (all nodes valid for fixed-topology grid)
            mask = torch.ones(bx.size(0), max_nodes, device=config.device)

            # Loss 1: Affected detection
            aff_bce = F.binary_cross_entropy_with_logits(
                out["affected_logits"], ba,
                pos_weight=affected_pos_weight.expand(max_nodes),
                reduction="none"
            )
            aff_probs = torch.sigmoid(out["affected_logits"])
            pt = ba * aff_probs + (1 - ba) * (1 - aff_probs)
            focal_w = (1 - pt) ** config.focal_gamma
            aff_loss = (aff_bce * focal_w * mask).sum() / mask.sum()

            # Loss 2: State classification
            logits_flat = out["state_logits"].reshape(-1, 4)
            targets_flat = bs.reshape(-1)
            mask_flat = mask.reshape(-1)
            fl = focal_loss(logits_flat, targets_flat, gamma=config.focal_gamma, weights=state_weights)
            state_loss = (fl * mask_flat).sum() / mask_flat.sum()

            # Loss 3: Health regression
            health_loss = ((out["health"] - bh) ** 2 * mask).sum() / mask.sum()

            # Loss 4: Severity
            severity_loss = F.mse_loss(out["severity"], bsev)

            loss = (
                config.affected_weight * aff_loss
                + config.state_weight * state_loss
                + config.health_weight * health_loss
                + config.severity_weight * severity_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            aff_preds = (out["affected_logits"] > 0).float()
            epoch_aff_tp += ((aff_preds == 1) & (ba == 1) & (mask == 1)).sum().item()
            epoch_aff_fp += ((aff_preds == 1) & (ba == 0) & (mask == 1)).sum().item()
            epoch_aff_fn += ((aff_preds == 0) & (ba == 1) & (mask == 1)).sum().item()
            epoch_loss += loss.item() * mask.sum().item()
            epoch_n += mask.sum().item()

        train_loss = epoch_loss / max(epoch_n, 1)
        aff_prec = epoch_aff_tp / max(epoch_aff_tp + epoch_aff_fp, 1)
        aff_rec = epoch_aff_tp / max(epoch_aff_tp + epoch_aff_fn, 1)
        train_aff_f1 = 2 * aff_prec * aff_rec / max(aff_prec + aff_rec, 1e-8)

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_aff_tp = val_aff_fp = val_aff_fn = 0
        val_state_correct = val_state_total = 0
        val_fail_total = val_fail_correct = 0
        val_sev_errors = []

        with torch.no_grad():
            for bx, bh, bs, ba, bsev, bn in val_loader:
                bx = bx.to(config.device)
                bs = bs.to(config.device)
                ba = ba.to(config.device)
                bsev = bsev.to(config.device)

                out = model(bx)
                mask = torch.ones(bx.size(0), max_nodes, device=config.device)

                aff_preds = (out["affected_logits"] > 0).float()
                val_aff_tp += ((aff_preds == 1) & (ba == 1) & (mask == 1)).sum().item()
                val_aff_fp += ((aff_preds == 1) & (ba == 0) & (mask == 1)).sum().item()
                val_aff_fn += ((aff_preds == 0) & (ba == 1) & (mask == 1)).sum().item()

                state_preds = out["state_logits"].argmax(dim=-1)
                val_state_correct += ((state_preds == bs) * mask).sum().item()
                val_state_total += mask.sum().item()

                fail_mask = (bs == 2) & (mask > 0)
                val_fail_total += fail_mask.sum().item()
                val_fail_correct += ((state_preds == 2) & fail_mask).sum().item()

                val_sev_errors.append(
                    F.mse_loss(out["severity"], bsev).item()
                )

        val_aff_prec = val_aff_tp / max(val_aff_tp + val_aff_fp, 1)
        val_aff_rec = val_aff_tp / max(val_aff_tp + val_aff_fn, 1)
        val_aff_f1 = 2 * val_aff_prec * val_aff_rec / max(val_aff_prec + val_aff_rec, 1e-8)
        val_state_acc = val_state_correct / max(val_state_total, 1)
        val_fail_rec = val_fail_correct / max(val_fail_total, 1)
        val_sev_mse = np.mean(val_sev_errors) if val_sev_errors else 0.0

        improved = val_aff_f1 > best_val_metric + 0.001
        marker = f"{C_SUCCESS}*{C_RESET}" if improved else " "

        if epoch <= 5 or epoch % 5 == 0 or improved or epoch == config.epochs:
            print(
                f"  {C_TEXT}{epoch:4d}{C_RESET} "
                f"{train_loss:7.4f} "
                f"{train_aff_f1:7.4f} "
                f"{val_aff_f1:7.4f} "
                f"{val_state_acc:7.4f} "
                f"{val_fail_rec:7.4f} "
                f"{val_sev_mse:7.5f} {marker}"
            )

        if improved:
            best_val_metric = val_aff_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_aff_f1": val_aff_f1,
                "val_state_acc": val_state_acc,
                "val_fail_recall": val_fail_rec,
                "val_sev_mse": val_sev_mse,
                "config": {
                    "state_dim": state_dim, "max_nodes": max_nodes,
                    "d_model": config.d_model, "n_layers": config.n_layers,
                    "d_state": config.d_state, "dropout": config.dropout,
                    "node_feat_dim": feat_dim, "input_ticks": input_ticks,
                },
                "grid_name": grid_name,
                "n_params": n_params,
            }, checkpoint_path / "best_powergraph.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\n  {C_DIM}Early stopping at epoch {epoch}{C_RESET}")
                break

    total_time = time.time() - t_start

    # ── Test ──
    print(f"\n  {C_GOLD}{C_BOLD}  Test Set Evaluation{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")

    ckpt = torch.load(checkpoint_path / "best_powergraph.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = TensorDataset(
        X_input[test_idx], Y_health[test_idx], Y_state[test_idx],
        Y_affected[test_idx], Y_severity[test_idx], N_nodes[test_idx],
    )
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    class_tp = torch.zeros(4)
    class_fp = torch.zeros(4)
    class_fn = torch.zeros(4)
    aff_tp = aff_fp = aff_fn = aff_tn = 0
    health_errors = []
    sev_errors = []

    # Also track graph-level accuracy (cascade vs no-cascade)
    graph_correct = 0
    graph_total = 0

    with torch.no_grad():
        for bx, bh, bs, ba, bsev, bn in test_loader:
            bx = bx.to(config.device)
            out = model(bx)
            mask = torch.ones(bx.size(0), max_nodes)

            aff_preds = (out["affected_logits"].cpu() > 0).float()
            aff_tp += ((aff_preds == 1) & (ba == 1) & (mask == 1)).sum().item()
            aff_fp += ((aff_preds == 1) & (ba == 0) & (mask == 1)).sum().item()
            aff_fn += ((aff_preds == 0) & (ba == 1) & (mask == 1)).sum().item()
            aff_tn += ((aff_preds == 0) & (ba == 0) & (mask == 1)).sum().item()

            state_preds = out["state_logits"].cpu().argmax(dim=-1)
            for c in range(4):
                c_true = (bs == c) & (mask > 0)
                c_pred = (state_preds == c) & (mask > 0)
                class_tp[c] += (c_true & c_pred).sum().item()
                class_fp[c] += (~c_true & c_pred).sum().item()
                class_fn[c] += (c_true & ~c_pred).sum().item()

            h_err = ((out["health"].cpu() - bh) ** 2 * mask).sum() / mask.sum()
            health_errors.append(h_err.item())

            sev_err = ((out["severity"].cpu() - bsev) ** 2).mean()
            sev_errors.append(sev_err.item())

            # Graph-level: any node affected → cascade predicted
            graph_pred_cascade = (aff_preds.sum(dim=1) > 0).float()
            graph_true_cascade = (ba.sum(dim=1) > 0).float()
            graph_correct += (graph_pred_cascade == graph_true_cascade).sum().item()
            graph_total += bx.size(0)

    # Report
    print(f"\n  {C_INFO}Affected Node Detection (binary):{C_RESET}")
    aff_prec = aff_tp / max(aff_tp + aff_fp, 1)
    aff_rec = aff_tp / max(aff_tp + aff_fn, 1)
    aff_f1 = 2 * aff_prec * aff_rec / max(aff_prec + aff_rec, 1e-8)
    print(f"    {C_TEXT}Precision:  {C_BRIGHT}{aff_prec:.4f}{C_RESET}")
    print(f"    {C_TEXT}Recall:     {C_BRIGHT}{aff_rec:.4f}{C_RESET}")
    print(f"    {C_TEXT}F1:         {C_BRIGHT}{aff_f1:.4f}{C_RESET}")
    print(f"    {C_DIM}TP={int(aff_tp)} FP={int(aff_fp)} FN={int(aff_fn)} TN={int(aff_tn)}{C_RESET}")

    print(f"\n  {C_INFO}Per-State Classification:{C_RESET}")
    for c in range(4):
        prec = class_tp[c] / max(class_tp[c] + class_fp[c], 1)
        rec = class_tp[c] / max(class_tp[c] + class_fn[c], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        support = int(class_tp[c] + class_fn[c])
        bar = "█" * int(f1 * 25)
        print(f"    {C_DIM}{state_names[c]:15s}{C_RESET} "
              f"P={C_TEXT}{prec:.3f}{C_RESET} "
              f"R={C_TEXT}{rec:.3f}{C_RESET} "
              f"F1={C_TEXT}{f1:.3f}{C_RESET} "
              f"(n={support:6d}) {C_GOLD}{bar}{C_RESET}")

    macro_f1 = sum(
        2 * (class_tp[c] / max(class_tp[c] + class_fp[c], 1)) *
        (class_tp[c] / max(class_tp[c] + class_fn[c], 1)) /
        max((class_tp[c] / max(class_tp[c] + class_fp[c], 1)) +
            (class_tp[c] / max(class_tp[c] + class_fn[c], 1)), 1e-8)
        for c in range(4)
    ) / 4

    print(f"\n  {C_TEXT}Macro F1:      {C_BRIGHT}{macro_f1:.4f}{C_RESET}")
    print(f"  {C_TEXT}Health MSE:    {C_BRIGHT}{np.mean(health_errors):.6f}{C_RESET}")
    print(f"  {C_TEXT}Severity MSE:  {C_BRIGHT}{np.mean(sev_errors):.6f}{C_RESET}")
    print(f"  {C_TEXT}Graph-level:   {C_BRIGHT}{graph_correct/graph_total:.4f}{C_RESET} accuracy")

    # Baselines
    print(f"\n  {C_INFO}Baselines:{C_RESET}")
    baseline_healthy_pct = (class_tp[0] + class_fn[0]) / sum(class_tp[c] + class_fn[c] for c in range(4))
    print(f"    {C_DIM}All-healthy accuracy: {baseline_healthy_pct:.4f} (catches 0% of failures){C_RESET}")

    if aff_f1 > 0.3:
        print(f"\n  {C_SUCCESS}{C_BOLD}CASCADE REASONING VALIDATED ON REAL PHYSICS{C_RESET}")
        print(f"  {C_TEXT}Mamba learns real cascade dynamics from PowerGraph.{C_RESET}")
    else:
        print(f"\n  {C_DANGER}Affected F1 below threshold — needs tuning{C_RESET}")

    # Summary
    print(f"\n  {C_GOLD}{C_BOLD}  Summary{C_RESET}")
    print(f"  {C_DIM}{'─' * 45}{C_RESET}")
    print(f"  {C_TEXT}Grid:         {C_BRIGHT}{grid_name}{C_RESET}")
    print(f"  {C_TEXT}Best epoch:   {C_BRIGHT}{ckpt['epoch']}{C_RESET}")
    print(f"  {C_TEXT}Total time:   {C_BRIGHT}{total_time:.0f}s ({total_time/60:.1f}m){C_RESET}")
    print(f"  {C_TEXT}Parameters:   {C_BRIGHT}{n_params:,}{C_RESET}")
    print(f"  {C_TEXT}Checkpoint:   {C_BRIGHT}{checkpoint_path / 'best_powergraph.pt'}{C_RESET}")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train Mamba on PowerGraph cascade dataset"
    )
    parser.add_argument("--grid", type=str, default="ieee39",
                        choices=["ieee24", "ieee39", "ieee118", "uk", "all"])
    parser.add_argument("--data-dir", type=str,
                        default=str(Path(__file__).parent / "data" / "PowerGraph"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-converted", type=str, default=None,
                        help="Save converted dataset to .pt file for reuse")
    parser.add_argument("--load-converted", type=str, default=None,
                        help="Load pre-converted .pt dataset instead of raw .mat")
    args = parser.parse_args()

    config = PGConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        patience=args.patience,
        device=args.device,
    )

    print(f"\n{C_GOLD}{C_BOLD}  PROJECT PARALLAX — Pillar 3: PowerGraph Data Pipeline{C_RESET}")
    print(f"  {C_DIM}{'═' * 55}{C_RESET}\n")

    if args.load_converted:
        print(f"  {C_INFO}Loading pre-converted dataset: {args.load_converted}{C_RESET}")
        dataset = torch.load(args.load_converted, weights_only=False)
    else:
        grids = ["ieee24", "ieee39", "ieee118", "uk"] if args.grid == "all" else [args.grid]

        for grid in grids:
            raw = load_powergraph_grid(args.data_dir, grid)
            dataset = convert_to_mamba_format(raw)

            if args.save_converted:
                save_path = args.save_converted.replace(".pt", f"_{grid}.pt")
                torch.save(dataset, save_path)
                size_mb = Path(save_path).stat().st_size / (1024 * 1024)
                print(f"  {C_SUCCESS}Saved: {save_path} ({size_mb:.1f} MB){C_RESET}")

            train(dataset, config)


if __name__ == "__main__":
    main()
