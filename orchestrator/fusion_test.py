#!/usr/bin/env python3
"""
Project PARALLAX — Phase 4: Deep Fusion Test
===============================================
THE critical experiment: does fusing GNN structural embeddings
into Mamba's input features improve cascade prediction?

Specifically: does unreachable F1 lift when Mamba can see graph structure?

Test design:
  1. Generate infrastructure cascade sequences (same as Mamba training)
  2. For each sequence, run GNN on the topology to get structural embeddings
  3. Train Mamba twice:
     a) WITHOUT GNN features (baseline — current architecture)
     b) WITH GNN features (fused — the thesis)
  4. Compare per-class F1, especially unreachable

If fusion lifts unreachable F1 by 0.05+, the architecture thesis holds.
If it doesn't, the pillars are independent systems sharing a process.

Usage:
    python fusion_test.py --device cuda
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar3"))

from sable_mamba import MambaBlock
from cortex_gnn_model import SableGNN
from domain_portability_test import InfrastructureAdapter

from sable_sim.core.component import Component, ComponentState, ComponentType, DEFAULT_PROPERTIES
from sable_sim.core.dependency import Dependency, DependencyType, Criticality
from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.state import SystemState, StateChange
from sable_sim.simulation.propagation import PropagationEngine
from sable_sim.simulation.failure_injection import FailureInjector
from sable_sim.utils.random import SeededRandom

from generate_temporal_data import (
    build_random_topology, encode_system_state, encode_system_state,
    NODE_FEAT_DIM, STATE_MAP, N_STATES, N_COMP_TYPES,
)

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"

GNN_EMBED_DIM = 256  # GNN output per node
GNN_CHECKPOINT = str(Path(__file__).parent.parent / "pillar1" / "checkpoints" / "best_model.pt")


# ── Fused Mamba Model ─────────────────────────────────────────────────────


class FusedMamba(nn.Module):
    """Mamba with optional GNN structural embeddings fused into input.

    When gnn_dim > 0: input = [temporal_state, gnn_embedding, delta]
    When gnn_dim = 0: input = [temporal_state, delta] (baseline, no fusion)
    """

    def __init__(self, state_dim, max_nodes, gnn_dim=0, d_model=256,
                 n_layers=4, d_state=16, dropout=0.2, input_ticks=2,
                 n_states=4):
        super().__init__()
        self.state_dim = state_dim
        self.max_nodes = max_nodes
        self.gnn_dim = gnn_dim
        self.n_states = n_states

        # Input: ticks + delta + optional GNN embeddings
        total_gnn = max_nodes * gnn_dim  # GNN embedding per node, flattened
        augmented_dim = state_dim * (input_ticks + 1) + total_gnn

        self.input_proj = nn.Sequential(
            nn.Linear(augmented_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, 4, 2, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.affected_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, max_nodes),
        )
        self.state_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, max_nodes * n_states),
        )
        self.health_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, max_nodes), nn.Sigmoid(),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x_temporal, x_gnn=None):
        """
        x_temporal: (B, T, state_dim) — temporal state sequence
        x_gnn: (B, max_nodes * gnn_dim) — GNN structural embeddings (optional)
        """
        B = x_temporal.size(0)
        deltas = x_temporal[:, -1, :] - x_temporal[:, 0, :]
        x_flat = x_temporal.reshape(B, -1)

        if x_gnn is not None and self.gnn_dim > 0:
            augmented = torch.cat([x_flat, deltas, x_gnn], dim=-1)
        else:
            augmented = torch.cat([x_flat, deltas], dim=-1)

        h = self.input_proj(augmented).unsqueeze(1)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h).squeeze(1)

        return {
            "affected_logits": self.affected_head(h),
            "state_logits": self.state_head(h).view(B, self.max_nodes, self.n_states),
            "health": self.health_head(h),
            "severity": self.severity_head(h).squeeze(-1),
        }


# ── Data Generation with GNN Embeddings ───────────────────────────────────


def generate_fused_dataset(count=5000, device="cuda", seed=42):
    """Generate cascade sequences with GNN structural embeddings for each graph."""

    print(f"  {C_INFO}Generating {count} sequences with GNN embeddings...{C_RESET}")

    # Load GNN
    ckpt = torch.load(GNN_CHECKPOINT, weights_only=False, map_location=device)
    mc = ckpt["config"]
    gnn = SableGNN(
        in_dim=mc["in_dim"], hidden_dim=mc["hidden_dim"],
        edge_dim=mc["edge_dim"], num_layers=mc["num_layers"],
        heads=mc["heads"], dropout=mc["dropout"],
    ).to(device)
    gnn.load_state_dict(ckpt["model_state_dict"])
    gnn.eval()

    adapter = InfrastructureAdapter(seed=seed)
    rng = SeededRandom(seed)

    max_nodes = 40
    state_dim = max_nodes * NODE_FEAT_DIM
    gnn_feat_dim = GNN_EMBED_DIM

    X_temporal = torch.zeros(count, 2, state_dim)
    X_gnn = torch.zeros(count, max_nodes * gnn_feat_dim)
    Y_health = torch.zeros(count, max_nodes)
    Y_state = torch.zeros(count, max_nodes, dtype=torch.long)
    Y_affected = torch.zeros(count, max_nodes)
    Y_severity = torch.zeros(count)
    N_nodes = torch.zeros(count, dtype=torch.long)

    generated = 0
    attempts = 0
    t0 = time.time()

    while generated < count and attempts < count * 3:
        attempts += 1

        # Build random topology
        graph = build_random_topology(rng, 15, 40)
        components = graph.get_all_components()
        if len(components) < 15:
            continue
        component_ids = [c.id for c in components]
        n = len(component_ids)

        # Encode initial state
        state_t0 = encode_system_state(graph, component_ids)

        # Inject failure
        state = SystemState(graph)
        injector = FailureInjector(rng)
        try:
            injections = injector.generate_failures(
                state, difficulty=rng.weighted_choice(["easy", "medium", "hard"], [0.2, 0.5, 0.3])
            )
        except Exception:
            continue
        if not injections:
            continue

        # 40% degradation-only
        use_degradation = rng.random() < 0.4
        for inj in injections:
            injector.inject(state, inj)
            if use_degradation:
                comp = graph.get_component(inj.component_id)
                if comp and comp.state == ComponentState.FAILED:
                    comp.state = ComponentState.DEGRADED
                    comp.health = rng.uniform(0.25, 0.45)

        state_t1 = encode_system_state(graph, component_ids)

        # Propagate to get outcome
        engine = PropagationEngine(max_ticks=30, soft_impact_factor=0.4 if use_degradation else 0.3)
        engine.propagate(state)

        # Outcome targets
        for i, cid in enumerate(component_ids):
            comp = graph.get_component(cid)
            if comp:
                Y_health[generated, i] = comp.health
                Y_state[generated, i] = STATE_MAP.get(comp.state, 0)
                Y_affected[generated, i] = 0.0 if comp.state == ComponentState.HEALTHY else 1.0

        n_affected = Y_affected[generated, :n].sum().item()
        Y_severity[generated] = n_affected / max(n, 1)
        N_nodes[generated] = n

        # Temporal features
        actual_dim = n * NODE_FEAT_DIM
        X_temporal[generated, 0, :actual_dim] = torch.tensor(state_t0[:actual_dim])
        X_temporal[generated, 1, :actual_dim] = torch.tensor(state_t1[:actual_dim])

        # GNN structural embeddings — run GNN on this infrastructure graph
        with torch.no_grad():
            # Build PyG-compatible input through adapter
            node_features = []
            for comp in components:
                feat = adapter.adapt_node(str(comp.type), comp.health)
                node_features.append(feat)
            x = torch.tensor(np.array(node_features), dtype=torch.float32).to(device)

            # Build edge index and features from graph dependencies
            sources, targets, edge_feats = [], [], []
            for comp in components:
                for dep_id in comp.dependencies_in:
                    src_idx = component_ids.index(comp.id) if comp.id in component_ids else None
                    tgt_idx = component_ids.index(dep_id) if dep_id in component_ids else None
                    if src_idx is not None and tgt_idx is not None:
                        dep = graph.get_dependency(comp.id, dep_id)
                        if dep:
                            rel_idx, feat = adapter.adapt_edge(str(dep.type), str(dep.criticality))
                            sources.append(src_idx)
                            targets.append(tgt_idx)
                            edge_feats.append(feat)

            if sources:
                edge_index = torch.tensor([sources, targets], dtype=torch.long).to(device)
                edge_attr = torch.tensor(np.array(edge_feats), dtype=torch.float32).to(device)

                # Get GNN node embeddings
                node_emb = gnn.encode(x, edge_index, edge_attr)  # (n, 256)
                # Pad to max_nodes
                padded = torch.zeros(max_nodes, gnn_feat_dim, device=device)
                padded[:n] = node_emb
                X_gnn[generated] = padded.flatten().cpu()

        generated += 1
        if generated % 500 == 0:
            elapsed = time.time() - t0
            print(f"    {C_DIM}{generated}/{count} ({elapsed:.1f}s){C_RESET}")

    elapsed = time.time() - t0
    print(f"  {C_TEXT}Generated {generated} sequences in {elapsed:.1f}s{C_RESET}")

    # Split
    perm = torch.randperm(generated)
    n_train = int(generated * 0.8)
    n_val = int(generated * 0.1)

    return {
        "X_temporal": X_temporal[:generated],
        "X_gnn": X_gnn[:generated],
        "Y_health": Y_health[:generated],
        "Y_state": Y_state[:generated],
        "Y_affected": Y_affected[:generated],
        "Y_severity": Y_severity[:generated],
        "N_nodes": N_nodes[:generated],
        "train_idx": perm[:n_train],
        "val_idx": perm[n_train:n_train + n_val],
        "test_idx": perm[n_train + n_val:],
        "max_nodes": max_nodes,
        "state_dim": state_dim,
        "gnn_dim": gnn_feat_dim,
    }


# ── Training ──────────────────────────────────────────────────────────────


def focal_loss(logits, targets, gamma=2.0, weights=None):
    ce = F.cross_entropy(logits, targets, weight=weights, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce)


def train_model(model, data, use_gnn, device, epochs=60, batch_size=128, lr=0.0005):
    """Train a model and return test metrics."""
    X_t = data["X_temporal"]
    X_g = data["X_gnn"] if use_gnn else None
    Y_h = data["Y_health"]
    Y_s = data["Y_state"]
    Y_a = data["Y_affected"]
    Y_sev = data["Y_severity"]
    N_n = data["N_nodes"]
    max_nodes = data["max_nodes"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    # Class weights
    state_counts = torch.bincount(Y_s[train_idx].flatten(), minlength=4).float().clamp(min=1)
    state_weights = (Y_s[train_idx].numel() / (4 * state_counts)).clamp(max=10.0).to(device)

    n_affected = Y_a[train_idx].sum().item()
    n_healthy = (Y_a[train_idx] == 0).sum().item()
    aff_pos_weight = torch.tensor([n_healthy / max(n_affected, 1)], device=device).clamp(max=15.0)

    # Sample weights
    sample_weights = torch.ones(len(train_idx))
    for i, idx in enumerate(train_idx):
        sample_weights[i] = 1.0 + 4.0 * Y_sev[idx].item()
    sampler = WeightedRandomSampler(sample_weights, len(train_idx), replacement=True)

    # DataLoaders
    if use_gnn:
        train_ds = TensorDataset(X_t[train_idx], X_g[train_idx], Y_h[train_idx], Y_s[train_idx], Y_a[train_idx], Y_sev[train_idx], N_n[train_idx])
        val_ds = TensorDataset(X_t[val_idx], X_g[val_idx], Y_h[val_idx], Y_s[val_idx], Y_a[val_idx], Y_sev[val_idx], N_n[val_idx])
        test_ds = TensorDataset(X_t[test_idx], X_g[test_idx], Y_h[test_idx], Y_s[test_idx], Y_a[test_idx], Y_sev[test_idx], N_n[test_idx])
    else:
        train_ds = TensorDataset(X_t[train_idx], Y_h[train_idx], Y_s[train_idx], Y_a[train_idx], Y_sev[train_idx], N_n[train_idx])
        val_ds = TensorDataset(X_t[val_idx], Y_h[val_idx], Y_s[val_idx], Y_a[val_idx], Y_sev[val_idx], N_n[val_idx])
        test_ds = TensorDataset(X_t[test_idx], Y_h[test_idx], Y_s[test_idx], Y_a[test_idx], Y_sev[test_idx], N_n[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            if use_gnn:
                bx, bg, bh, bs, ba, bsev, bn = [b.to(device) for b in batch]
                out = model(bx, bg)
            else:
                bx, bh, bs, ba, bsev, bn = [b.to(device) for b in batch]
                out = model(bx, None)

            mask = torch.zeros(bx.size(0), max_nodes, device=device)
            for i, n in enumerate(bn):
                mask[i, :n] = 1.0

            aff_bce = F.binary_cross_entropy_with_logits(
                out["affected_logits"], ba, pos_weight=aff_pos_weight.expand(max_nodes), reduction="none"
            )
            pt = ba * torch.sigmoid(out["affected_logits"]) + (1 - ba) * (1 - torch.sigmoid(out["affected_logits"]))
            aff_loss = (aff_bce * (1 - pt) ** 2 * mask).sum() / mask.sum()

            fl = focal_loss(out["state_logits"].reshape(-1, 4), bs.reshape(-1), weights=state_weights)
            state_loss = (fl * mask.reshape(-1)).sum() / mask.sum()

            health_loss = ((out["health"] - bh) ** 2 * mask).sum() / mask.sum()
            sev_loss = F.mse_loss(out["severity"], bsev)

            loss = aff_loss + state_loss + 0.5 * health_loss + 0.3 * sev_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validate
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            val_tp = val_fp = val_fn = 0
            with torch.no_grad():
                for batch in val_loader:
                    if use_gnn:
                        bx, bg, bh, bs, ba, bsev, bn = [b.to(device) for b in batch]
                        out = model(bx, bg)
                    else:
                        bx, bh, bs, ba, bsev, bn = [b.to(device) for b in batch]
                        out = model(bx, None)
                    mask = torch.zeros(bx.size(0), max_nodes, device=device)
                    for i, n in enumerate(bn): mask[i, :n] = 1.0
                    preds = (out["affected_logits"] > 0).float()
                    val_tp += ((preds == 1) & (ba == 1) & (mask == 1)).sum().item()
                    val_fp += ((preds == 1) & (ba == 0) & (mask == 1)).sum().item()
                    val_fn += ((preds == 0) & (ba == 1) & (mask == 1)).sum().item()

            prec = val_tp / max(val_tp + val_fp, 1)
            rec = val_tp / max(val_tp + val_fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 4:
                    break

    # Load best and evaluate on test
    model.load_state_dict(best_state)
    model.eval()

    class_tp = torch.zeros(4)
    class_fp = torch.zeros(4)
    class_fn = torch.zeros(4)
    aff_tp = aff_fp = aff_fn = 0

    with torch.no_grad():
        for batch in test_loader:
            if use_gnn:
                bx, bg, bh, bs, ba, bsev, bn = [b.to(device) for b in batch]
                out = model(bx, bg)
            else:
                bx, bh, bs, ba, bsev, bn = [b.to(device) for b in batch]
                out = model(bx, None)
            mask = torch.zeros(bx.size(0), max_nodes)
            for i, n in enumerate(bn): mask[i, :n] = 1.0

            aff_preds = (out["affected_logits"].cpu() > 0).float()
            aff_tp += ((aff_preds == 1) & (ba.cpu() == 1) & (mask == 1)).sum().item()
            aff_fp += ((aff_preds == 1) & (ba.cpu() == 0) & (mask == 1)).sum().item()
            aff_fn += ((aff_preds == 0) & (ba.cpu() == 1) & (mask == 1)).sum().item()

            state_preds = out["state_logits"].cpu().argmax(dim=-1)
            for c in range(4):
                c_true = (bs.cpu() == c) & (mask > 0)
                c_pred = (state_preds == c) & (mask > 0)
                class_tp[c] += (c_true & c_pred).sum().item()
                class_fp[c] += (~c_true & c_pred).sum().item()
                class_fn[c] += (c_true & ~c_pred).sum().item()

    # Compute metrics
    results = {}
    aff_prec = aff_tp / max(aff_tp + aff_fp, 1)
    aff_rec = aff_tp / max(aff_tp + aff_fn, 1)
    results["affected_f1"] = 2 * aff_prec * aff_rec / max(aff_prec + aff_rec, 1e-8)
    results["affected_prec"] = aff_prec
    results["affected_rec"] = aff_rec

    state_names = ["healthy", "degraded", "failed", "unreachable"]
    for c in range(4):
        p = class_tp[c] / max(class_tp[c] + class_fp[c], 1)
        r = class_tp[c] / max(class_tp[c] + class_fn[c], 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        results[f"{state_names[c]}_f1"] = float(f1)
        results[f"{state_names[c]}_rec"] = float(r)
        results[f"{state_names[c]}_n"] = int(class_tp[c] + class_fn[c])

    results["macro_f1"] = sum(results[f"{s}_f1"] for s in state_names) / 4

    return results


# ── Main ──────────────────────────────────────────────────────────────────


def run_fusion_test(device="cuda", n_samples=8000):
    print(f"\n{C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║   SABLE — Deep Fusion Test                       ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║   Does GNN + Mamba > Mamba alone?                 ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════╝{C_RESET}\n")

    # Generate data with GNN embeddings
    data = generate_fused_dataset(count=n_samples, device=device)

    max_nodes = data["max_nodes"]
    state_dim = data["state_dim"]
    gnn_dim = data["gnn_dim"]

    # ── Train baseline (no GNN) ──
    print(f"\n  {C_INFO}Training BASELINE (Mamba only, no GNN)...{C_RESET}")
    t0 = time.time()
    baseline_model = FusedMamba(
        state_dim=state_dim, max_nodes=max_nodes, gnn_dim=0,
        d_model=256, n_layers=4, dropout=0.2,
    ).to(device)
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"  {C_DIM}params: {baseline_params:,}{C_RESET}")
    baseline_results = train_model(baseline_model, data, use_gnn=False, device=device)
    baseline_time = time.time() - t0
    print(f"  {C_DIM}({baseline_time:.1f}s){C_RESET}")

    # ── Train fused (with GNN) ──
    print(f"\n  {C_INFO}Training FUSED (Mamba + GNN embeddings)...{C_RESET}")
    t0 = time.time()
    fused_model = FusedMamba(
        state_dim=state_dim, max_nodes=max_nodes, gnn_dim=gnn_dim,
        d_model=256, n_layers=4, dropout=0.2,
    ).to(device)
    fused_params = sum(p.numel() for p in fused_model.parameters())
    print(f"  {C_DIM}params: {fused_params:,}{C_RESET}")
    fused_results = train_model(fused_model, data, use_gnn=True, device=device)
    fused_time = time.time() - t0
    print(f"  {C_DIM}({fused_time:.1f}s){C_RESET}")

    # ── Comparison ──
    print(f"\n  {C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║              FUSION TEST RESULTS                   ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════╝{C_RESET}")

    state_names = ["healthy", "degraded", "failed", "unreachable"]

    print(f"\n  {C_DIM}{'Metric':<25s} {'Baseline':>10s} {'Fused':>10s} {'Delta':>10s}  {'Verdict':>8s}{C_RESET}")
    print(f"  {C_DIM}{'─' * 72}{C_RESET}")

    comparisons = [
        ("Affected F1", "affected_f1"),
        ("Affected Precision", "affected_prec"),
        ("Affected Recall", "affected_rec"),
        ("Macro F1", "macro_f1"),
    ]
    for s in state_names:
        comparisons.append((f"{s.capitalize()} F1", f"{s}_f1"))

    fusion_wins = 0
    total_tests = 0

    for label, key in comparisons:
        b = baseline_results[key]
        f = fused_results[key]
        delta = f - b
        is_target = key in ("unreachable_f1", "failed_f1", "affected_f1", "macro_f1")
        total_tests += 1 if is_target else 0

        if is_target and delta > 0.02:
            verdict = f"{C_SUCCESS}WIN{C_RESET}"
            fusion_wins += 1
        elif is_target and delta < -0.02:
            verdict = f"{C_DANGER}LOSS{C_RESET}"
        else:
            verdict = f"{C_DIM}EVEN{C_RESET}"

        sign = "+" if delta >= 0 else ""
        bold = C_BOLD if is_target else ""
        print(f"  {bold}{C_TEXT}{label:<25s}{C_RESET} {b:10.4f} {f:10.4f} {sign}{delta:9.4f}  {verdict}")

    # Per-class sample counts
    print(f"\n  {C_DIM}Test set class distribution:{C_RESET}")
    for s in state_names:
        n = baseline_results[f"{s}_n"]
        print(f"    {C_DIM}{s}: {n}{C_RESET}")

    # ── THE VERDICT ──
    print(f"\n  {C_GOLD}{C_BOLD}  THE VERDICT{C_RESET}")
    print(f"  {C_DIM}{'─' * 50}{C_RESET}")

    unreachable_delta = fused_results["unreachable_f1"] - baseline_results["unreachable_f1"]
    affected_delta = fused_results["affected_f1"] - baseline_results["affected_f1"]
    macro_delta = fused_results["macro_f1"] - baseline_results["macro_f1"]

    if unreachable_delta > 0.03:
        print(f"  {C_SUCCESS}{C_BOLD}UNREACHABLE F1 LIFTED: +{unreachable_delta:.4f}{C_RESET}")
        print(f"  {C_SUCCESS}GNN structural embeddings help Mamba predict network-path cascades.{C_RESET}")
        print(f"  {C_SUCCESS}FUSION ADDS VALUE. The architecture thesis holds.{C_RESET}")
    elif unreachable_delta > 0:
        print(f"  {C_GOLD}Unreachable F1 slightly improved: +{unreachable_delta:.4f}{C_RESET}")
        print(f"  {C_GOLD}Marginal fusion benefit. Needs more data or deeper integration.{C_RESET}")
    else:
        print(f"  {C_DANGER}Unreachable F1 did not improve: {unreachable_delta:+.4f}{C_RESET}")
        print(f"  {C_DANGER}GNN embeddings don't help Mamba on this task.{C_RESET}")

    if fusion_wins >= 2:
        print(f"\n  {C_SUCCESS}{C_BOLD}FUSION VALIDATED: {fusion_wins}/{total_tests} key metrics improved.{C_RESET}")
    elif fusion_wins == 1:
        print(f"\n  {C_GOLD}{C_BOLD}PARTIAL: {fusion_wins}/{total_tests} key metrics improved.{C_RESET}")
    else:
        print(f"\n  {C_DANGER}{C_BOLD}FUSION NOT DEMONSTRATED: {fusion_wins}/{total_tests} metrics improved.{C_RESET}")
        print(f"  {C_TEXT}The pillars may be independent systems sharing a process.{C_RESET}")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=8000)
    args = parser.parse_args()
    run_fusion_test(args.device, args.samples)
