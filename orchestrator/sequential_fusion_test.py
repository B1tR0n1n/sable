#!/usr/bin/env python3
"""
Project PARALLAX — Sequential Fusion Test
============================================
The REAL fusion test. Static GNN embeddings failed. This tests
DYNAMIC GNN embeddings — the GNN re-embeds at each tick, and
the CHANGE in structural representation becomes the signal.

The hypothesis: when a node fails, the GNN's embedding of its
neighbors SHIFTS because the message-passing now flows through
a broken node. That shift encodes "this part of the graph just
lost structural support" — information the raw state vector
doesn't contain.

Design:
  1. Tick 0: run GNN on healthy graph → embeddings_t0
  2. Tick 1: inject failure, run GNN again → embeddings_t1
  3. Compute delta: embeddings_t1 - embeddings_t0 (structural shift)
  4. Feed Mamba: [state_t0, state_t1, gnn_delta]
  5. Compare vs baseline: [state_t0, state_t1, state_delta]

The GNN delta captures HOW the graph's structural understanding
changed in response to the failure — which nodes lost support,
which paths broke, which clusters disconnected. The state delta
only captures which individual nodes changed health values.

Usage:
    python sequential_fusion_test.py --device cuda
"""

import sys
import time
from pathlib import Path

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
    build_random_topology, encode_system_state,
    NODE_FEAT_DIM, STATE_MAP,
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

GNN_EMBED_DIM = 256
GNN_CHECKPOINT = str(Path(__file__).parent.parent / "pillar1" / "checkpoints" / "best_model.pt")


# ── Sequential Fusion Model ──────────────────────────────────────────────


class SequentialFusedMamba(nn.Module):
    """Mamba with GNN structural DELTA as input.

    Input: [state_t0, state_t1, state_delta, gnn_delta]
    The gnn_delta captures how the graph's structural understanding
    changed between tick 0 and tick 1 — where structural support was lost.
    """

    def __init__(self, state_dim, max_nodes, gnn_delta_dim=0,
                 d_model=256, n_layers=4, d_state=16, dropout=0.2,
                 n_states=4):
        super().__init__()
        self.max_nodes = max_nodes
        self.n_states = n_states
        self.gnn_delta_dim = gnn_delta_dim

        # Input: state_t0 + state_t1 + state_delta + gnn_delta
        total_dim = state_dim * 3 + gnn_delta_dim

        self.input_proj = nn.Sequential(
            nn.Linear(total_dim, d_model * 2),
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

    def forward(self, state_t0, state_t1, gnn_delta=None):
        B = state_t0.size(0)
        state_delta = state_t1 - state_t0

        if gnn_delta is not None and self.gnn_delta_dim > 0:
            x = torch.cat([state_t0, state_t1, state_delta, gnn_delta], dim=-1)
        else:
            x = torch.cat([state_t0, state_t1, state_delta], dim=-1)

        h = self.input_proj(x).unsqueeze(1)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h).squeeze(1)

        return {
            "affected_logits": self.affected_head(h),
            "state_logits": self.state_head(h).view(B, self.max_nodes, self.n_states),
        }


# ── Data Generation with Dynamic GNN Embeddings ──────────────────────────


def compute_gnn_embeddings(gnn, graph, components, component_ids, adapter, device):
    """Run GNN on an infrastructure graph and return per-node embeddings."""
    node_features = []
    for comp in components:
        feat = adapter.adapt_node(str(comp.type), comp.health)
        node_features.append(feat)
    x = torch.tensor(np.array(node_features), dtype=torch.float32).to(device)

    sources, targets, edge_feats = [], [], []
    for comp in components:
        for dep_id in comp.dependencies_in:
            src_idx = component_ids.index(comp.id) if comp.id in component_ids else None
            tgt_idx = component_ids.index(dep_id) if dep_id in component_ids else None
            if src_idx is not None and tgt_idx is not None:
                dep = graph.get_dependency(comp.id, dep_id)
                if dep:
                    _, feat = adapter.adapt_edge(str(dep.type), str(dep.criticality))
                    sources.append(src_idx)
                    targets.append(tgt_idx)
                    edge_feats.append(feat)

    if not sources:
        return torch.zeros(len(components), GNN_EMBED_DIM, device=device)

    edge_index = torch.tensor([sources, targets], dtype=torch.long).to(device)
    edge_attr = torch.tensor(np.array(edge_feats), dtype=torch.float32).to(device)

    with torch.no_grad():
        node_emb = gnn.encode(x, edge_index, edge_attr)
    return node_emb


def generate_sequential_data(count=8000, device="cuda", seed=42):
    """Generate data with DYNAMIC GNN embeddings at each tick."""

    print(f"  {C_INFO}Generating {count} sequences with dynamic GNN...{C_RESET}")

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
    gnn_delta_dim = max_nodes * GNN_EMBED_DIM

    S_t0 = torch.zeros(count, state_dim)
    S_t1 = torch.zeros(count, state_dim)
    G_delta = torch.zeros(count, gnn_delta_dim)
    Y_state = torch.zeros(count, max_nodes, dtype=torch.long)
    Y_affected = torch.zeros(count, max_nodes)
    Y_severity = torch.zeros(count)
    N_nodes = torch.zeros(count, dtype=torch.long)

    generated = 0
    t0 = time.time()

    while generated < count:
        graph = build_random_topology(rng, 15, 40)
        components = graph.get_all_components()
        if len(components) < 15:
            continue
        component_ids = [c.id for c in components]
        n = len(component_ids)

        # Tick 0: encode state AND run GNN on healthy graph
        s0 = encode_system_state(graph, component_ids)
        gnn_t0 = compute_gnn_embeddings(gnn, graph, components, component_ids, adapter, device)

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

        use_degradation = rng.random() < 0.4
        for inj in injections:
            injector.inject(state, inj)
            if use_degradation:
                comp = graph.get_component(inj.component_id)
                if comp and comp.state == ComponentState.FAILED:
                    comp.state = ComponentState.DEGRADED
                    comp.health = rng.uniform(0.25, 0.45)

        # Tick 1: encode state AND run GNN on damaged graph
        s1 = encode_system_state(graph, component_ids)
        gnn_t1 = compute_gnn_embeddings(gnn, graph, components, component_ids, adapter, device)

        # GNN delta: how structural understanding changed
        gnn_d = gnn_t1 - gnn_t0  # (n, 256)

        # Propagate to get outcome
        engine = PropagationEngine(max_ticks=30, soft_impact_factor=0.4 if use_degradation else 0.3)
        engine.propagate(state)

        # Targets
        for i, cid in enumerate(component_ids):
            comp = graph.get_component(cid)
            if comp:
                Y_state[generated, i] = STATE_MAP.get(comp.state, 0)
                Y_affected[generated, i] = 0.0 if comp.state == ComponentState.HEALTHY else 1.0

        Y_severity[generated] = Y_affected[generated, :n].sum().item() / max(n, 1)
        N_nodes[generated] = n

        actual_dim = n * NODE_FEAT_DIM
        S_t0[generated, :actual_dim] = torch.tensor(s0[:actual_dim])
        S_t1[generated, :actual_dim] = torch.tensor(s1[:actual_dim])

        # Pad GNN delta to max_nodes
        padded = torch.zeros(max_nodes, GNN_EMBED_DIM, device=device)
        padded[:n] = gnn_d
        G_delta[generated] = padded.flatten().cpu()

        generated += 1
        if generated % 1000 == 0:
            print(f"    {C_DIM}{generated}/{count} ({time.time()-t0:.1f}s){C_RESET}")

    print(f"  {C_TEXT}Generated {generated} sequences in {time.time()-t0:.1f}s{C_RESET}")

    perm = torch.randperm(generated)
    n_train = int(generated * 0.8)
    n_val = int(generated * 0.1)

    return {
        "S_t0": S_t0, "S_t1": S_t1, "G_delta": G_delta,
        "Y_state": Y_state, "Y_affected": Y_affected, "Y_severity": Y_severity,
        "N_nodes": N_nodes,
        "train_idx": perm[:n_train], "val_idx": perm[n_train:n_train+n_val],
        "test_idx": perm[n_train+n_val:],
        "max_nodes": max_nodes, "state_dim": state_dim,
        "gnn_delta_dim": gnn_delta_dim,
    }


# ── Training ──────────────────────────────────────────────────────────────


def focal_loss(logits, targets, gamma=2.0, weights=None):
    ce = F.cross_entropy(logits, targets, weight=weights, reduction="none")
    return ((1 - torch.exp(-ce)) ** gamma * ce)


def train_and_eval(model, data, use_gnn_delta, device, epochs=60, batch_size=128, lr=0.0005):
    """Train and evaluate, return per-class metrics."""
    max_nodes = data["max_nodes"]
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]

    state_counts = torch.bincount(data["Y_state"][train_idx].flatten(), minlength=4).float().clamp(min=1)
    state_weights = (data["Y_state"][train_idx].numel() / (4 * state_counts)).clamp(max=10.0).to(device)

    n_aff = data["Y_affected"][train_idx].sum().item()
    n_hlt = (data["Y_affected"][train_idx] == 0).sum().item()
    aff_pw = torch.tensor([n_hlt / max(n_aff, 1)], device=device).clamp(max=15.0)

    sw = torch.ones(len(train_idx))
    for i, idx in enumerate(train_idx):
        sw[i] = 1.0 + 4.0 * data["Y_severity"][idx].item()
    sampler = WeightedRandomSampler(sw, len(train_idx), replacement=True)

    fields = [data["S_t0"], data["S_t1"]]
    if use_gnn_delta:
        fields.append(data["G_delta"])
    fields += [data["Y_state"], data["Y_affected"], data["N_nodes"]]

    def make_ds(idx):
        return TensorDataset(*[f[idx] for f in fields])

    train_loader = DataLoader(make_ds(train_idx), batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(make_ds(val_idx), batch_size=batch_size)
    test_loader = DataLoader(make_ds(test_idx), batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = [b.to(device) for b in batch]
            if use_gnn_delta:
                s0, s1, gd, ys, ya, bn = batch
                out = model(s0, s1, gd)
            else:
                s0, s1, ys, ya, bn = batch
                out = model(s0, s1, None)

            mask = torch.zeros(s0.size(0), max_nodes, device=device)
            for i, n in enumerate(bn): mask[i, :n] = 1.0

            aff_bce = F.binary_cross_entropy_with_logits(
                out["affected_logits"], ya, pos_weight=aff_pw.expand(max_nodes), reduction="none"
            )
            pt = ya * torch.sigmoid(out["affected_logits"]) + (1-ya) * (1-torch.sigmoid(out["affected_logits"]))
            aff_loss = (aff_bce * (1-pt)**2 * mask).sum() / mask.sum()

            fl = focal_loss(out["state_logits"].reshape(-1, 4), ys.reshape(-1), weights=state_weights)
            state_loss = (fl * mask.reshape(-1)).sum() / mask.sum()

            loss = aff_loss + state_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            tp = fp = fn = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = [b.to(device) for b in batch]
                    if use_gnn_delta:
                        s0, s1, gd, ys, ya, bn = batch
                        out = model(s0, s1, gd)
                    else:
                        s0, s1, ys, ya, bn = batch
                        out = model(s0, s1, None)
                    mask = torch.zeros(s0.size(0), max_nodes, device=device)
                    for i, n in enumerate(bn): mask[i, :n] = 1.0
                    preds = (out["affected_logits"] > 0).float()
                    tp += ((preds==1)&(ya==1)&(mask==1)).sum().item()
                    fp += ((preds==1)&(ya==0)&(mask==1)).sum().item()
                    fn += ((preds==0)&(ya==1)&(mask==1)).sum().item()

            p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
            f1 = 2*p*r/max(p+r,1e-8)
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 4: break

    model.load_state_dict(best_state)
    model.eval()

    class_tp = torch.zeros(4); class_fp = torch.zeros(4); class_fn = torch.zeros(4)
    aff_tp = aff_fp = aff_fn = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = [b.to(device) for b in batch]
            if use_gnn_delta:
                s0, s1, gd, ys, ya, bn = batch
                out = model(s0, s1, gd)
            else:
                s0, s1, ys, ya, bn = batch
                out = model(s0, s1, None)
            mask = torch.zeros(s0.size(0), max_nodes)
            for i, n in enumerate(bn): mask[i, :n] = 1.0

            ap = (out["affected_logits"].cpu() > 0).float()
            aff_tp += ((ap==1)&(ya.cpu()==1)&(mask==1)).sum().item()
            aff_fp += ((ap==1)&(ya.cpu()==0)&(mask==1)).sum().item()
            aff_fn += ((ap==0)&(ya.cpu()==1)&(mask==1)).sum().item()

            sp = out["state_logits"].cpu().argmax(dim=-1)
            for c in range(4):
                ct = (ys.cpu()==c)&(mask>0); cp = (sp==c)&(mask>0)
                class_tp[c] += (ct&cp).sum().item()
                class_fp[c] += (~ct&cp).sum().item()
                class_fn[c] += (ct&~cp).sum().item()

    results = {}
    p = aff_tp/max(aff_tp+aff_fp,1); r = aff_tp/max(aff_tp+aff_fn,1)
    results["affected_f1"] = 2*p*r/max(p+r,1e-8)

    names = ["healthy","degraded","failed","unreachable"]
    for c in range(4):
        p = class_tp[c]/max(class_tp[c]+class_fp[c],1)
        r = class_tp[c]/max(class_tp[c]+class_fn[c],1)
        results[f"{names[c]}_f1"] = float(2*p*r/max(p+r,1e-8))
        results[f"{names[c]}_n"] = int(class_tp[c]+class_fn[c])
    results["macro_f1"] = sum(results[f"{s}_f1"] for s in names)/4

    return results


# ── Main ──────────────────────────────────────────────────────────────────


def run(device="cuda", n_samples=8000):
    print(f"\n{C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║   SABLE — Sequential Fusion Test                 ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║   Does the GNN's REACTION to failure help Mamba?  ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════╝{C_RESET}\n")

    data = generate_sequential_data(count=n_samples, device=device)
    state_dim = data["state_dim"]
    max_nodes = data["max_nodes"]
    gnn_delta_dim = data["gnn_delta_dim"]

    # Baseline: state only
    print(f"\n  {C_INFO}Training BASELINE (state delta only)...{C_RESET}")
    t0 = time.time()
    baseline = SequentialFusedMamba(state_dim, max_nodes, gnn_delta_dim=0, d_model=256, n_layers=4).to(device)
    bp = sum(p.numel() for p in baseline.parameters())
    print(f"  {C_DIM}params: {bp:,}{C_RESET}")
    b_results = train_and_eval(baseline, data, use_gnn_delta=False, device=device)
    print(f"  {C_DIM}({time.time()-t0:.1f}s){C_RESET}")

    # Fused: state + GNN delta
    print(f"\n  {C_INFO}Training FUSED (state delta + GNN structural delta)...{C_RESET}")
    t0 = time.time()
    fused = SequentialFusedMamba(state_dim, max_nodes, gnn_delta_dim=gnn_delta_dim, d_model=256, n_layers=4).to(device)
    fp = sum(p.numel() for p in fused.parameters())
    print(f"  {C_DIM}params: {fp:,}{C_RESET}")
    f_results = train_and_eval(fused, data, use_gnn_delta=True, device=device)
    print(f"  {C_DIM}({time.time()-t0:.1f}s){C_RESET}")

    # Results
    print(f"\n  {C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║         SEQUENTIAL FUSION RESULTS                  ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════╝{C_RESET}")

    metrics = [
        ("Affected F1", "affected_f1", True),
        ("Macro F1", "macro_f1", True),
        ("Healthy F1", "healthy_f1", False),
        ("Degraded F1", "degraded_f1", False),
        ("Failed F1", "failed_f1", True),
        ("Unreachable F1", "unreachable_f1", True),
    ]

    print(f"\n  {C_DIM}{'Metric':<22s} {'Baseline':>10s} {'Fused':>10s} {'Delta':>10s}  {'':>8s}{C_RESET}")
    print(f"  {C_DIM}{'─' * 68}{C_RESET}")

    wins = 0
    for label, key, is_target in metrics:
        b = b_results[key]; f = f_results[key]; d = f - b
        if is_target and d > 0.02:
            v = f"{C_SUCCESS}WIN{C_RESET}"; wins += 1
        elif is_target and d < -0.02:
            v = f"{C_DANGER}LOSS{C_RESET}"
        else:
            v = f"{C_DIM}EVEN{C_RESET}"
        bold = C_BOLD if is_target else ""
        print(f"  {bold}{C_TEXT}{label:<22s}{C_RESET} {b:10.4f} {f:10.4f} {'+' if d>=0 else ''}{d:9.4f}  {v}")

    # Class counts
    print(f"\n  {C_DIM}Test set:{C_RESET}")
    for s in ["healthy","degraded","failed","unreachable"]:
        print(f"    {C_DIM}{s}: {b_results[f'{s}_n']}{C_RESET}")

    # Verdict
    ud = f_results["unreachable_f1"] - b_results["unreachable_f1"]

    print(f"\n  {C_GOLD}{C_BOLD}  VERDICT{C_RESET}")
    print(f"  {C_DIM}{'─' * 50}{C_RESET}")

    if ud > 0.03:
        print(f"  {C_SUCCESS}{C_BOLD}UNREACHABLE LIFTED: +{ud:.4f}{C_RESET}")
        print(f"  {C_SUCCESS}The GNN's structural REACTION to failure contains{C_RESET}")
        print(f"  {C_SUCCESS}information raw state vectors don't have.{C_RESET}")
    elif ud > 0:
        print(f"  {C_GOLD}Unreachable marginal: +{ud:.4f}{C_RESET}")
    else:
        print(f"  {C_DANGER}Unreachable not improved: {ud:+.4f}{C_RESET}")

    if wins >= 2:
        print(f"\n  {C_SUCCESS}{C_BOLD}SEQUENTIAL FUSION VALIDATED: {wins}/4 key metrics improved.{C_RESET}")
        print(f"  {C_SUCCESS}Dynamic GNN re-embedding adds value over static state vectors.{C_RESET}")
    elif wins == 1:
        print(f"\n  {C_GOLD}{C_BOLD}PARTIAL: {wins}/4. Some signal but not conclusive.{C_RESET}")
    else:
        print(f"\n  {C_DANGER}{C_BOLD}SEQUENTIAL FUSION NOT DEMONSTRATED: {wins}/4.{C_RESET}")
        print(f"  {C_TEXT}Neither static nor dynamic GNN features help Mamba.{C_RESET}")
        print(f"  {C_TEXT}The pillars provide independent value through separate outputs,{C_RESET}")
        print(f"  {C_TEXT}not through shared feature spaces.{C_RESET}")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=8000)
    args = parser.parse_args()
    run(args.device, args.samples)
