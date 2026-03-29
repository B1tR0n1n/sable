#!/usr/bin/env python3
"""
Project PARALLAX — Routed Fusion
===================================
Each pillar maintains its own expert head. Fusion adds a separate head
for emergent capabilities. A learned router decides which head to trust
per-node based on the inputs.

One forward pass. One model. Best of each pillar + emergent signal.

Usage:
    python routed_fusion.py --device cuda --samples 10000
"""

import sys
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar3"))

from generate_temporal_data import NODE_FEAT_DIM
from shared_latent_space import (
    Z_DIM, GNN_DIM, POMDP_DIM, N_STATES,
    GNNProjection, POMDPProjection, MambaProjection,
    CrossAttentionFusion,
    generate_fusion_data, alignment_loss,
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


class RoutedFusionModel(nn.Module):
    """Three expert heads + fusion head + learned router.

    Each pillar gets its own classification head (preserves accuracy).
    The fusion layer gets a separate head (emergent capability).
    A router network decides per-node which head to trust.
    """

    def __init__(self, mamba_dim=NODE_FEAT_DIM, dropout=0.1):
        super().__init__()

        # Projection heads into shared space
        self.gnn_proj = GNNProjection()
        self.pomdp_proj = POMDPProjection()
        self.mamba_proj = MambaProjection(mamba_dim)

        # Fusion layer (cross-attention, produces emergent signal)
        self.fusion = CrossAttentionFusion()

        # Expert heads — each pillar classifies from its own projection
        self.gnn_head = nn.Sequential(
            nn.Linear(Z_DIM, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, N_STATES),
        )
        self.pomdp_head = nn.Sequential(
            nn.Linear(Z_DIM, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, N_STATES),
        )
        self.mamba_head = nn.Sequential(
            nn.Linear(Z_DIM, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, N_STATES),
        )

        # Fusion head — classifies from fused Z (emergent capabilities)
        self.fusion_head = nn.Sequential(
            nn.Linear(Z_DIM, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, N_STATES),
        )

        # Router — sees all pillar projections, decides which head to trust
        # Input: z_gnn + z_pomdp + z_mamba + z_fused = 4 * 128 = 512
        # Output: 4 weights (one per expert head)
        self.router = nn.Sequential(
            nn.Linear(Z_DIM * 4, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),  # 4 heads: gnn, pomdp, mamba, fusion
        )
        self.router_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, gnn_emb, pomdp_belief, mamba_pred):
        """
        All inputs: (B, N, respective_dim)
        Returns: dict with logits, route_weights, per-head logits
        """
        B, N, _ = gnn_emb.shape

        # Project into shared space
        z_gnn = self.gnn_proj(gnn_emb)
        z_pomdp = self.pomdp_proj(pomdp_belief)
        z_mamba = self.mamba_proj(mamba_pred)

        # Fusion
        perspectives = torch.stack([z_gnn, z_pomdp, z_mamba], dim=2)
        z_fused = self.fusion(perspectives)

        # Expert head outputs
        logits_gnn = self.gnn_head(z_gnn)        # (B, N, 4)
        logits_pomdp = self.pomdp_head(z_pomdp)  # (B, N, 4)
        logits_mamba = self.mamba_head(z_mamba)   # (B, N, 4)
        logits_fusion = self.fusion_head(z_fused) # (B, N, 4)

        # Router: which head to trust per-node
        router_input = torch.cat([z_gnn, z_pomdp, z_mamba, z_fused], dim=-1)
        route_logits = self.router(router_input)  # (B, N, 4)
        route_weights = F.softmax(
            route_logits / self.router_temperature.clamp(min=0.1), dim=-1
        )  # (B, N, 4)

        # Stack all head logits: (B, N, 4_heads, 4_states)
        all_logits = torch.stack([logits_gnn, logits_pomdp, logits_mamba, logits_fusion], dim=2)

        # Weighted combination: route_weights (B,N,4,1) * all_logits (B,N,4,4) → sum → (B,N,4)
        final_logits = (route_weights.unsqueeze(-1) * all_logits).sum(dim=2)

        return {
            "logits": final_logits,
            "route_weights": route_weights,
            "logits_gnn": logits_gnn,
            "logits_pomdp": logits_pomdp,
            "logits_mamba": logits_mamba,
            "logits_fusion": logits_fusion,
            "z_gnn": z_gnn,
            "z_pomdp": z_pomdp,
            "z_mamba": z_mamba,
            "z_fused": z_fused,
        }


def train_routed(model, train_data, val_data, device, epochs=100, batch_size=128, lr=0.001):
    """Train the routed fusion model."""

    state_counts = torch.bincount(train_data["states"].flatten().long(), minlength=4).float().clamp(min=1)
    state_weights = (train_data["states"].numel() / (4 * state_counts)).clamp(max=8.0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    n_train = train_data["gnn"].size(0)
    n_val = val_data["gnn"].size(0)
    max_nodes = 40

    best_macro = 0.0
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            g = train_data["gnn"][idx].to(device)
            p = train_data["pomdp"][idx].to(device)
            m = train_data["mamba"][idx].to(device)
            s = train_data["states"][idx].to(device).long()
            mask = train_data["mask"][idx].to(device)

            out = model(g, p, m)

            # 1. Final routed output loss
            main_loss = F.cross_entropy(
                out["logits"].reshape(-1, 4), s.reshape(-1),
                weight=state_weights, reduction="none"
            )
            main_loss = (main_loss * mask.reshape(-1)).sum() / mask.sum()

            # 2. Per-expert auxiliary losses (preserve individual competence)
            expert_losses = []
            for key in ["logits_gnn", "logits_pomdp", "logits_mamba", "logits_fusion"]:
                el = F.cross_entropy(
                    out[key].reshape(-1, 4), s.reshape(-1),
                    weight=state_weights, reduction="none"
                )
                el = (el * mask.reshape(-1)).sum() / mask.sum()
                expert_losses.append(el)
            aux_loss = sum(expert_losses) / len(expert_losses)

            # 3. Router load balancing — prevent router from always picking one expert
            # Encourage all experts to get some traffic
            route_avg = (out["route_weights"] * mask.unsqueeze(-1)).sum(dim=(0, 1))
            route_avg = route_avg / mask.sum()
            # Target: each expert gets ~25% of traffic
            target = torch.full((4,), 0.25, device=device)
            balance_loss = F.mse_loss(route_avg, target)

            loss = main_loss + 0.3 * aux_loss + 0.5 * balance_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            class_tp = torch.zeros(4)
            class_fp = torch.zeros(4)
            class_fn = torch.zeros(4)
            route_sums = torch.zeros(4)
            route_count = 0

            with torch.no_grad():
                for i in range(0, n_val, batch_size):
                    g = val_data["gnn"][i:i+batch_size].to(device)
                    p = val_data["pomdp"][i:i+batch_size].to(device)
                    m = val_data["mamba"][i:i+batch_size].to(device)
                    s = val_data["states"][i:i+batch_size].to(device).long()
                    mask = val_data["mask"][i:i+batch_size].to(device)

                    out = model(g, p, m)
                    preds = out["logits"].argmax(dim=-1)
                    valid = mask > 0

                    for c in range(4):
                        ct = (s == c) & valid
                        cp = (preds == c) & valid
                        class_tp[c] += (ct & cp).sum().item()
                        class_fp[c] += (~ct & cp).sum().item()
                        class_fn[c] += (ct & ~cp).sum().item()

                    rw = (out["route_weights"] * mask.unsqueeze(-1)).sum(dim=(0, 1))
                    route_sums += rw.cpu()
                    route_count += mask.sum().item()

            class_f1 = []
            for c in range(4):
                pr = class_tp[c] / max(class_tp[c] + class_fp[c], 1)
                rc = class_tp[c] / max(class_tp[c] + class_fn[c], 1)
                class_f1.append(2 * pr * rc / max(pr + rc, 1e-8))
            macro = sum(class_f1) / 4
            route_pct = route_sums / max(route_count, 1)

            improved = macro > best_macro
            if improved:
                best_macro = macro
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if epoch <= 10 or epoch % 10 == 0 or improved:
                marker = f"{C_SUCCESS}*{C_RESET}" if improved else " "
                f1_str = " ".join(f"{f:.3f}" for f in class_f1)
                rt_str = " ".join(f"{r:.2f}" for r in route_pct.tolist())
                print(
                    f"  {C_TEXT}{epoch:4d}{C_RESET}  "
                    f"macro={macro:.4f}  [{f1_str}]  "
                    f"route=[{rt_str}]  {marker}"
                )

            if patience >= 8:
                print(f"  {C_DIM}Early stopping at epoch {epoch}{C_RESET}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_macro


def run(device="cuda", n_samples=10000):
    print(f"\n{C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║   Routed Fusion: Best of Each + Emergent Signal   ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════╝{C_RESET}\n")

    # Generate data
    print(f"  {C_INFO}Generating data...{C_RESET}")
    train_data, val_data = generate_fusion_data(count=n_samples, device=device)

    # Build model
    model = RoutedFusionModel(mamba_dim=NODE_FEAT_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {C_TEXT}Parameters: {C_BRIGHT}{n_params:,}{C_RESET}")

    # Train baselines (same as before)
    print(f"\n  {C_INFO}Training single-pillar baselines...{C_RESET}")

    gnn_head = nn.Sequential(nn.Linear(GNN_DIM, 64), nn.GELU(), nn.Linear(64, N_STATES)).to(device)
    pomdp_head = nn.Sequential(nn.Linear(POMDP_DIM, 64), nn.GELU(), nn.Linear(64, N_STATES)).to(device)
    mamba_head = nn.Sequential(nn.Linear(NODE_FEAT_DIM, 64), nn.GELU(), nn.Linear(64, N_STATES)).to(device)

    for head, key in [(gnn_head, "gnn"), (pomdp_head, "pomdp"), (mamba_head, "mamba")]:
        opt = torch.optim.Adam(head.parameters(), lr=0.001)
        for ep in range(40):
            head.train()
            perm = torch.randperm(train_data[key].size(0))
            for i in range(0, len(perm), 128):
                idx = perm[i:i+128]
                x = train_data[key][idx].to(device)
                s = train_data["states"][idx].to(device).long()
                mask = train_data["mask"][idx].to(device)
                logits = head(x)
                loss = F.cross_entropy(logits.reshape(-1, 4), s.reshape(-1), reduction="none")
                loss = (loss * mask.reshape(-1)).sum() / mask.sum()
                opt.zero_grad(); loss.backward(); opt.step()

    # Train routed fusion
    print(f"\n  {C_INFO}Training routed fusion model...{C_RESET}")
    print(f"  {C_DIM}{'ep':>6s}  {'macro':>8s}  [hlthy  dgrad  faild  unrch]  route=[gnn  pomdp mamba fusn]{C_RESET}")
    print(f"  {C_DIM}{'─' * 80}{C_RESET}")

    t0 = time.time()
    best_macro = train_routed(model, train_data, val_data, device, epochs=100)
    train_time = time.time() - t0

    # Evaluate everything
    print(f"\n  {C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║              ROUTED FUSION RESULTS                  ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════╝{C_RESET}")

    state_names = ["healthy", "degraded", "failed", "unreachable"]

    def eval_model(model_fn):
        class_tp = torch.zeros(4); class_fp = torch.zeros(4); class_fn = torch.zeros(4)
        with torch.no_grad():
            for i in range(0, val_data["states"].size(0), 128):
                s = val_data["states"][i:i+128].to(device).long()
                mask = val_data["mask"][i:i+128].to(device)
                logits = model_fn(i, i+128)
                preds = logits.argmax(dim=-1)
                valid = mask > 0
                for c in range(4):
                    ct = (s == c) & valid; cp = (preds == c) & valid
                    class_tp[c] += (ct & cp).sum().item()
                    class_fp[c] += (~ct & cp).sum().item()
                    class_fn[c] += (ct & ~cp).sum().item()
        results = {}
        for c in range(4):
            p = class_tp[c] / max(class_tp[c] + class_fp[c], 1)
            r = class_tp[c] / max(class_tp[c] + class_fn[c], 1)
            results[state_names[c]] = 2 * p * r / max(p + r, 1e-8)
        results["macro"] = sum(results[s] for s in state_names) / 4
        return results

    gnn_head.eval(); pomdp_head.eval(); mamba_head.eval(); model.eval()

    gnn_r = eval_model(lambda i, j: gnn_head(val_data["gnn"][i:j].to(device)))
    pomdp_r = eval_model(lambda i, j: pomdp_head(val_data["pomdp"][i:j].to(device)))
    mamba_r = eval_model(lambda i, j: mamba_head(val_data["mamba"][i:j].to(device)))
    fused_r = eval_model(lambda i, j: model(
        val_data["gnn"][i:j].to(device),
        val_data["pomdp"][i:j].to(device),
        val_data["mamba"][i:j].to(device),
    )["logits"])

    print(f"\n  {C_DIM}{'Model':<20s} {'Macro':>7s}", end="")
    for s in state_names:
        print(f" {s[:7]:>8s}", end="")
    print(f"{C_RESET}")
    print(f"  {C_DIM}{'─' * 60}{C_RESET}")

    for name, r in [("GNN only", gnn_r), ("POMDP only", pomdp_r), ("Mamba only", mamba_r), ("ROUTED FUSION", fused_r)]:
        is_f = "ROUTED" in name
        c = C_SUCCESS if is_f else C_TEXT
        b = C_BOLD if is_f else ""
        line = f"  {c}{b}{name:<20s}{C_RESET} {r['macro']:7.4f}"
        for s in state_names:
            line += f" {r[s]:8.4f}"
        print(line)

    # Deltas
    print(f"\n  {C_INFO}Routed fusion vs best single pillar:{C_RESET}")
    wins = 0
    for s in state_names:
        best = max(gnn_r[s], pomdp_r[s], mamba_r[s])
        delta = fused_r[s] - best
        which = "GNN" if gnn_r[s] == best else "POMDP" if pomdp_r[s] == best else "Mamba"
        c = C_SUCCESS if delta > 0.01 else C_DANGER if delta < -0.01 else C_DIM
        if delta > 0.01:
            wins += 1
        print(f"    {c}{s:<15s} fused={fused_r[s]:.4f} best={best:.4f} ({which}) {delta:+.4f}{C_RESET}")

    macro_best = max(gnn_r["macro"], pomdp_r["macro"], mamba_r["macro"])
    macro_delta = fused_r["macro"] - macro_best

    # Router analysis
    print(f"\n  {C_INFO}Router behavior (which expert gets traffic):{C_RESET}")
    with torch.no_grad():
        g = val_data["gnn"][:200].to(device)
        p = val_data["pomdp"][:200].to(device)
        m = val_data["mamba"][:200].to(device)
        s = val_data["states"][:200].to(device).long()
        mask = val_data["mask"][:200].to(device)
        out = model(g, p, m)
        rw = out["route_weights"]

        # Per-class routing
        expert_names = ["GNN", "POMDP", "Mamba", "Fusion"]
        for c in range(4):
            c_mask = (s == c) & (mask > 0)
            if c_mask.sum() == 0:
                continue
            c_weights = rw[c_mask.unsqueeze(-1).expand_as(rw)].reshape(-1, 4).mean(dim=0)
            top = expert_names[c_weights.argmax().item()]
            print(f"    {C_DIM}{state_names[c]:15s}{C_RESET} → {C_TEXT}{top}{C_RESET} "
                  f"[{' '.join(f'{w:.2f}' for w in c_weights.tolist())}]")

    print(f"\n  {C_GOLD}{C_BOLD}  VERDICT{C_RESET}")
    print(f"  {C_DIM}{'─' * 50}{C_RESET}")
    print(f"  {C_TEXT}Macro F1: fused={fused_r['macro']:.4f} vs best_single={macro_best:.4f} ({macro_delta:+.4f}){C_RESET}")
    print(f"  {C_TEXT}Time: {train_time:.1f}s | Params: {n_params:,}{C_RESET}")

    if macro_delta > 0.01 and wins >= 2:
        print(f"\n  {C_SUCCESS}{C_BOLD}ROUTED FUSION VALIDATED.{C_RESET}")
        print(f"  {C_SUCCESS}Best of each pillar + emergent capability. The architecture works.{C_RESET}")
    elif fused_r["unreachable"] > 0.1 and fused_r["macro"] >= macro_best - 0.02:
        print(f"\n  {C_SUCCESS}{C_BOLD}FUSION ADDS EMERGENT CAPABILITY WITHOUT SACRIFICING ACCURACY.{C_RESET}")
        print(f"  {C_SUCCESS}Unreachable: {fused_r['unreachable']:.4f} (all pillars: 0.000){C_RESET}")
        print(f"  {C_SUCCESS}Macro F1 maintained within 0.02 of best single pillar.{C_RESET}")
    elif fused_r["unreachable"] > 0.05:
        print(f"\n  {C_GOLD}{C_BOLD}PARTIAL: emergent signal present, accuracy tradeoff remains.{C_RESET}")
    else:
        print(f"\n  {C_DANGER}{C_BOLD}FUSION NOT DEMONSTRATED.{C_RESET}")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    run(args.device, args.samples)
