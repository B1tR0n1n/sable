#!/usr/bin/env python3
"""
Project PARALLAX — Staged Fusion
===================================
Stage 1: Train each expert head independently on its pillar's native features.
Stage 2: Freeze experts. Train ONLY the fusion projection + fusion head + router.

The experts can't degrade because they're frozen.
The fusion adds emergent capability on top.

Usage:
    python staged_fusion.py --device cuda --samples 10000
"""

import sys
import time
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
    generate_fusion_data,
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


class StagedFusionModel(nn.Module):
    """Two-stage architecture.

    Stage 1 components (frozen after pre-training):
      - GNN expert: GNN_DIM → 4 classes
      - POMDP expert: POMDP_DIM → 4 classes
      - Mamba expert: MAMBA_DIM → 4 classes

    Stage 2 components (trained on frozen experts):
      - Pillar projections into shared space
      - Cross-attention fusion
      - Fusion expert head
      - Router (over 4 experts: 3 frozen + 1 fusion)
    """

    def __init__(self, mamba_dim=NODE_FEAT_DIM, dropout=0.1):
        super().__init__()

        # Stage 1: Expert heads (trained first, then frozen)
        self.gnn_expert = nn.Sequential(
            nn.Linear(GNN_DIM, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, N_STATES),
        )
        self.pomdp_expert = nn.Sequential(
            nn.Linear(POMDP_DIM, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, N_STATES),
        )
        self.mamba_expert = nn.Sequential(
            nn.Linear(mamba_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, N_STATES),
        )

        # Stage 2: Fusion components
        self.gnn_proj = GNNProjection()
        self.pomdp_proj = POMDPProjection()
        self.mamba_proj = MambaProjection(mamba_dim)
        self.fusion = CrossAttentionFusion()
        self.fusion_expert = nn.Sequential(
            nn.Linear(Z_DIM, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, N_STATES),
        )

        # Router: sees raw pillar features + expert logits, decides routing
        # Input: expert logits (4*3=12) + fusion logits (4) + pillar feature norms (3) = 19
        self.router = nn.Sequential(
            nn.Linear(4 * N_STATES + 3, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4),  # 4 expert heads
        )
        self.router_temp = nn.Parameter(torch.tensor(0.5))

    def forward(self, gnn_raw, pomdp_raw, mamba_raw):
        B, N, _ = gnn_raw.shape

        # Expert outputs
        l_gnn = self.gnn_expert(gnn_raw)
        l_pomdp = self.pomdp_expert(pomdp_raw)
        l_mamba = self.mamba_expert(mamba_raw)

        # Fusion
        z_gnn = self.gnn_proj(gnn_raw)
        z_pomdp = self.pomdp_proj(pomdp_raw)
        z_mamba = self.mamba_proj(mamba_raw)
        perspectives = torch.stack([z_gnn, z_pomdp, z_mamba], dim=2)
        z_fused = self.fusion(perspectives)
        l_fusion = self.fusion_expert(z_fused)

        # Router input: expert logits + fusion logits + feature magnitude signals
        # Feature magnitudes tell the router about input characteristics
        gnn_norm = gnn_raw.norm(dim=-1, keepdim=True)  # (B, N, 1)
        pomdp_norm = pomdp_raw.norm(dim=-1, keepdim=True)
        mamba_norm = mamba_raw.norm(dim=-1, keepdim=True)

        router_in = torch.cat([
            l_gnn, l_pomdp, l_mamba, l_fusion,  # 4+4+4+4 = 16
            gnn_norm, pomdp_norm, mamba_norm,    # 3
        ], dim=-1)  # (B, N, 19)

        route_logits = self.router(router_in)
        route_weights = F.softmax(route_logits / self.router_temp.clamp(min=0.1), dim=-1)

        # Combine
        all_logits = torch.stack([l_gnn, l_pomdp, l_mamba, l_fusion], dim=2)
        final = (route_weights.unsqueeze(-1) * all_logits).sum(dim=2)

        return {
            "logits": final,
            "route_weights": route_weights,
            "l_gnn": l_gnn, "l_pomdp": l_pomdp,
            "l_mamba": l_mamba, "l_fusion": l_fusion,
        }


def train_stage1(model, train_data, val_data, device, epochs=50, batch_size=128):
    """Train expert heads independently."""
    state_counts = torch.bincount(train_data["states"].flatten().long(), minlength=4).float().clamp(min=1)
    state_weights = (train_data["states"].numel() / (4 * state_counts)).clamp(max=8.0).to(device)

    experts = [
        ("GNN", model.gnn_expert, "gnn"),
        ("POMDP", model.pomdp_expert, "pomdp"),
        ("Mamba", model.mamba_expert, "mamba"),
    ]

    n_train = train_data["gnn"].size(0)

    for name, expert, key in experts:
        optimizer = torch.optim.AdamW(expert.parameters(), lr=0.001, weight_decay=1e-3)
        best_acc = 0.0
        best_state = None

        for epoch in range(1, epochs + 1):
            expert.train()
            perm = torch.randperm(n_train)
            for i in range(0, n_train, batch_size):
                idx = perm[i:i+batch_size]
                x = train_data[key][idx].to(device)
                s = train_data["states"][idx].to(device).long()
                mask = train_data["mask"][idx].to(device)
                logits = expert(x)
                loss = F.cross_entropy(logits.reshape(-1, 4), s.reshape(-1), weight=state_weights, reduction="none")
                loss = (loss * mask.reshape(-1)).sum() / mask.sum()
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            if epoch % 10 == 0 or epoch == epochs:
                expert.eval()
                correct = total = 0
                with torch.no_grad():
                    for i in range(0, val_data[key].size(0), batch_size):
                        x = val_data[key][i:i+batch_size].to(device)
                        s = val_data["states"][i:i+batch_size].to(device).long()
                        mask = val_data["mask"][i:i+batch_size].to(device)
                        preds = expert(x).argmax(dim=-1)
                        valid = mask > 0
                        correct += ((preds == s) & valid).sum().item()
                        total += valid.sum().item()
                acc = correct / max(total, 1)
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.clone() for k, v in expert.state_dict().items()}

        expert.load_state_dict(best_state)
        print(f"    {C_TEXT}{name}: acc={best_acc:.4f}{C_RESET}")

    # Freeze all experts
    for _, expert, _ in experts:
        for param in expert.parameters():
            param.requires_grad = False


def train_stage2(model, train_data, val_data, device, epochs=80, batch_size=128):
    """Train fusion + router with frozen experts."""

    # Only optimize non-frozen parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  {C_DIM}Trainable params: {n_trainable:,} (experts frozen){C_RESET}")

    state_counts = torch.bincount(train_data["states"].flatten().long(), minlength=4).float().clamp(min=1)
    state_weights = (train_data["states"].numel() / (4 * state_counts)).clamp(max=8.0).to(device)

    optimizer = torch.optim.AdamW(trainable, lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    n_train = train_data["gnn"].size(0)
    best_macro = 0.0
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            g = train_data["gnn"][idx].to(device)
            p = train_data["pomdp"][idx].to(device)
            m = train_data["mamba"][idx].to(device)
            s = train_data["states"][idx].to(device).long()
            mask = train_data["mask"][idx].to(device)

            out = model(g, p, m)

            # Main loss: routed output
            main_loss = F.cross_entropy(
                out["logits"].reshape(-1, 4), s.reshape(-1),
                weight=state_weights, reduction="none"
            )
            main_loss = (main_loss * mask.reshape(-1)).sum() / mask.sum()

            # Fusion expert auxiliary (so fusion head learns independently too)
            fusion_loss = F.cross_entropy(
                out["l_fusion"].reshape(-1, 4), s.reshape(-1),
                weight=state_weights, reduction="none"
            )
            fusion_loss = (fusion_loss * mask.reshape(-1)).sum() / mask.sum()

            # Load balance — but allow specialization (softer target)
            route_avg = (out["route_weights"] * mask.unsqueeze(-1)).sum(dim=(0, 1))
            route_avg = route_avg / mask.sum()
            # Don't force perfect balance — allow 15-35% per expert
            balance_loss = F.relu(0.10 - route_avg).sum() + F.relu(route_avg - 0.40).sum()

            loss = main_loss + 0.3 * fusion_loss + 0.3 * balance_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

        scheduler.step()

        # Validate
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            class_tp = torch.zeros(4); class_fp = torch.zeros(4); class_fn = torch.zeros(4)
            route_sums = torch.zeros(4); route_n = 0

            with torch.no_grad():
                for i in range(0, val_data["gnn"].size(0), batch_size):
                    g = val_data["gnn"][i:i+batch_size].to(device)
                    p = val_data["pomdp"][i:i+batch_size].to(device)
                    m = val_data["mamba"][i:i+batch_size].to(device)
                    s = val_data["states"][i:i+batch_size].to(device).long()
                    mask = val_data["mask"][i:i+batch_size].to(device)

                    out = model(g, p, m)
                    preds = out["logits"].argmax(dim=-1)
                    valid = mask > 0

                    for c in range(4):
                        ct = (s == c) & valid; cp = (preds == c) & valid
                        class_tp[c] += (ct & cp).sum().item()
                        class_fp[c] += (~ct & cp).sum().item()
                        class_fn[c] += (ct & ~cp).sum().item()

                    route_sums += (out["route_weights"] * mask.unsqueeze(-1)).sum(dim=(0,1)).cpu()
                    route_n += mask.sum().item()

            class_f1 = []
            for c in range(4):
                pr = class_tp[c] / max(class_tp[c] + class_fp[c], 1)
                rc = class_tp[c] / max(class_tp[c] + class_fn[c], 1)
                class_f1.append(2 * pr * rc / max(pr + rc, 1e-8))
            macro = sum(class_f1) / 4
            rpct = route_sums / max(route_n, 1)

            improved = macro > best_macro
            if improved:
                best_macro = macro
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if epoch <= 10 or epoch % 10 == 0 or improved:
                marker = f"{C_SUCCESS}*{C_RESET}" if improved else " "
                f1s = " ".join(f"{f:.3f}" for f in class_f1)
                rts = " ".join(f"{r:.2f}" for r in rpct.tolist())
                print(f"  {C_TEXT}{epoch:4d}{C_RESET}  macro={macro:.4f}  [{f1s}]  route=[{rts}]  {marker}")

            if patience >= 8:
                print(f"  {C_DIM}Early stopping at epoch {epoch}{C_RESET}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_macro


def run(device="cuda", n_samples=10000):
    print(f"\n{C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║   Staged Fusion: Freeze Experts, Train Fusion     ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════╝{C_RESET}\n")

    print(f"  {C_INFO}Generating data...{C_RESET}")
    train_data, val_data = generate_fusion_data(count=n_samples, device=device)

    model = StagedFusionModel(mamba_dim=NODE_FEAT_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {C_TEXT}Total params: {C_BRIGHT}{total_params:,}{C_RESET}")

    # Stage 1: Train experts independently
    print(f"\n  {C_GOLD}{C_BOLD}  Stage 1: Training Expert Heads{C_RESET}")
    print(f"  {C_DIM}{'─' * 40}{C_RESET}")
    t0 = time.time()
    train_stage1(model, train_data, val_data, device)
    print(f"  {C_DIM}({time.time()-t0:.1f}s){C_RESET}")

    # Stage 2: Train fusion + router (experts frozen)
    print(f"\n  {C_GOLD}{C_BOLD}  Stage 2: Training Fusion + Router (experts frozen){C_RESET}")
    print(f"  {C_DIM}{'─' * 50}{C_RESET}")
    print(f"  {C_DIM}{'ep':>6s}  {'macro':>8s}  [hlthy  dgrad  faild  unrch]  route=[gnn  pomdp mamba fusn]{C_RESET}")
    print(f"  {C_DIM}{'─' * 80}{C_RESET}")
    t0 = time.time()
    best_macro = train_stage2(model, train_data, val_data, device)
    s2_time = time.time() - t0
    print(f"  {C_DIM}({s2_time:.1f}s){C_RESET}")

    # Final eval — compare against single-pillar baselines
    print(f"\n  {C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║              STAGED FUSION RESULTS                  ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════╝{C_RESET}")

    state_names = ["healthy", "degraded", "failed", "unreachable"]

    def eval_fn(logits_fn):
        tp = torch.zeros(4); fp = torch.zeros(4); fn = torch.zeros(4)
        with torch.no_grad():
            for i in range(0, val_data["states"].size(0), 128):
                s = val_data["states"][i:i+128].to(device).long()
                mask = val_data["mask"][i:i+128].to(device)
                logits = logits_fn(i, i+128)
                preds = logits.argmax(dim=-1)
                valid = mask > 0
                for c in range(4):
                    ct = (s==c) & valid; cp = (preds==c) & valid
                    tp[c] += (ct&cp).sum().item()
                    fp[c] += (~ct&cp).sum().item()
                    fn[c] += (ct&~cp).sum().item()
        r = {}
        for c in range(4):
            p_ = tp[c]/max(tp[c]+fp[c],1); r_ = tp[c]/max(tp[c]+fn[c],1)
            r[state_names[c]] = float(2*p_*r_/max(p_+r_,1e-8))
        r["macro"] = sum(r[s] for s in state_names)/4
        return r

    model.eval()

    # Individual expert results (from the frozen Stage 1 heads)
    gnn_r = eval_fn(lambda i,j: model.gnn_expert(val_data["gnn"][i:j].to(device)))
    pomdp_r = eval_fn(lambda i,j: model.pomdp_expert(val_data["pomdp"][i:j].to(device)))
    mamba_r = eval_fn(lambda i,j: model.mamba_expert(val_data["mamba"][i:j].to(device)))

    # Fusion-only results
    def fusion_only(i, j):
        g = val_data["gnn"][i:j].to(device)
        p = val_data["pomdp"][i:j].to(device)
        m = val_data["mamba"][i:j].to(device)
        return model(g, p, m)["l_fusion"]
    fusion_r = eval_fn(fusion_only)

    # Full routed results
    def routed(i, j):
        g = val_data["gnn"][i:j].to(device)
        p = val_data["pomdp"][i:j].to(device)
        m = val_data["mamba"][i:j].to(device)
        return model(g, p, m)["logits"]
    staged_r = eval_fn(routed)

    print(f"\n  {C_DIM}{'Model':<20s} {'Macro':>7s}", end="")
    for s in state_names: print(f" {s[:7]:>8s}", end="")
    print(f"{C_RESET}")
    print(f"  {C_DIM}{'─' * 60}{C_RESET}")

    for name, r in [("GNN expert", gnn_r), ("POMDP expert", pomdp_r),
                     ("Mamba expert", mamba_r), ("Fusion only", fusion_r),
                     ("STAGED ROUTED", staged_r)]:
        is_s = "STAGED" in name
        c = C_SUCCESS if is_s else C_INFO if "Fusion" in name else C_TEXT
        b = C_BOLD if is_s else ""
        line = f"  {c}{b}{name:<20s}{C_RESET} {r['macro']:7.4f}"
        for s in state_names: line += f" {r[s]:8.4f}"
        print(line)

    # Deltas vs best expert
    print(f"\n  {C_INFO}Staged routed vs best expert per class:{C_RESET}")
    wins = 0
    for s in state_names:
        best = max(gnn_r[s], pomdp_r[s], mamba_r[s])
        delta = staged_r[s] - best
        which = "GNN" if gnn_r[s]==best else "POMDP" if pomdp_r[s]==best else "Mamba"
        c = C_SUCCESS if delta > 0.01 else C_DANGER if delta < -0.01 else C_DIM
        if delta > 0.01: wins += 1
        print(f"    {c}{s:<15s} staged={staged_r[s]:.4f} best_expert={best:.4f} ({which}) {delta:+.4f}{C_RESET}")

    macro_best = max(gnn_r["macro"], pomdp_r["macro"], mamba_r["macro"])
    macro_delta = staged_r["macro"] - macro_best

    # Router analysis
    print(f"\n  {C_INFO}Router per-class behavior:{C_RESET}")
    expert_names = ["GNN", "POMDP", "Mamba", "Fusion"]
    with torch.no_grad():
        g = val_data["gnn"][:300].to(device)
        p = val_data["pomdp"][:300].to(device)
        m = val_data["mamba"][:300].to(device)
        s = val_data["states"][:300].to(device).long()
        mask = val_data["mask"][:300].to(device)
        out = model(g, p, m)
        for c in range(4):
            c_mask = (s==c) & (mask > 0)
            if c_mask.sum() == 0: continue
            w = out["route_weights"][c_mask.unsqueeze(-1).expand_as(out["route_weights"])].reshape(-1,4).mean(0)
            top = expert_names[w.argmax().item()]
            print(f"    {C_DIM}{state_names[c]:15s}{C_RESET} → {C_BRIGHT}{top}{C_RESET} [{' '.join(f'{x:.2f}' for x in w.tolist())}]")

    print(f"\n  {C_GOLD}{C_BOLD}  VERDICT{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")
    print(f"  {C_TEXT}Macro: staged={staged_r['macro']:.4f} vs best_expert={macro_best:.4f} ({macro_delta:+.4f}){C_RESET}")

    if macro_delta >= -0.01 and staged_r["unreachable"] > 0.05:
        print(f"\n  {C_SUCCESS}{C_BOLD}STAGED FUSION WORKS.{C_RESET}")
        print(f"  {C_SUCCESS}Expert accuracy preserved. Emergent unreachable: {staged_r['unreachable']:.4f}.{C_RESET}")
        print(f"  {C_SUCCESS}The architecture fuses without sacrificing.{C_RESET}")
    elif staged_r["unreachable"] > 0.05 and macro_delta >= -0.03:
        print(f"\n  {C_GOLD}{C_BOLD}CLOSE. Emergent signal + near-expert accuracy.{C_RESET}")
    else:
        print(f"\n  {C_DANGER}{C_BOLD}Accuracy still degraded with fusion.{C_RESET}")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    run(args.device, args.samples)
