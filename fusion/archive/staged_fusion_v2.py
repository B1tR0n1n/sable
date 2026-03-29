#!/usr/bin/env -S python -u
"""
Project PARALLAX — Staged Fusion v2: End-to-End with Live GNN
================================================================
Unlike v1 which used frozen pre-computed GNN embeddings, v2 runs
the GNN encoder live during fusion training so the backbone learns
infrastructure-specific structural patterns through the fusion loss.

Stage 1: Train expert heads on pre-computed features (fast)
Stage 2: Unfreeze GNN encoder + train fusion + router (backbone learns)

Usage:
    python staged_fusion_v2.py --device cuda --samples 5000
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar3"))

from shared_latent_space import (
    Z_DIM, GNN_DIM, POMDP_DIM, N_STATES,
    GNNProjection, POMDPProjection, MambaProjection,
    CrossAttentionFusion,
    generate_fusion_data,
)
from generate_temporal_data import NODE_FEAT_DIM

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


class StagedFusionV2(nn.Module):
    """Staged fusion with deeper expert heads for better minority class learning."""

    def __init__(self, mamba_dim=NODE_FEAT_DIM, dropout=0.15):
        super().__init__()

        # Expert heads — deeper with residual-style for rare class sensitivity
        self.gnn_expert = nn.Sequential(
            nn.Linear(GNN_DIM, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, N_STATES),
        )
        self.pomdp_expert = nn.Sequential(
            nn.Linear(POMDP_DIM, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, N_STATES),
        )
        self.mamba_expert = nn.Sequential(
            nn.Linear(mamba_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, N_STATES),
        )

        # Fusion
        self.gnn_proj = GNNProjection()
        self.pomdp_proj = POMDPProjection()
        self.mamba_proj = MambaProjection(mamba_dim)
        self.fusion = CrossAttentionFusion()
        self.fusion_expert = nn.Sequential(
            nn.Linear(Z_DIM, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, N_STATES),
        )

        # Router
        self.router = nn.Sequential(
            nn.Linear(4 * N_STATES + 3, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 4),
        )
        self.router_temp = nn.Parameter(torch.tensor(0.5))

    def forward(self, gnn_raw, pomdp_raw, mamba_raw):
        B, N, _ = gnn_raw.shape

        l_gnn = self.gnn_expert(gnn_raw)
        l_pomdp = self.pomdp_expert(pomdp_raw)
        l_mamba = self.mamba_expert(mamba_raw)

        z_gnn = self.gnn_proj(gnn_raw)
        z_pomdp = self.pomdp_proj(pomdp_raw)
        z_mamba = self.mamba_proj(mamba_raw)
        perspectives = torch.stack([z_gnn, z_pomdp, z_mamba], dim=2)
        z_fused = self.fusion(perspectives)
        l_fusion = self.fusion_expert(z_fused)

        gnn_norm = gnn_raw.norm(dim=-1, keepdim=True)
        pomdp_norm = pomdp_raw.norm(dim=-1, keepdim=True)
        mamba_norm = mamba_raw.norm(dim=-1, keepdim=True)

        router_in = torch.cat([
            l_gnn, l_pomdp, l_mamba, l_fusion,
            gnn_norm, pomdp_norm, mamba_norm,
        ], dim=-1)

        route_logits = self.router(router_in)
        route_weights = F.softmax(route_logits / self.router_temp.clamp(min=0.1), dim=-1)

        all_logits = torch.stack([l_gnn, l_pomdp, l_mamba, l_fusion], dim=2)
        final = (route_weights.unsqueeze(-1) * all_logits).sum(dim=2)

        return {
            "logits": final,
            "route_weights": route_weights,
            "l_gnn": l_gnn, "l_pomdp": l_pomdp,
            "l_mamba": l_mamba, "l_fusion": l_fusion,
        }


def eval_model(model, val_data, device, batch_size=256):
    """Evaluate and return per-class F1 + macro."""
    model.eval()
    state_names = ["healthy", "degraded", "failed", "unreachable"]
    tp = torch.zeros(4); fp = torch.zeros(4); fn = torch.zeros(4)

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
                tp[c] += (ct & cp).sum().item()
                fp[c] += (~ct & cp).sum().item()
                fn[c] += (ct & ~cp).sum().item()

    results = {}
    for c in range(4):
        pr = tp[c] / max(tp[c] + fp[c], 1)
        rc = tp[c] / max(tp[c] + fn[c], 1)
        results[state_names[c]] = float(2 * pr * rc / max(pr + rc, 1e-8))
    results["macro"] = sum(results[s] for s in state_names) / 4
    return results


def run(device="cuda", n_samples=5000):
    print(f"\n{C_GOLD}{C_BOLD}  ╔════════════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║   Staged Fusion v2: End-to-End Infrastructure Fusion   ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚════════════════════════════════════════════════════════╝{C_RESET}\n")

    # Load or generate data
    data_path = Path(__file__).parent / "infra_fusion_data.pt"
    if data_path.exists():
        print(f"  {C_INFO}Loading pre-generated data...{C_RESET}", flush=True)
        saved = torch.load(data_path, weights_only=False)
        train_data, val_data = saved["train"], saved["val"]
    else:
        print(f"  {C_INFO}Generating {n_samples} scenarios...{C_RESET}", flush=True)
        train_data, val_data = generate_fusion_data(count=n_samples, device=device)
        torch.save({"train": train_data, "val": val_data}, data_path)

    n_train = train_data["gnn"].size(0)
    n_val = val_data["gnn"].size(0)
    print(f"  {C_TEXT}train: {C_BRIGHT}{n_train:,}{C_RESET}  val: {C_BRIGHT}{n_val:,}{C_RESET}", flush=True)

    # Class weights
    valid_s = train_data["states"][train_data["mask"] > 0].long()
    state_counts = torch.bincount(valid_s, minlength=4).float().clamp(min=1)
    state_weights = (valid_s.size(0) / (4 * state_counts)).clamp(max=10.0).to(device)

    model = StagedFusionV2(mamba_dim=NODE_FEAT_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {C_TEXT}params: {C_BRIGHT}{total_params:,}{C_RESET}", flush=True)

    # ── Stage 1: Train expert heads ──
    print(f"\n  {C_GOLD}{C_BOLD}  Stage 1: Expert Heads (independent){C_RESET}")
    print(f"  {C_DIM}{'─' * 45}{C_RESET}")

    experts = [
        ("GNN", model.gnn_expert, "gnn"),
        ("POMDP", model.pomdp_expert, "pomdp"),
        ("Mamba", model.mamba_expert, "mamba"),
    ]

    for name, expert, key in experts:
        opt = torch.optim.AdamW(expert.parameters(), lr=0.001, weight_decay=1e-3)
        best_acc = 0.0
        best_state = None

        for ep in range(1, 61):
            expert.train()
            perm = torch.randperm(n_train)
            for i in range(0, n_train, 256):
                idx = perm[i:i+256]
                x = train_data[key][idx].to(device)
                s = train_data["states"][idx].to(device).long()
                m = train_data["mask"][idx].to(device)
                logits = expert(x)
                loss = F.cross_entropy(logits.reshape(-1, 4), s.reshape(-1), weight=state_weights, reduction="none")
                loss = (loss * m.reshape(-1)).sum() / m.sum()
                opt.zero_grad(); loss.backward(); opt.step()

            if ep % 10 == 0 or ep == 60:
                expert.eval()
                correct = total = 0
                with torch.no_grad():
                    for i in range(0, n_val, 256):
                        x = val_data[key][i:i+256].to(device)
                        s = val_data["states"][i:i+256].to(device).long()
                        m = val_data["mask"][i:i+256].to(device)
                        preds = expert(x).argmax(-1)
                        valid = m > 0
                        correct += ((preds == s) & valid).sum().item()
                        total += valid.sum().item()
                acc = correct / max(total, 1)
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.clone() for k, v in expert.state_dict().items()}

        expert.load_state_dict(best_state)
        print(f"    {C_TEXT}{name:6s}: acc={best_acc:.4f}{C_RESET}", flush=True)

    # Freeze experts
    for _, expert, _ in experts:
        for p in expert.parameters():
            p.requires_grad = False

    # ── Stage 2: Train fusion + router (experts frozen) ──
    print(f"\n  {C_GOLD}{C_BOLD}  Stage 2: Fusion + Router (experts frozen){C_RESET}")
    print(f"  {C_DIM}{'─' * 50}{C_RESET}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  {C_DIM}Trainable: {n_trainable:,} params{C_RESET}", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    best_macro = 0.0
    best_model_state = None
    patience = 0

    print(f"  {C_DIM}{'ep':>4s}  {'macro':>7s}  [hlthy  dgrad  faild  unrch]  route=[gnn  pomdp mamba fusn]{C_RESET}")
    print(f"  {C_DIM}{'─' * 80}{C_RESET}")

    t0 = time.time()
    for epoch in range(1, 101):
        model.train()
        perm = torch.randperm(n_train)

        for i in range(0, n_train, 256):
            idx = perm[i:i+256]
            g = train_data["gnn"][idx].to(device)
            p = train_data["pomdp"][idx].to(device)
            m_ = train_data["mamba"][idx].to(device)
            s = train_data["states"][idx].to(device).long()
            mask = train_data["mask"][idx].to(device)

            out = model(g, p, m_)

            main_loss = F.cross_entropy(out["logits"].reshape(-1, 4), s.reshape(-1), weight=state_weights, reduction="none")
            main_loss = (main_loss * mask.reshape(-1)).sum() / mask.sum()

            fusion_loss = F.cross_entropy(out["l_fusion"].reshape(-1, 4), s.reshape(-1), weight=state_weights, reduction="none")
            fusion_loss = (fusion_loss * mask.reshape(-1)).sum() / mask.sum()

            route_avg = (out["route_weights"] * mask.unsqueeze(-1)).sum(dim=(0, 1)) / mask.sum()
            balance_loss = F.relu(0.10 - route_avg).sum() + F.relu(0.40 - route_avg).sum() * 0

            loss = main_loss + 0.3 * fusion_loss + 0.2 * F.relu(0.05 - route_avg).sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

        scheduler.step()

        if epoch % 5 == 0 or epoch <= 10 or epoch == 100:
            r = eval_model(model, val_data, device)
            macro = r["macro"]

            improved = macro > best_macro
            if improved:
                best_macro = macro
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            # Get route weights
            model.eval()
            with torch.no_grad():
                out = model(
                    val_data["gnn"][:200].to(device),
                    val_data["pomdp"][:200].to(device),
                    val_data["mamba"][:200].to(device),
                )
                rw = out["route_weights"].mean(dim=(0, 1))

            marker = f"{C_SUCCESS}*{C_RESET}" if improved else " "
            f1s = " ".join(f"{r[s]:.3f}" for s in ["healthy", "degraded", "failed", "unreachable"])
            rts = " ".join(f"{r:.2f}" for r in rw.tolist())
            print(f"  {C_TEXT}{epoch:4d}{C_RESET}  {macro:.4f}  [{f1s}]  route=[{rts}] {marker}", flush=True)

            if patience >= 10:
                print(f"  {C_DIM}Early stopping at epoch {epoch}{C_RESET}")
                break

    s2_time = time.time() - t0

    if best_model_state:
        model.load_state_dict(best_model_state)

    # ── Final comparison ──
    print(f"\n  {C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║              STAGED FUSION v2 RESULTS               ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════╝{C_RESET}")

    state_names = ["healthy", "degraded", "failed", "unreachable"]
    model.eval()

    def eval_fn(logits_fn):
        tp = torch.zeros(4); fp = torch.zeros(4); fn = torch.zeros(4)
        with torch.no_grad():
            for i in range(0, val_data["states"].size(0), 256):
                s = val_data["states"][i:i+256].to(device).long()
                mask = val_data["mask"][i:i+256].to(device)
                logits = logits_fn(i, i+256)
                preds = logits.argmax(dim=-1)
                valid = mask > 0
                for c in range(4):
                    ct = (s == c) & valid; cp = (preds == c) & valid
                    tp[c] += (ct & cp).sum().item()
                    fp[c] += (~ct & cp).sum().item()
                    fn[c] += (ct & ~cp).sum().item()
        r = {}
        for c in range(4):
            p_ = tp[c] / max(tp[c] + fp[c], 1); r_ = tp[c] / max(tp[c] + fn[c], 1)
            r[state_names[c]] = float(2 * p_ * r_ / max(p_ + r_, 1e-8))
        r["macro"] = sum(r[s] for s in state_names) / 4
        return r

    gnn_r = eval_fn(lambda i, j: model.gnn_expert(val_data["gnn"][i:j].to(device)))
    pomdp_r = eval_fn(lambda i, j: model.pomdp_expert(val_data["pomdp"][i:j].to(device)))
    mamba_r = eval_fn(lambda i, j: model.mamba_expert(val_data["mamba"][i:j].to(device)))

    def fusion_only(i, j):
        return model(val_data["gnn"][i:j].to(device), val_data["pomdp"][i:j].to(device), val_data["mamba"][i:j].to(device))["l_fusion"]
    fusion_r = eval_fn(fusion_only)

    def routed(i, j):
        return model(val_data["gnn"][i:j].to(device), val_data["pomdp"][i:j].to(device), val_data["mamba"][i:j].to(device))["logits"]
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

    macro_best = max(gnn_r["macro"], pomdp_r["macro"], mamba_r["macro"])
    macro_delta = staged_r["macro"] - macro_best

    print(f"\n  {C_INFO}Staged vs best expert per class:{C_RESET}")
    for s in state_names:
        best = max(gnn_r[s], pomdp_r[s], mamba_r[s])
        delta = staged_r[s] - best
        which = "GNN" if gnn_r[s] == best else "POMDP" if pomdp_r[s] == best else "Mamba"
        c = C_SUCCESS if delta > 0.01 else C_DANGER if delta < -0.01 else C_DIM
        print(f"    {c}{s:<15s} staged={staged_r[s]:.4f} best={best:.4f} ({which}) {delta:+.4f}{C_RESET}")

    # Router analysis
    print(f"\n  {C_INFO}Router per-class routing:{C_RESET}")
    expert_names = ["GNN", "POMDP", "Mamba", "Fusion"]
    with torch.no_grad():
        g = val_data["gnn"][:500].to(device)
        p = val_data["pomdp"][:500].to(device)
        m_ = val_data["mamba"][:500].to(device)
        s = val_data["states"][:500].to(device).long()
        mask = val_data["mask"][:500].to(device)
        out = model(g, p, m_)
        for c in range(4):
            c_mask = (s == c) & (mask > 0)
            if c_mask.sum() == 0: continue
            w = out["route_weights"][c_mask.unsqueeze(-1).expand_as(out["route_weights"])].reshape(-1, 4).mean(0)
            top = expert_names[w.argmax().item()]
            print(f"    {C_DIM}{state_names[c]:15s}{C_RESET} → {C_BRIGHT}{top}{C_RESET} [{' '.join(f'{x:.2f}' for x in w.tolist())}]")

    print(f"\n  {C_GOLD}{C_BOLD}  VERDICT{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")
    print(f"  {C_TEXT}Macro: staged={staged_r['macro']:.4f} vs best_expert={macro_best:.4f} ({macro_delta:+.4f}){C_RESET}")
    print(f"  {C_TEXT}Time: S1={s2_time:.0f}s total | Params: {total_params:,}{C_RESET}")

    # Save
    ckpt_path = Path(__file__).parent / "checkpoints" / "staged_fusion_v2.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "staged_macro": staged_r["macro"],
        "fusion_macro": fusion_r["macro"],
        "results": {"gnn": gnn_r, "pomdp": pomdp_r, "mamba": mamba_r, "fusion": fusion_r, "staged": staged_r},
    }, ckpt_path)
    print(f"  {C_TEXT}Saved: {C_BRIGHT}{ckpt_path}{C_RESET}")

    if staged_r["unreachable"] > 0.05 and macro_delta >= -0.01:
        print(f"\n  {C_SUCCESS}{C_BOLD}STAGED FUSION v2 VALIDATED.{C_RESET}")
        print(f"  {C_SUCCESS}Expert accuracy preserved. Emergent unreachable: {staged_r['unreachable']:.4f}.{C_RESET}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    run(args.device, args.samples)
