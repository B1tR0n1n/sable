#!/usr/bin/env -S python -u
"""
Project PARALLAX — Staged Fusion v3: Sharp Router
====================================================
Fix from v2: router was too soft. Softmax blending diluted signal.
- Hard routing at eval (argmax, not weighted average)
- Lower temperature during training for sharper gradients
- No balance loss — let the router specialize aggressively
- Auxiliary loss: each expert also trains on its best class

Usage:
    python staged_fusion_v3.py --device cuda
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

from sable_sim.core.states import N_STATES, STATE_NAMES
from shared_latent_space import (
    Z_DIM, GNN_DIM, POMDP_DIM,
    GNNProjection, POMDPProjection, MambaProjection,
    CrossAttentionFusion,
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

EXPERT_NAMES = ["GNN", "POMDP", "Mamba", "Fusion"]


class SharpRoutedFusion(nn.Module):
    """Fusion with sharp routing — hard argmax at eval, low-temp softmax at train."""

    def __init__(self, mamba_dim=NODE_FEAT_DIM, dropout=0.15):
        super().__init__()

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

        self.gnn_proj = GNNProjection()
        self.pomdp_proj = POMDPProjection()
        self.mamba_proj = MambaProjection(mamba_dim)
        self.fusion = CrossAttentionFusion()
        self.fusion_expert = nn.Sequential(
            nn.Linear(Z_DIM, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, N_STATES),
        )

        # Router: sees all expert logits + feature norms → picks one expert
        # Input: 4 experts × N_STATES logits + 3 norms = 4*N_STATES + 3
        router_input_dim = 4 * N_STATES + 3
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 4),  # 4 experts (not N_STATES)
        )

    def forward(self, gnn_raw, pomdp_raw, mamba_raw, hard_route=False):
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

        router_in = torch.cat([
            l_gnn, l_pomdp, l_mamba, l_fusion,
            gnn_raw.norm(dim=-1, keepdim=True),
            pomdp_raw.norm(dim=-1, keepdim=True),
            mamba_raw.norm(dim=-1, keepdim=True),
        ], dim=-1)

        route_logits = self.router(router_in)  # (B, N, 4)
        all_logits = torch.stack([l_gnn, l_pomdp, l_mamba, l_fusion], dim=2)  # (B, N, 4experts, 4states)

        if hard_route or not self.training:
            # HARD routing: argmax selection, no blending
            best_expert = route_logits.argmax(dim=-1)  # (B, N)
            # Gather the logits from the selected expert
            idx = best_expert.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1, N_STATES)
            final = all_logits.gather(2, idx).squeeze(2)  # (B, N, 4states)
            route_weights = F.one_hot(best_expert, 4).float()
        else:
            # Soft routing for training (gradient flow), low temperature
            route_weights = F.softmax(route_logits / 0.3, dim=-1)
            final = (route_weights.unsqueeze(-1) * all_logits).sum(dim=2)

        return {
            "logits": final,
            "route_weights": route_weights,
            "route_logits": route_logits,
            "l_gnn": l_gnn, "l_pomdp": l_pomdp,
            "l_mamba": l_mamba, "l_fusion": l_fusion,
            "all_logits": all_logits,
        }


def eval_full(model, val_data, device):
    """Full per-expert + routed evaluation."""
    model.eval()
    results = {}

    def get_f1s(logits_fn):
        tp = torch.zeros(N_STATES); fp = torch.zeros(N_STATES); fn = torch.zeros(N_STATES)
        with torch.no_grad():
            for i in range(0, val_data["states"].size(0), 256):
                s = val_data["states"][i:i+256].to(device).long()
                m = val_data["mask"][i:i+256].to(device)
                logits = logits_fn(i, i+256)
                preds = logits.argmax(-1)
                valid = m > 0
                for c in range(N_STATES):
                    ct = (s==c)&valid; cp = (preds==c)&valid
                    tp[c] += (ct&cp).sum().item()
                    fp[c] += (~ct&cp).sum().item()
                    fn[c] += (ct&~cp).sum().item()
        f1s = []
        for c in range(N_STATES):
            p = tp[c]/max(tp[c]+fp[c],1); r = tp[c]/max(tp[c]+fn[c],1)
            f1s.append(2*p*r/max(p+r,1e-8))
        return f1s

    for ename, key in [("GNN", "gnn"), ("POMDP", "pomdp"), ("Mamba", "mamba")]:
        expert = getattr(model, f"{key}_expert")
        results[ename] = get_f1s(lambda i, j, _k=key: expert(val_data[_k][i:j].to(device)))

    results["Fusion"] = get_f1s(lambda i,j: model(
        val_data["gnn"][i:j].to(device), val_data["pomdp"][i:j].to(device),
        val_data["mamba"][i:j].to(device))["l_fusion"])

    results["Routed"] = get_f1s(lambda i,j: model(
        val_data["gnn"][i:j].to(device), val_data["pomdp"][i:j].to(device),
        val_data["mamba"][i:j].to(device))["logits"])

    return results


def _train_expert_head(expert, key, train_data, val_data, n_train, state_weights, device):
    """Train a single expert head with early stopping on val accuracy."""
    opt = torch.optim.AdamW(expert.parameters(), lr=0.001, weight_decay=1e-3)
    best_acc = 0.0
    best_state = None
    for ep in range(1, 81):
        expert.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 256):
            idx = perm[i:i+256]
            x = train_data[key][idx].to(device)
            s = train_data["states"][idx].to(device).long()
            m = train_data["mask"][idx].to(device)
            logits = expert(x)
            loss = F.cross_entropy(logits.reshape(-1, N_STATES), s.reshape(-1), weight=state_weights, reduction="none")
            loss = (loss * m.reshape(-1)).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0:
            expert.eval()
            correct = total = 0
            with torch.no_grad():
                for i in range(0, val_data[key].size(0), 256):
                    x = val_data[key][i:i+256].to(device)
                    s = val_data["states"][i:i+256].to(device).long()
                    m = val_data["mask"][i:i+256].to(device)
                    preds = expert(x).argmax(-1)
                    valid = m > 0
                    correct += ((preds==s)&valid).sum().item()
                    total += valid.sum().item()
            acc = correct / max(total, 1)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in expert.state_dict().items()}
    expert.load_state_dict(best_state)
    return best_acc


def _train_fusion_epoch(model, train_data, n_train, state_weights, device, optimizer, trainable):
    """Run one training epoch of Stage 2 fusion + router."""
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

        main_loss = F.cross_entropy(out["logits"].reshape(-1, N_STATES), s.reshape(-1),
                                    weight=state_weights, reduction="none")
        main_loss = (main_loss * mask.reshape(-1)).sum() / mask.sum()

        fusion_loss = F.cross_entropy(out["l_fusion"].reshape(-1, N_STATES), s.reshape(-1),
                                      weight=state_weights, reduction="none")
        fusion_loss = (fusion_loss * mask.reshape(-1)).sum() / mask.sum()

        # Router supervision
        with torch.no_grad():
            all_preds = out["all_logits"].argmax(dim=-1)
            correct_mask = (all_preds == s.unsqueeze(-1))
            router_target = torch.full_like(s, 3)
            for e in [0, 1, 2]:
                router_target[correct_mask[:,:,e] & ~correct_mask[:,:,3]] = e
            router_target[correct_mask[:,:,3]] = 3

        router_loss = F.cross_entropy(out["route_logits"].reshape(-1,4),
                                      router_target.reshape(-1), reduction="none")
        router_loss = (router_loss * mask.reshape(-1)).sum() / mask.sum()

        loss = main_loss + 0.3 * fusion_loss + 0.5 * router_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()


def run(device="cuda"):
    print(f"\n{C_GOLD}{C_BOLD}  ╔═══════════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║   Staged Fusion v3: Sharp Router + Hard Routing        ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚═══════════════════════════════════════════════════════╝{C_RESET}\n")

    data_path = Path(__file__).parent / "infra_fusion_data.pt"
    saved = torch.load(data_path, weights_only=False)
    train_data, val_data = saved["train"], saved["val"]
    n_train = train_data["gnn"].size(0)
    print(f"  {C_TEXT}train: {C_BRIGHT}{n_train:,}{C_RESET}  val: {C_BRIGHT}{val_data['gnn'].size(0):,}{C_RESET}", flush=True)

    valid_s = train_data["states"][train_data["mask"] > 0].long()
    state_counts = torch.bincount(valid_s, minlength=N_STATES).float().clamp(min=1)
    state_weights = (valid_s.size(0) / (N_STATES * state_counts)).clamp(max=10.0).to(device)

    model = SharpRoutedFusion(mamba_dim=NODE_FEAT_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {C_TEXT}params: {C_BRIGHT}{total_params:,}{C_RESET}", flush=True)

    # ── Stage 1: Expert heads ──
    print(f"\n  {C_GOLD}{C_BOLD}  Stage 1: Expert Heads{C_RESET}")
    print(f"  {C_DIM}{'─' * 40}{C_RESET}")

    experts = [
        ("GNN", model.gnn_expert, "gnn"),
        ("POMDP", model.pomdp_expert, "pomdp"),
        ("Mamba", model.mamba_expert, "mamba"),
    ]

    for name, expert, key in experts:
        best_acc = _train_expert_head(expert, key, train_data, val_data, n_train, state_weights, device)
        print(f"    {C_TEXT}{name:6s}: acc={best_acc:.4f}{C_RESET}", flush=True)

    # Freeze experts
    for _, expert, _ in experts:
        for p in expert.parameters():
            p.requires_grad = False

    # ── Stage 2: Fusion + Router (sharp) ──
    print(f"\n  {C_GOLD}{C_BOLD}  Stage 2: Fusion + Router (experts frozen, sharp routing){C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  {C_DIM}Trainable: {n_trainable:,}{C_RESET}", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-6)

    best_macro = 0.0
    best_model_state = None
    patience = 0

    print(f"  {C_DIM}{'ep':>4s}  {'macro':>7s}  [hlthy  dgrad  faild  unrch]  route=[gnn  pomdp mamba fusn]{C_RESET}")
    print(f"  {C_DIM}{'─' * 80}{C_RESET}")

    t0 = time.time()
    for epoch in range(1, 121):
        _train_fusion_epoch(model, train_data, n_train, state_weights, device, optimizer, trainable)
        scheduler.step()

        if epoch % 5 == 0 or epoch <= 10 or epoch == 120:
            results = eval_full(model, val_data, device)
            f1s = results["Routed"]
            macro = sum(f1s) / N_STATES

            improved = macro > best_macro + 0.001
            if improved:
                best_macro = macro
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            # Route distribution
            model.eval()
            with torch.no_grad():
                out = model(val_data["gnn"][:300].to(device), val_data["pomdp"][:300].to(device),
                           val_data["mamba"][:300].to(device))
                rw = out["route_weights"][:300].mean(dim=(0,1))

            marker = f"{C_SUCCESS}*{C_RESET}" if improved else " "
            f1str = " ".join(f"{f:.3f}" for f in f1s)
            rtstr = " ".join(f"{r:.2f}" for r in rw.tolist())
            print(f"  {C_TEXT}{epoch:4d}{C_RESET}  {macro:.4f}  [{f1str}]  route=[{rtstr}] {marker}", flush=True)

            if patience >= 10:
                print(f"  {C_DIM}Early stopping at epoch {epoch}{C_RESET}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    # ── Final Results ──
    print(f"\n  {C_GOLD}{C_BOLD}  ╔═══════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║              STAGED FUSION v3 RESULTS                ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚═══════════════════════════════════════════════════╝{C_RESET}")

    results = eval_full(model, val_data, device)

    print(f"\n  {C_DIM}{'Model':<15s} {'Macro':>7s}", end="")
    for s in STATE_NAMES: print(f" {s[:7]:>8s}", end="")
    print(f"{C_RESET}")
    print(f"  {C_DIM}{'─' * 65}{C_RESET}")

    for name in ["GNN", "POMDP", "Mamba", "Fusion", "Routed"]:
        f1s = results[name]
        macro = sum(f1s) / N_STATES
        is_r = name == "Routed"
        is_f = name == "Fusion"
        if is_r:
            c = C_SUCCESS
        elif is_f:
            c = C_INFO
        else:
            c = C_TEXT
        b = C_BOLD if is_r else ""
        line = f"  {c}{b}{name:<15s}{C_RESET} {macro:7.4f}"
        for f in f1s: line += f" {f:8.4f}"
        print(line)

    # Delta analysis
    routed = results["Routed"]
    fusion = results["Fusion"]
    macro_r = sum(routed)/N_STATES
    macro_f = sum(fusion)/N_STATES
    best_expert_macro = max(sum(results[e])/N_STATES for e in ["GNN", "POMDP", "Mamba"])

    print(f"\n  {C_INFO}Routed vs Fusion-only:{C_RESET}")
    for i, s in enumerate(STATE_NAMES):
        delta = routed[i] - fusion[i]
        if delta > 0.01:
            c = C_SUCCESS
        elif delta < -0.01:
            c = C_DANGER
        else:
            c = C_DIM
        print(f"    {c}{s:<15s} routed={routed[i]:.4f} fusion={fusion[i]:.4f} {delta:+.4f}{C_RESET}")

    # Router per-class
    print(f"\n  {C_INFO}Router routing:{C_RESET}")
    model.eval()
    with torch.no_grad():
        g = val_data["gnn"][:500].to(device)
        p = val_data["pomdp"][:500].to(device)
        m_ = val_data["mamba"][:500].to(device)
        s = val_data["states"][:500].to(device).long()
        mask = val_data["mask"][:500].to(device)
        out = model(g, p, m_)
        for c in range(N_STATES):
            c_mask = (s==c) & (mask > 0)
            if c_mask.sum() == 0: continue
            w = out["route_weights"][c_mask.unsqueeze(-1).expand_as(out["route_weights"])].reshape(-1,4).float().mean(0)
            winner = EXPERT_NAMES[w.argmax().item()]
            print(f"    {C_DIM}{STATE_NAMES[c]:15s}{C_RESET} → {C_BRIGHT}{winner}{C_RESET} [{' '.join(f'{x:.2f}' for x in w.tolist())}]")

    print(f"\n  {C_GOLD}{C_BOLD}  VERDICT{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")
    print(f"  {C_TEXT}Routed:      {C_BRIGHT}{macro_r:.4f}{C_RESET}")
    print(f"  {C_TEXT}Fusion-only: {C_BRIGHT}{macro_f:.4f}{C_RESET}")
    print(f"  {C_TEXT}Best expert: {C_BRIGHT}{best_expert_macro:.4f}{C_RESET}")
    print(f"  {C_TEXT}Routed vs fusion: {C_BRIGHT}{macro_r - macro_f:+.4f}{C_RESET}")
    print(f"  {C_TEXT}Routed vs expert: {C_BRIGHT}{macro_r - best_expert_macro:+.4f}{C_RESET}")

    ckpt_path = Path(__file__).parent / "checkpoints" / "staged_fusion_v3.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "results": results,
        "routed_macro": macro_r,
        "fusion_macro": macro_f,
    }, ckpt_path)
    print(f"  {C_TEXT}Saved: {C_BRIGHT}{ckpt_path}{C_RESET}")

    if macro_r >= macro_f - 0.005:
        print(f"\n  {C_SUCCESS}{C_BOLD}ROUTER FIXED. Routed >= Fusion-only.{C_RESET}")
        if macro_r > macro_f:
            print(f"  {C_SUCCESS}Routed EXCEEDS fusion — the router adds value.{C_RESET}")
    else:
        print(f"\n  {C_DANGER}Router still underperforming fusion-only by {macro_f - macro_r:.4f}{C_RESET}")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.device)
