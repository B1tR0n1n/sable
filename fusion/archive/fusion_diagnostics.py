#!/usr/bin/env python3
"""
Project PARALLAX — Fusion Diagnostics
========================================
Three diagnostic tests for the staged fusion model:

1. Confusion matrix for Failed class (why 0.151 ceiling?)
2. Calibration curves per class (does router confidence track accuracy?)
3. Distribution shift robustness (does router collapse under shift?)

Usage:
    python fusion_diagnostics.py --device cuda
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
    generate_fusion_data,
)
from staged_fusion import StagedFusionModel, train_stage1, train_stage2

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"

STATE_NAMES = ["healthy", "degraded", "failed", "unreachable"]


def train_model(device, n_samples=10000):
    """Train a staged fusion model and return it with val data."""
    print(f"  {C_INFO}Generating data and training model...{C_RESET}")
    train_data, val_data = generate_fusion_data(count=n_samples, device=device)

    model = StagedFusionModel(mamba_dim=NODE_FEAT_DIM).to(device)

    print(f"  {C_DIM}Stage 1: experts...{C_RESET}")
    train_stage1(model, train_data, val_data, device, epochs=50)

    print(f"  {C_DIM}Stage 2: fusion + router...{C_RESET}")
    train_stage2(model, train_data, val_data, device, epochs=60)

    return model, train_data, val_data


# ── Diagnostic 1: Confusion Matrix ───────────────────────────────────────


def confusion_matrix_analysis(model, val_data, device):
    """Full confusion matrix with focus on Failed class."""

    print(f"\n  {C_GOLD}{C_BOLD}  Diagnostic 1: Confusion Matrix{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")

    model.eval()
    # Full confusion matrix
    cm = torch.zeros(4, 4, dtype=torch.long)  # cm[true][pred]
    n_samples = val_data["gnn"].size(0)

    with torch.no_grad():
        for i in range(0, n_samples, 128):
            g = val_data["gnn"][i:i+128].to(device)
            p = val_data["pomdp"][i:i+128].to(device)
            m = val_data["mamba"][i:i+128].to(device)
            s = val_data["states"][i:i+128].long()
            mask = val_data["mask"][i:i+128]

            out = model(g, p, m)
            preds = out["logits"].cpu().argmax(dim=-1)

            valid = mask > 0
            for true_c in range(4):
                for pred_c in range(4):
                    cm[true_c][pred_c] += (
                        (s == true_c) & (preds == pred_c) & valid
                    ).sum().item()

    # Print confusion matrix
    print(f"\n  {C_DIM}{'':>15s}", end="")
    for s in STATE_NAMES:
        print(f" {s[:8]:>9s}", end="")
    print(f"  {'recall':>7s}{C_RESET}")

    print(f"  {C_DIM}{'':>15s}{'─' * 46}{C_RESET}")

    for true_c in range(4):
        row_total = cm[true_c].sum().item()
        recall = cm[true_c][true_c].item() / max(row_total, 1)
        line = f"  {C_TEXT}{STATE_NAMES[true_c]:>15s}{C_RESET}"
        for pred_c in range(4):
            val = cm[true_c][pred_c].item()
            if true_c == pred_c:
                line += f" {C_SUCCESS}{val:9d}{C_RESET}"
            elif val > row_total * 0.1:
                line += f" {C_DANGER}{val:9d}{C_RESET}"
            else:
                line += f" {C_DIM}{val:9d}{C_RESET}"
        line += f"  {recall:7.3f}"
        print(line)

    # Precision row
    print(f"  {C_DIM}{'':>15s}{'─' * 46}{C_RESET}")
    line = f"  {C_DIM}{'precision':>15s}{C_RESET}"
    for pred_c in range(4):
        col_total = cm[:, pred_c].sum().item()
        prec = cm[pred_c][pred_c].item() / max(col_total, 1)
        line += f" {prec:9.3f}"
    print(line)

    # Focus on Failed class
    print(f"\n  {C_INFO}Failed Class Breakdown:{C_RESET}")
    failed_total = cm[2].sum().item()
    if failed_total > 0:
        for pred_c in range(4):
            count = cm[2][pred_c].item()
            pct = count / failed_total * 100
            bar = "█" * int(pct / 2)
            is_correct = pred_c == 2
            c = C_SUCCESS if is_correct else C_DANGER if pct > 10 else C_DIM
            print(f"    {c}True FAILED predicted as {STATE_NAMES[pred_c]:12s}: "
                  f"{count:5d} ({pct:5.1f}%) {bar}{C_RESET}")

    print(f"\n  {C_TEXT}Total failed nodes in val set: {C_BRIGHT}{failed_total}{C_RESET}")
    if failed_total < 100:
        print(f"  {C_DANGER}Low sample count may limit F1 ceiling.{C_RESET}")

    return cm


# ── Diagnostic 2: Calibration Curves ─────────────────────────────────────


def calibration_analysis(model, val_data, device):
    """Check if router confidence tracks actual accuracy per class."""

    print(f"\n  {C_GOLD}{C_BOLD}  Diagnostic 2: Calibration Analysis{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")

    model.eval()
    # Collect per-node: predicted class, confidence, true class, route weights
    all_probs = []
    all_true = []
    all_routes = []

    with torch.no_grad():
        for i in range(0, val_data["gnn"].size(0), 128):
            g = val_data["gnn"][i:i+128].to(device)
            p = val_data["pomdp"][i:i+128].to(device)
            m = val_data["mamba"][i:i+128].to(device)
            s = val_data["states"][i:i+128].long()
            mask = val_data["mask"][i:i+128]

            out = model(g, p, m)
            probs = F.softmax(out["logits"], dim=-1).cpu()
            routes = out["route_weights"].cpu()

            valid = mask > 0
            for b in range(g.size(0)):
                for n in range(g.size(1)):
                    if valid[b, n]:
                        all_probs.append(probs[b, n].numpy())
                        all_true.append(int(s[b, n].item()))
                        all_routes.append(routes[b, n].numpy())

    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    all_routes = np.array(all_routes)

    # Per-class calibration: bin by predicted probability, check actual accuracy
    n_bins = 10
    print(f"\n  {C_INFO}Calibration per class (predicted prob vs actual accuracy):{C_RESET}")

    for c in range(4):
        probs_c = all_probs[:, c]
        true_c = (all_true == c).astype(float)

        bins = np.linspace(0, 1, n_bins + 1)
        print(f"\n    {C_TEXT}{STATE_NAMES[c]}:{C_RESET}")
        print(f"    {C_DIM}{'bin':>10s} {'pred_p':>7s} {'actual':>7s} {'count':>6s} {'gap':>7s}{C_RESET}")

        for b in range(n_bins):
            mask_bin = (probs_c >= bins[b]) & (probs_c < bins[b + 1])
            count = mask_bin.sum()
            if count < 5:
                continue
            avg_pred = probs_c[mask_bin].mean()
            avg_actual = true_c[mask_bin].mean()
            gap = abs(avg_pred - avg_actual)
            gc = C_SUCCESS if gap < 0.05 else C_GOLD if gap < 0.15 else C_DANGER
            print(f"    {gc}{bins[b]:.1f}-{bins[b+1]:.1f}"
                  f"  {avg_pred:7.3f} {avg_actual:7.3f} {count:6d} {gap:7.3f}{C_RESET}")

    # Router confidence vs accuracy
    print(f"\n  {C_INFO}Router confidence analysis:{C_RESET}")
    expert_names = ["GNN", "POMDP", "Mamba", "Fusion"]

    # Dominant expert accuracy
    dominant = all_routes.argmax(axis=1)
    dominant_conf = all_routes.max(axis=1)
    preds = all_probs.argmax(axis=1)
    correct = (preds == all_true)

    # Bin by dominant expert confidence
    conf_bins = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 1.01]
    print(f"    {C_DIM}{'conf_range':>12s} {'accuracy':>8s} {'count':>6s} {'dominant_expert':>15s}{C_RESET}")
    for i in range(len(conf_bins) - 1):
        mask_bin = (dominant_conf >= conf_bins[i]) & (dominant_conf < conf_bins[i + 1])
        count = mask_bin.sum()
        if count < 10:
            continue
        acc = correct[mask_bin].mean()
        # Which expert dominates in this confidence range?
        dom_counts = np.bincount(dominant[mask_bin], minlength=4)
        top_expert = expert_names[dom_counts.argmax()]
        gc = C_SUCCESS if acc > 0.7 else C_GOLD if acc > 0.5 else C_DANGER
        print(f"    {gc}{conf_bins[i]:.2f}-{conf_bins[i+1]:.2f}"
              f"  {acc:8.3f} {count:6d} {top_expert:>15s}{C_RESET}")


# ── Diagnostic 3: Distribution Shift ─────────────────────────────────────


def distribution_shift_analysis(model, val_data, device):
    """Test router behavior under distribution shift."""

    print(f"\n  {C_GOLD}{C_BOLD}  Diagnostic 3: Distribution Shift Robustness{C_RESET}")
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")

    model.eval()

    def eval_shifted(shift_fn, name):
        """Evaluate model on shifted data."""
        tp = torch.zeros(4); fp = torch.zeros(4); fn = torch.zeros(4)
        route_sums = torch.zeros(4); route_n = 0

        with torch.no_grad():
            for i in range(0, val_data["gnn"].size(0), 128):
                g, p, m = shift_fn(
                    val_data["gnn"][i:i+128].clone(),
                    val_data["pomdp"][i:i+128].clone(),
                    val_data["mamba"][i:i+128].clone(),
                )
                g, p, m = g.to(device), p.to(device), m.to(device)
                s = val_data["states"][i:i+128].to(device).long()
                mask = val_data["mask"][i:i+128].to(device)

                out = model(g, p, m)
                preds = out["logits"].argmax(dim=-1)
                valid = mask > 0

                for c in range(4):
                    ct = (s == c) & valid; cp = (preds == c) & valid
                    tp[c] += (ct & cp).sum().item()
                    fp[c] += (~ct & cp).sum().item()
                    fn[c] += (ct & ~cp).sum().item()

                route_sums += (out["route_weights"] * mask.unsqueeze(-1)).sum(dim=(0,1)).cpu()
                route_n += mask.sum().item()

        class_f1 = []
        for c in range(4):
            pr = tp[c] / max(tp[c] + fp[c], 1)
            rc = tp[c] / max(tp[c] + fn[c], 1)
            class_f1.append(2 * pr * rc / max(pr + rc, 1e-8))
        macro = sum(class_f1) / 4
        rpct = route_sums / max(route_n, 1)
        return macro, class_f1, rpct

    # Baseline: no shift
    base_macro, base_f1, base_route = eval_shifted(
        lambda g, p, m: (g, p, m), "No shift"
    )

    shifts = []

    # Shift 1: GNN features corrupted (simulates GNN failure / domain shift)
    def corrupt_gnn(g, p, m):
        noise = torch.randn_like(g) * 0.5
        return g + noise, p, m
    s1_macro, s1_f1, s1_route = eval_shifted(corrupt_gnn, "GNN corrupted")
    shifts.append(("GNN corrupted (noise)", s1_macro, s1_f1, s1_route))

    # Shift 2: GNN completely zeroed (pillar offline)
    def zero_gnn(g, p, m):
        return torch.zeros_like(g), p, m
    s2_macro, s2_f1, s2_route = eval_shifted(zero_gnn, "GNN offline")
    shifts.append(("GNN offline (zeroed)", s2_macro, s2_f1, s2_route))

    # Shift 3: POMDP beliefs all uniform (no observation info)
    def uniform_pomdp(g, p, m):
        p_new = p.clone()
        p_new[:, :, :4] = 0.25  # Uniform state beliefs
        p_new[:, :, 4] = 0.0    # Zero confidence
        return g, p_new, m
    s3_macro, s3_f1, s3_route = eval_shifted(uniform_pomdp, "POMDP uninformative")
    shifts.append(("POMDP uninformative", s3_macro, s3_f1, s3_route))

    # Shift 4: Mamba features corrupted
    def corrupt_mamba(g, p, m):
        noise = torch.randn_like(m) * 0.3
        return g, p, m + noise
    s4_macro, s4_f1, s4_route = eval_shifted(corrupt_mamba, "Mamba corrupted")
    shifts.append(("Mamba corrupted (noise)", s4_macro, s4_f1, s4_route))

    # Shift 5: All features scaled (global distribution shift)
    def scale_all(g, p, m):
        return g * 1.5, p * 1.5, m * 1.5
    s5_macro, s5_f1, s5_route = eval_shifted(scale_all, "All scaled 1.5x")
    shifts.append(("All scaled 1.5x", s5_macro, s5_f1, s5_route))

    # Shift 6: Two pillars offline (only Mamba)
    def only_mamba(g, p, m):
        return torch.zeros_like(g), torch.zeros_like(p), m
    s6_macro, s6_f1, s6_route = eval_shifted(only_mamba, "Only Mamba")
    shifts.append(("Only Mamba alive", s6_macro, s6_f1, s6_route))

    # Display results
    expert_names = ["GNN", "POMDP", "Mamba", "Fusion"]

    print(f"\n  {C_DIM}{'Condition':<25s} {'Macro':>7s} {'Δ':>7s}  route=[gnn  pomdp mamba fusn]{C_RESET}")
    print(f"  {C_DIM}{'─' * 75}{C_RESET}")

    # Baseline
    rt_str = " ".join(f"{r:.2f}" for r in base_route.tolist())
    print(f"  {C_TEXT}{'Baseline (no shift)':<25s} {base_macro:7.4f} {'':>7s}  [{rt_str}]{C_RESET}")

    for name, macro, f1, route in shifts:
        delta = macro - base_macro
        rt_str = " ".join(f"{r:.2f}" for r in route.tolist())

        # Check if router shifted away from the corrupted pillar
        gc = C_SUCCESS if delta > -0.05 else C_GOLD if delta > -0.15 else C_DANGER
        print(f"  {gc}{name:<25s} {macro:7.4f} {delta:+7.4f}  [{rt_str}]{C_RESET}")

    # Analysis
    print(f"\n  {C_INFO}Interpretation:{C_RESET}")

    # Check: did the router reduce weight on corrupted pillars?
    gnn_offline_route = shifts[1][3]  # GNN zeroed
    gnn_route_drop = base_route[0] - gnn_offline_route[0]
    print(f"    {C_TEXT}GNN offline → GNN route weight: {base_route[0]:.3f} → {gnn_offline_route[0]:.3f} "
          f"(Δ={-gnn_route_drop:.3f}){C_RESET}")
    if gnn_route_drop > 0.02:
        print(f"    {C_SUCCESS}Router reduces GNN weight when GNN is offline. Graceful.{C_RESET}")
    else:
        print(f"    {C_DANGER}Router doesn't adapt to GNN failure.{C_RESET}")

    # Check: does router collapse to one expert under severe shift?
    only_mamba_route = shifts[5][3]
    max_weight = only_mamba_route.max().item()
    dominant = expert_names[only_mamba_route.argmax().item()]
    print(f"\n    {C_TEXT}Only Mamba alive → dominant: {dominant} ({max_weight:.3f}){C_RESET}")
    if max_weight > 0.6:
        print(f"    {C_SUCCESS}Router correctly concentrates on surviving pillar.{C_RESET}")
    elif max_weight > 0.4:
        print(f"    {C_GOLD}Partial concentration. Router hedges even with only one source.{C_RESET}")
    else:
        print(f"    {C_DANGER}Router doesn't concentrate. May not handle pillar failure well.{C_RESET}")

    # Overall graceful degradation assessment
    worst_delta = min(s[1] - base_macro for s in shifts)
    avg_delta = np.mean([s[1] - base_macro for s in shifts])
    print(f"\n    {C_TEXT}Worst degradation: {worst_delta:+.4f}{C_RESET}")
    print(f"    {C_TEXT}Average degradation: {avg_delta:+.4f}{C_RESET}")
    if worst_delta > -0.15:
        print(f"    {C_SUCCESS}Router degrades gracefully under all shift conditions.{C_RESET}")
    elif worst_delta > -0.30:
        print(f"    {C_GOLD}Moderate degradation under severe shifts.{C_RESET}")
    else:
        print(f"    {C_DANGER}Severe degradation under shift. Router is fragile.{C_RESET}")


# ── Main ──────────────────────────────────────────────────────────────────


def run(device="cuda"):
    print(f"\n{C_GOLD}{C_BOLD}  ╔══════════════════════════════════════════════════╗{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ║   Fusion Diagnostics: Three Deep Tests             ║{C_RESET}")
    print(f"  {C_GOLD}{C_BOLD}  ╚══════════════════════════════════════════════════╝{C_RESET}\n")

    model, train_data, val_data = train_model(device)

    confusion_matrix_analysis(model, val_data, device)
    calibration_analysis(model, val_data, device)
    distribution_shift_analysis(model, val_data, device)

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.device)
