#!/usr/bin/env -S python -u
"""
SABLE LoRA Fine-Tuning
========================
Adapts the frozen fusion model to new data distributions using
Low-Rank Adaptation. The base model weights never change - LoRA
adds small trainable rank-decomposition matrices to the expert heads.

This is the "last mile" for customer deployment:
  1. Base model trained on sable_sim (general infrastructure patterns)
  2. LoRA adapter trained on customer data (specific environment)
  3. At inference: base + adapter merged, no latency overhead

Usage:
    python lora_finetune.py --data path/to/customer_scenarios.pt --rank 8
    python lora_finetune.py --data path/to/customer_scenarios.pt --rank 4 --epochs 30
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar3"))

from sable_sim.core.states import N_STATES, STATE_NAMES
from staged_fusion_v3 import SharpRoutedFusion
from temporal_chain import TemporalChainFusion, TemporalState
from generate_temporal_data import NODE_FEAT_DIM
from shared_latent_space import GNN_DIM, POMDP_DIM

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_DANGER = "\033[38;2;201;74;58m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


class LoRALinear(nn.Module):
    """LoRA adapter for a frozen linear layer.

    Adds low-rank matrices A and B such that:
        output = frozen_linear(x) + (x @ A @ B) * scale

    A is (in_features, rank), B is (rank, out_features).
    Only A and B are trainable. The original weight is frozen.
    """

    def __init__(self, frozen_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.frozen = frozen_linear
        self.rank = rank
        self.scale = alpha / rank

        in_f = frozen_linear.in_features
        out_f = frozen_linear.out_features

        # A: initialized with kaiming, B: initialized with zeros
        # This means LoRA starts as identity (zero contribution)
        self.enabled = True
        self.lora_A = nn.Parameter(torch.empty(in_f, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        nn.init.kaiming_uniform_(self.lora_A)

        # Freeze the original
        for p in self.frozen.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.frozen(x)
        if not self.enabled:
            return base
        lora = (x @ self.lora_A @ self.lora_B) * self.scale
        return base + lora

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into the base linear layer. Returns a new Linear."""
        merged = nn.Linear(self.frozen.in_features, self.frozen.out_features,
                           bias=self.frozen.bias is not None)
        with torch.no_grad():
            merged.weight.copy_(self.frozen.weight + (self.lora_B.T @ self.lora_A.T) * self.scale)
            if self.frozen.bias is not None:
                merged.bias.copy_(self.frozen.bias)
        return merged


def apply_lora(model, rank: int = 8, alpha: float = 16.0):
    """Wrap linear layers with LoRA adapters.

    Works on both SharpRoutedFusion (base) and TemporalChainFusion (full).
    Adapts expert heads, router, and temporal chain components.
    """
    lora_params = []

    def _wrap_sequential(seq):
        """Apply LoRA to all Linear layers in a Sequential."""
        for i, layer in enumerate(seq):
            if isinstance(layer, nn.Linear):
                lora = LoRALinear(layer, rank=rank, alpha=alpha)
                seq[i] = lora
                lora_params.extend([lora.lora_A, lora.lora_B])

    def _wrap_linear(module, attr):
        """Apply LoRA to a single Linear attribute."""
        layer = getattr(module, attr)
        if isinstance(layer, nn.Linear):
            lora = LoRALinear(layer, rank=rank, alpha=alpha)
            setattr(module, attr, lora)
            lora_params.extend([lora.lora_A, lora.lora_B])

    # Get the base fusion (might be wrapped in TemporalChainFusion)
    if hasattr(model, 'base_fusion'):
        base = model.base_fusion
    else:
        base = model

    # Expert heads + router
    for name in ["gnn_expert", "pomdp_expert", "mamba_expert", "fusion_expert"]:
        _wrap_sequential(getattr(base, name))
    _wrap_sequential(base.router)

    # Temporal chain components (if present)
    if hasattr(model, 'context_mixer'):
        mixer = model.context_mixer
        _wrap_sequential(mixer.traj_encoder)
        for gate_name in ['gnn_gate', 'pomdp_gate', 'mamba_gate']:
            gate = getattr(mixer, gate_name)
            _wrap_linear(gate, 'context_linear1')
            _wrap_linear(gate, 'context_linear2')
            _wrap_sequential(gate.gate)

    if hasattr(model, 'revision_gate'):
        rev = model.revision_gate
        _wrap_sequential(rev.revision_head)
        _wrap_sequential(rev.confidence_head)
        _wrap_sequential(rev.transition_head)

    return lora_params


def set_lora_enabled(model, enabled: bool):
    """Toggle all LoRA adapters on or off at runtime. Instant, no reload."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


def load_scenario_as_training_data(scenario_path: str | Path):
    """Load a precomputed scenario file and format for LoRA training."""
    s = torch.load(str(scenario_path), weights_only=False)

    gnn = s["gnn"].squeeze(0)       # (T, N, GNN_DIM)
    pomdp = s["pomdp"].squeeze(0)   # (T, N, POMDP_DIM)
    mamba = s["mamba"].squeeze(0)    # (T, N, NODE_FEAT_DIM)
    gt = s["ground_truth"]           # (T, N)

    return {
        "gnn": gnn,
        "pomdp": pomdp,
        "mamba": mamba,
        "states": gt,
        "n_nodes": s["n_nodes"],
        "n_ticks": s["n_ticks"],
        "name": s.get("name", "unknown"),
    }


def train_lora(
    model_path: str,
    data_paths: list[str],
    rank: int = 8,
    alpha: float = 16.0,
    lr: float = 1e-3,
    epochs: int = 50,
    device: str = "cuda",
    output_path: str = "checkpoints/lora_adapter.pt",
):
    """Fine-tune SABLE fusion model with LoRA on customer data.

    Loads the frozen base model, applies LoRA adapters, trains only
    the low-rank matrices on the provided scenario data.
    """
    print(f"\n{C_GOLD}{C_BOLD}  SABLE - LoRA Fine-Tuning{C_RESET}")
    print(f"  {C_DIM}{'=' * 40}{C_RESET}\n")

    # Load base fusion + temporal chain
    print(f"  {C_INFO}Loading base model...{C_RESET}")
    base_fusion = SharpRoutedFusion(mamba_dim=NODE_FEAT_DIM)
    fusion_path = Path(model_path)
    ckpt = torch.load(str(fusion_path), weights_only=False, map_location=device)
    if "model_state_dict" in ckpt:
        base_fusion.load_state_dict(ckpt["model_state_dict"])
    else:
        base_fusion.load_state_dict(ckpt)

    # Wrap with temporal chain if checkpoint exists
    temporal_path = fusion_path.parent / "temporal_chain.pt"
    if temporal_path.exists():
        model = TemporalChainFusion(base_fusion)
        tc_ckpt = torch.load(str(temporal_path), weights_only=False, map_location=device)
        model.load_state_dict(tc_ckpt["model_state_dict"], strict=False)
        print(f"  {C_TEXT}Temporal chain loaded - full model LoRA{C_RESET}")
    else:
        model = base_fusion
        print(f"  {C_TEXT}Fusion only (no temporal chain){C_RESET}")
    model.to(device)

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Apply LoRA to all components (fusion experts + router + temporal chain)
    print(f"  {C_INFO}Applying LoRA (rank={rank}, alpha={alpha})...{C_RESET}")
    lora_params = apply_lora(model, rank=rank, alpha=alpha)
    model.to(device)
    n_lora = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {C_TEXT}LoRA parameters: {C_BRIGHT}{n_lora:,}{C_RESET} / {n_total:,} total "
          f"({n_lora/n_total*100:.1f}%)")

    # Load training data
    print(f"  {C_INFO}Loading training data...{C_RESET}")
    all_gnn, all_pomdp, all_mamba, all_gt = [], [], [], []
    for dp in data_paths:
        d = load_scenario_as_training_data(dp)
        all_gnn.append(d["gnn"])
        all_pomdp.append(d["pomdp"])
        all_mamba.append(d["mamba"])
        all_gt.append(d["states"])
        print(f"    {C_DIM}{d['name']}: {d['n_ticks']}t x {d['n_nodes']}n{C_RESET}")

    # Concatenate along time dimension
    gnn = torch.cat(all_gnn, dim=0).to(device)    # (total_ticks, N, GNN_DIM)
    pomdp = torch.cat(all_pomdp, dim=0).to(device)
    mamba = torch.cat(all_mamba, dim=0).to(device)
    gt = torch.cat(all_gt, dim=0).to(device)        # (total_ticks, N)

    n_samples = gnn.shape[0]
    n_nodes = gnn.shape[1]
    print(f"  {C_TEXT}Total samples: {C_BRIGHT}{n_samples} ticks x {n_nodes} nodes{C_RESET}")

    # Class weights for imbalanced data
    class_counts = torch.zeros(N_STATES)
    for c in range(N_STATES):
        class_counts[c] = (gt == c).sum().float()
    class_weights = (1.0 / class_counts.clamp(min=1)).to(device)
    class_weights /= class_weights.sum()

    print(f"  {C_TEXT}Class distribution:{C_RESET}")
    for c in range(N_STATES):
        pct = class_counts[c] / gt.numel() * 100
        print(f"    {C_DIM}{STATE_NAMES[c]:12s}: {int(class_counts[c]):6d} ({pct:.1f}%){C_RESET}")

    # Keep scenarios separate for sequential temporal processing
    # Split scenarios: 75% train, 25% val
    n_scenarios = len(all_gnn)
    scenario_perm = torch.randperm(n_scenarios)
    n_train_sc = max(1, int(n_scenarios * 0.75))
    train_sc = scenario_perm[:n_train_sc].tolist()
    val_sc = scenario_perm[n_train_sc:].tolist()
    if not val_sc:
        val_sc = train_sc[-1:]  # At least 1 val scenario

    is_temporal = hasattr(model, 'context_mixer')

    # Optimizer - only LoRA params
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    print(f"  {C_TEXT}Training mode: {C_BRIGHT}{'temporal (sequential ticks)' if is_temporal else 'per-tick (shuffled)'}{C_RESET}")
    print(f"  {C_TEXT}Train scenarios: {C_BRIGHT}{len(train_sc)}{C_RESET}, Val: {C_BRIGHT}{len(val_sc)}{C_RESET}")

    # Training loop
    print(f"\n  {C_DIM}  ep    loss   v_acc  v_macro  hlthy  dgrad  faild  unrch    t{C_RESET}")
    print(f"  {C_DIM}{'=' * 70}{C_RESET}")

    best_macro = 0
    best_state = None
    patience = 0
    max_patience = 15

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0
        n_batches = 0

        # Shuffle scenario order each epoch
        np.random.shuffle(train_sc)

        for sc_idx in train_sc:
            sc_gnn = all_gnn[sc_idx].to(device)     # (T, N, GNN_DIM)
            sc_pomdp = all_pomdp[sc_idx].to(device)
            sc_mamba = all_mamba[sc_idx].to(device)
            sc_gt = all_gt[sc_idx].to(device)        # (T, N)
            T = sc_gnn.shape[0]
            N = sc_gnn.shape[1]

            if is_temporal:
                # Sequential: process ticks in order with temporal state
                temporal_state = TemporalState.cold_start(N, device=device)
                for t in range(T):
                    g = sc_gnn[t:t+1]   # (1, N, D)
                    p = sc_pomdp[t:t+1]
                    m = sc_mamba[t:t+1]
                    y = sc_gt[t]          # (N,)

                    out = model(g, p, m, temporal_state=temporal_state)
                    logits = out["revised_logits"]  # (1, N, N_STATES)
                    loss = loss_fn(logits.reshape(-1, N_STATES), y.reshape(-1))

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    optimizer.step()

                    # Update temporal state (detached - no backprop through time)
                    with torch.no_grad():
                        temporal_state.update(
                            out["revised_logits"].detach(),
                            out["confidence"].detach(),
                            out.get("z_fused", torch.zeros(1, N, 128, device=device)).detach(),
                        )

                    epoch_loss += loss.item()
                    n_batches += 1
            else:
                # Shuffled per-tick (no temporal context)
                perm = torch.randperm(T)
                for i in range(0, T, 8):
                    idx = perm[i:i+8]
                    g = sc_gnn[idx]
                    p = sc_pomdp[idx]
                    m = sc_mamba[idx]
                    y = sc_gt[idx]

                    out = model(g, p, m)
                    logits = out["logits"]
                    loss = loss_fn(logits.reshape(-1, N_STATES), y.reshape(-1))

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation - sequential for temporal, batched otherwise
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for sc_idx in val_sc:
                v_gnn = all_gnn[sc_idx].to(device)
                v_pomdp = all_pomdp[sc_idx].to(device)
                v_mamba = all_mamba[sc_idx].to(device)
                v_gt = all_gt[sc_idx].to(device)
                T = v_gnn.shape[0]
                N = v_gnn.shape[1]

                if is_temporal:
                    ts = TemporalState.cold_start(N, device=device)
                    for t in range(T):
                        out = model(v_gnn[t:t+1], v_pomdp[t:t+1], v_mamba[t:t+1], temporal_state=ts)
                        preds = out["revised_logits"].argmax(dim=-1).reshape(-1)
                        all_preds.append(preds)
                        all_labels.append(v_gt[t].reshape(-1))
                        ts.update(
                            out["revised_logits"].detach(),
                            out["confidence"].detach(),
                            out.get("z_fused", torch.zeros(1, N, 128, device=device)).detach(),
                        )
                else:
                    out = model(v_gnn, v_pomdp, v_mamba)
                    preds = out["logits"].argmax(dim=-1).reshape(-1)
                    all_preds.append(preds)
                    all_labels.append(v_gt.reshape(-1))

        v_preds = torch.cat(all_preds)
        v_y = torch.cat(all_labels)
        v_acc = (v_preds == v_y).float().mean().item()

        f1s = []
        for c in range(N_STATES):
            tp = ((v_preds == c) & (v_y == c)).sum().float()
            fp = ((v_preds == c) & (v_y != c)).sum().float()
            fn = ((v_preds != c) & (v_y == c)).sum().float()
            p_val = tp / max(tp + fp, 1)
            r_val = tp / max(tp + fn, 1)
            f1 = 2 * p_val * r_val / max(p_val + r_val, 1e-8)
            f1s.append(f1.item())
        macro_f1 = sum(f1s) / len(f1s)

        elapsed = time.time() - t0
        marker = ""
        if macro_f1 > best_macro:
            best_macro = macro_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()
                          if "lora_" in k}
            patience = 0
            marker = f" {C_SUCCESS}*{C_RESET}"
        else:
            patience += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or marker:
            print(f"  {C_TEXT}{epoch+1:4d}{C_RESET}  {avg_loss:.4f}  {v_acc:.4f}  "
                  f"{macro_f1:.4f}  {f1s[0]:.3f}  {f1s[1]:.3f}  {f1s[2]:.3f}  "
                  f"{f1s[3]:.3f}  {elapsed:.1f}s{marker}")

        if patience >= max_patience:
            print(f"\n  {C_DIM}Early stopping at epoch {epoch+1}{C_RESET}")
            break

    # Save LoRA adapter (just the low-rank matrices)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "lora_state_dict": best_state,
        "rank": rank,
        "alpha": alpha,
        "best_macro_f1": best_macro,
        "base_model": str(model_path),
    }, str(out_path))

    print(f"\n  {C_SUCCESS}{C_BOLD}Saved:{C_RESET} {C_TEXT}{out_path}{C_RESET}")
    print(f"  {C_TEXT}Best macro F1: {C_BRIGHT}{best_macro:.4f}{C_RESET}")
    print(f"  {C_TEXT}LoRA params:   {C_BRIGHT}{n_lora:,}{C_RESET}")
    print()

    return best_macro


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SABLE LoRA Fine-Tuning")
    parser.add_argument("--data", nargs="+", required=True, help="Scenario .pt files")
    parser.add_argument("--model", default=str(Path(__file__).parent / "checkpoints" / "staged_fusion_v3.pt"))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output", default=str(Path(__file__).parent / "checkpoints" / "lora_adapter.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_lora(
        model_path=args.model,
        data_paths=args.data,
        rank=args.rank,
        alpha=args.alpha,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        output_path=args.output,
    )
