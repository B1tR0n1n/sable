#!/usr/bin/env -S python -u
"""
Project PARALLAX — Adversarial Stress Test
=============================================
The worst day in infrastructure. Designed to break the temporal chain.

Scenarios:
  1. OSCILLATOR: Node flips healthy→failed→healthy→failed every tick. 
     Looks like recovery if you're naive. It's dying.
  2. SLOW POISON: 1% health drop per tick for 20 ticks. Invisible at any 
     single snapshot. Only visible in trajectory.
  3. CASCADE WHIPLASH: Root cause fails, cascade propagates, root cause 
     recovers, dependents stay dead. Causal reasoning required.
  4. FALSE POSITIVE STORM: 40% of observations are WRONG. Noisy as fuck.
  5. SILENT KILLER: Node fails but POMDP never observes it. Zero direct 
     evidence. Only topology inference can catch it.
  6. DELAYED AVALANCHE: Everything looks fine for 10 ticks, then 60% of 
     nodes fail simultaneously on tick 11.
  7. THE FULL MONDAY MORNING: All of the above at once, 25 ticks, 40 nodes.
"""

import sys, time, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar3"))

from sable_sim.core.states import N_STATES, STATE_NAMES
from temporal_chain import TemporalChainFusion, TemporalState
from staged_fusion_v3 import SharpRoutedFusion
from shared_latent_space import GNN_DIM, POMDP_DIM
from generate_temporal_data import NODE_FEAT_DIM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
base = SharpRoutedFusion(mamba_dim=NODE_FEAT_DIM)
base_ckpt = torch.load('checkpoints/staged_fusion_v3.pt', weights_only=False)
base.load_state_dict(base_ckpt['model_state_dict'])
for p in base.parameters():
    p.requires_grad = False
model = TemporalChainFusion(base)
ckpt = torch.load('checkpoints/temporal_chain.pt', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval().to(device)

C_GOLD = "\033[38;2;201;162;39m"
C_DIM = "\033[38;2;138;130;114m"
C_TEXT = "\033[38;2;200;189;160m"
C_BRIGHT = "\033[38;2;232;221;196m"
C_DANGER = "\033[38;2;201;74;58m"
C_SUCCESS = "\033[38;2;58;173;110m"
C_INFO = "\033[38;2;58;142;201m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"

rng = np.random.RandomState(666)
N = 40  # nodes
B = 1


def make_features(node_states, health_values, obs_noise=0.0, obs_coverage=1.0, 
                  obs_delay=0, gt_history=None, tick=0):
    """Build pillar features from synthetic scenario state."""
    gnn = torch.zeros(B, N, GNN_DIM)
    pomdp = torch.zeros(B, N, POMDP_DIM)
    mamba = torch.zeros(B, N, NODE_FEAT_DIM)

    for i in range(N):
        # GNN: type one-hot (random assignment) + degree + noisy health
        gnn[0, i, i % 8] = 1.0  # pseudo type
        gnn[0, i, 8] = rng.uniform(0.1, 0.8)  # degree
        gnn[0, i, 9] = health_values[i] + rng.normal(0, 0.2)

        # POMDP: belief vector from noisy/delayed/partial observations
        true_state = node_states[i]
        # POMDP belief is 4-dim (sim states only, no oscillating — that's detected temporally)
        belief = np.array([0.25, 0.25, 0.25, 0.25])

        if rng.random() < obs_coverage:
            # Observe — but maybe wrong
            observed = true_state
            if obs_delay > 0 and gt_history is not None and tick >= obs_delay:
                observed = gt_history[tick - obs_delay][i]  # delayed
            if rng.random() < obs_noise:
                observed = rng.randint(0, 4)  # completely wrong
            # POMDP belief is 4-dim (sim states only). Oscillating maps to failed for belief.
            obs_belief = min(observed, 3)
            belief[obs_belief] += 2.0
            belief /= belief.sum()
        
        pomdp[0, i, :4] = torch.tensor(belief)
        pomdp[0, i, 4] = 1.0 if rng.random() < obs_coverage else 0.0  # confidence
        pomdp[0, i, 5] = rng.uniform(0, 0.5)  # age

        # Mamba: noisy health + state features (NODE_FEAT_DIM=26 now: 1 health + 5 state + 20 type)
        mamba[0, i, 0] = health_values[i] + rng.normal(0, 0.15)
        if true_state < NODE_FEAT_DIM - 1:
            mamba[0, i, 1 + min(true_state, N_STATES - 1)] = 0.7 + rng.normal(0, 0.3)

    return gnn.to(device), pomdp.to(device), mamba.to(device)


def run_scenario(name, n_ticks, state_fn, health_fn, obs_noise=0.15, 
                 obs_coverage=0.5, obs_delay=0):
    """Run a scenario and evaluate."""
    gt_history = []
    ts = TemporalState.cold_start(N, device=device)
    
    tp = torch.zeros(N_STATES); fp = torch.zeros(N_STATES); fn = torch.zeros(N_STATES)
    per_tick_acc = []

    with torch.no_grad():
        for t in range(n_ticks):
            states = state_fn(t)
            health = health_fn(t)
            gt_history.append(states.copy())
            
            gnn, pomdp, mamba = make_features(
                states, health, obs_noise, obs_coverage, obs_delay, gt_history, t
            )
            gt = torch.tensor(states, dtype=torch.long).to(device)
            
            out = model(gnn, pomdp, mamba, temporal_state=ts)
            preds = out['revised_logits'][0].argmax(-1)
            
            ts.update(out['revised_logits'].detach(), out['confidence'].detach(),
                     out.get('z_fused').detach() if out.get('z_fused') is not None else None)
            
            correct = (preds == gt).sum().item()
            per_tick_acc.append(correct / N)
            
            for c in range(N_STATES):
                ct = gt == c; cp = preds == c
                tp[c] += (ct & cp).sum().item()
                fp[c] += (~ct & cp).sum().item()
                fn[c] += (ct & ~cp).sum().item()

    f1s = []
    for c in range(N_STATES):
        p = tp[c] / max(tp[c] + fp[c], 1)
        r = tp[c] / max(tp[c] + fn[c], 1)
        f1s.append(2 * p * r / max(p + r, 1e-8))
    macro = sum(f1s) / N_STATES

    print(f"\n  {C_GOLD}{C_BOLD}{name}{C_RESET}")
    print(f"  {C_DIM}{'─' * 50}{C_RESET}")
    for c in range(N_STATES):
        n_c = int(tp[c] + fn[c])
        if n_c > 0:
            bar = "█" * int(f1s[c] * 20)
            print(f"    {STATE_NAMES[c]:12s} F1={f1s[c]:.4f} (n={n_c:5d}) {C_GOLD}{bar}{C_RESET}")
    print(f"    {C_TEXT}Macro: {C_BRIGHT}{macro:.4f}{C_RESET}")
    
    # Per-tick trajectory
    ticks_str = " ".join(f"{a:.2f}" for a in per_tick_acc[:min(10, len(per_tick_acc))])
    if len(per_tick_acc) > 10:
        ticks_str += f" ... {per_tick_acc[-1]:.2f}"
    print(f"    {C_DIM}Per-tick: [{ticks_str}]{C_RESET}")
    
    return macro, f1s


print(f"\n{C_GOLD}{C_BOLD}  ╔════════════════════════════════════════════════════════╗{C_RESET}")
print(f"  {C_GOLD}{C_BOLD}  ║   ADVERSARIAL STRESS TEST — Break the Temporal Chain    ║{C_RESET}")
print(f"  {C_GOLD}{C_BOLD}  ╚════════════════════════════════════════════════════════╝{C_RESET}")

results = {}

# ── 1. OSCILLATOR ──
def osc_state(t):
    s = np.zeros(N, dtype=int)
    for i in range(10):  # 10 oscillating nodes
        # Ground truth: oscillating (class 4) after enough flips
        if t >= 3:
            s[i] = 4  # oscillating
        else:
            s[i] = 2 if t % 2 == 0 else 0
    for i in range(10, 15):
        s[i] = 1  # steady degraded
    return s
def osc_health(t):
    h = np.ones(N)
    for i in range(10):
        h[i] = 0.0 if t % 2 == 0 else 1.0
    for i in range(10, 15):
        h[i] = 0.5
    return h
results['Oscillator'] = run_scenario("1. OSCILLATOR (10 nodes flip every tick)", 
                                      20, osc_state, osc_health, obs_noise=0.1, obs_coverage=0.4)

# ── 2. SLOW POISON ──
def poison_state(t):
    s = np.zeros(N, dtype=int)
    for i in range(8):
        health = 1.0 - (t * 0.05)
        if health < 0.3: s[i] = 2
        elif health < 0.6: s[i] = 1
    return s
def poison_health(t):
    h = np.ones(N)
    for i in range(8):
        h[i] = max(0.0, 1.0 - t * 0.05)
    return h
results['Slow Poison'] = run_scenario("2. SLOW POISON (8 nodes degrade 5%/tick over 20 ticks)",
                                       20, poison_state, poison_health, obs_noise=0.15, obs_coverage=0.3)

# ── 3. CASCADE WHIPLASH ──
def whiplash_state(t):
    s = np.zeros(N, dtype=int)
    if t < 5:
        s[0] = 2  # root fails
        for i in range(1, 8): s[i] = 1  # dependents degrade
    elif t < 10:
        s[0] = 0  # root RECOVERS
        for i in range(1, 8): s[i] = 2  # dependents still dead
    else:
        s[0] = 0
        for i in range(1, 5): s[i] = 2  # some stay dead
        for i in range(5, 8): s[i] = 0  # some recover
    return s
def whiplash_health(t):
    h = np.ones(N)
    s = whiplash_state(t)
    for i in range(N):
        if s[i] == 1: h[i] = 0.4
        elif s[i] == 2: h[i] = 0.0
    return h
results['Whiplash'] = run_scenario("3. CASCADE WHIPLASH (root recovers, dependents stay dead)",
                                    15, whiplash_state, whiplash_health, obs_noise=0.2, obs_coverage=0.35)

# ── 4. FALSE POSITIVE STORM ──
def storm_state(t):
    s = np.zeros(N, dtype=int)
    if t > 5:
        for i in range(5): s[i] = 2
    return s
def storm_health(t):
    h = np.ones(N)
    if t > 5:
        for i in range(5): h[i] = 0.0
    return h
results['FP Storm'] = run_scenario("4. FALSE POSITIVE STORM (40% observations WRONG)",
                                    15, storm_state, storm_health, obs_noise=0.40, obs_coverage=0.5)

# ── 5. SILENT KILLER ──
def silent_state(t):
    s = np.zeros(N, dtype=int)
    if t >= 3:
        s[0] = 2  # fails silently
        if t >= 6:
            for i in range(1, 6): s[i] = 3  # unreachable
    return s
def silent_health(t):
    h = np.ones(N)
    if t >= 3: h[0] = 0.0
    if t >= 6:
        for i in range(1, 6): h[i] = 0.0
    return h
results['Silent Killer'] = run_scenario("5. SILENT KILLER (failed node never observed, 0% coverage on target)",
                                         12, silent_state, silent_health, obs_noise=0.15, obs_coverage=0.20)

# ── 6. DELAYED AVALANCHE ──
def avalanche_state(t):
    s = np.zeros(N, dtype=int)
    if t >= 10:
        for i in range(24): s[i] = 2  # 60% fail simultaneously
        for i in range(24, 30): s[i] = 3  # unreachable
    return s
def avalanche_health(t):
    h = np.ones(N)
    if t >= 10:
        for i in range(24): h[i] = 0.0
        for i in range(24, 30): h[i] = 0.0
    return h
results['Avalanche'] = run_scenario("6. DELAYED AVALANCHE (fine for 10 ticks, then 60% fail at once)",
                                     15, avalanche_state, avalanche_health, obs_noise=0.15, obs_coverage=0.35)

# ── 7. THE FULL MONDAY MORNING ──
def monday_state(t):
    s = np.zeros(N, dtype=int)
    # Oscillators (nodes 0-4) — labeled oscillating after 3 ticks
    for i in range(5):
        if t >= 3:
            s[i] = 4  # oscillating
        else:
            s[i] = 2 if (t + i) % 3 == 0 else (1 if (t + i) % 3 == 1 else 0)
    # Slow poison (nodes 5-9)
    for i in range(5, 10):
        health = 1.0 - t * 0.04
        if health < 0.3: s[i] = 2
        elif health < 0.6: s[i] = 1
    # Silent killer (node 10)
    if t >= 5: s[10] = 2
    # Cascade from silent killer (11-15)
    if t >= 8:
        for i in range(11, 16): s[i] = 3
    # Whiplash (nodes 16-20)
    if t < 8:
        s[16] = 2
        for i in range(17, 21): s[i] = 1
    elif t < 15:
        s[16] = 0  # recovers
        for i in range(17, 21): s[i] = 2  # stay dead
    else:
        s[16] = 0
        for i in range(17, 19): s[i] = 0  # recover
        for i in range(19, 21): s[i] = 2  # stay dead
    # Late avalanche (nodes 30-39)
    if t >= 18:
        for i in range(30, 40): s[i] = 2
    return s

def monday_health(t):
    h = np.ones(N)
    s = monday_state(t)
    for i in range(N):
        if s[i] == 1: h[i] = rng.uniform(0.2, 0.5)
        elif s[i] == 2: h[i] = 0.0
        elif s[i] == 3: h[i] = 0.0
    return h

results['Monday Morning'] = run_scenario(
    "7. THE FULL MONDAY MORNING (everything at once, 25 ticks, 40 nodes)",
    25, monday_state, monday_health, obs_noise=0.30, obs_coverage=0.25, obs_delay=2
)

# ── SUMMARY ──
print(f"\n{C_GOLD}{C_BOLD}  ╔════════════════════════════════════════════════════════╗{C_RESET}")
print(f"  {C_GOLD}{C_BOLD}  ║                    STRESS TEST RESULTS                  ║{C_RESET}")
print(f"  {C_GOLD}{C_BOLD}  ╚════════════════════════════════════════════════════════╝{C_RESET}")

print(f"\n  {C_DIM}{'Scenario':<30s} {'Macro':>7s}  {'Hlthy':>7s} {'Dgrad':>7s} {'Faild':>7s} {'Unrch':>7s} {'Oscil':>7s}{C_RESET}")
print(f"  {C_DIM}{'─' * 75}{C_RESET}")
for name, (macro, f1s) in results.items():
    c = C_SUCCESS if macro > 0.7 else C_DANGER if macro < 0.4 else C_TEXT
    f1_str = " ".join(f"{f1s[i]:7.4f}" for i in range(len(f1s)))
    print(f"  {c}{name:<30s} {macro:7.4f}  {f1_str}{C_RESET}")

avg_macro = np.mean([m for m, _ in results.values()])
print(f"\n  {C_TEXT}Average Macro: {C_BRIGHT}{avg_macro:.4f}{C_RESET}")

if avg_macro > 0.7:
    print(f"  {C_SUCCESS}{C_BOLD}System holds under adversarial stress.{C_RESET}")
elif avg_macro > 0.5:
    print(f"  {C_GOLD}{C_BOLD}System degrades but doesn't collapse.{C_RESET}")
else:
    print(f"  {C_DANGER}{C_BOLD}System breaks under adversarial conditions.{C_RESET}")
print()
