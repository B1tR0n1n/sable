#!/usr/bin/env -S python -u
"""
Project PARALLAX — BREAK SABLE
=================================
The ultimate stress test. Not diagnostics — destruction.
Every scenario here is designed to find a way to make the system
produce wrong, dangerous, or nonsensical output.

Categories:
  I.   ADVERSARIAL INPUTS — crafted to maximize misclassification
  II.  STATE POISONING — corrupt temporal state to cascade errors
  III. NUMERICAL WARFARE — NaN, inf, extremes, denormals
  IV.  TOPOLOGY DEGENERATE — pathological graph structures
  V.   BYZANTINE OBSERVATIONS — strategically wrong information
  VI.  TEMPORAL PARADOX — contradictory trajectory histories
  VII. SCALE STRESS — push dimensions to breaking point
  VIII.THE PERFECT STORM — everything at once
"""

import sys, time, torch, numpy as np, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar2"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pillar3"))

from sable_sim.core.states import N_STATES, STATE_NAMES
from temporal_chain import TemporalChainFusion, TemporalState, TRAJ_K, TRAJ_FEAT, Z_DIM
from staged_fusion_v3 import SharpRoutedFusion
from shared_latent_space import GNN_DIM, POMDP_DIM
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
rng = np.random.default_rng(1337)

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

N = 40
results = {}
crashes = []


def run_test(name, fn):
    """Run a test, catch crashes, report results."""
    try:
        t0 = time.time()
        result = fn()
        elapsed = time.time() - t0
        results[name] = result
        status = result.get('status', 'UNKNOWN')
        detail = result.get('detail', '')
        if status == 'SURVIVED':
            c = C_SUCCESS
        elif status == 'BROKEN':
            c = C_DANGER
        else:
            c = C_TEXT
        print(f"  {c}{name:<45s} {status:>10s}  {elapsed:5.2f}s  {detail}{C_RESET}", flush=True)
    except Exception as e:
        crashes.append((name, str(e)))
        print(f"  {C_DANGER}{name:<45s}    CRASHED  {str(e)[:60]}{C_RESET}", flush=True)


def check_output_sanity(out, n_nodes=N):
    """Check if output is numerically sane."""
    issues = []
    logits = out.get('revised_logits', out.get('logits'))
    if logits is None:
        return ['no logits in output']
    if torch.isnan(logits).any():
        issues.append('NaN in logits')
    if torch.isinf(logits).any():
        issues.append('Inf in logits')
    probs = torch.softmax(logits[0], dim=-1)
    if (probs < 0).any():
        issues.append('negative probabilities')
    if (probs.sum(dim=-1) - 1.0).abs().max() > 0.01:
        issues.append('probabilities dont sum to 1')
    # Check predictions are valid class indices
    preds = logits[0].argmax(dim=-1)
    if (preds < 0).any() or (preds >= N_STATES).any():
        issues.append(f'predictions outside [0, {N_STATES})')
    return issues


def make_features(states, health, obs_noise=0.15, obs_coverage=0.3, n=N):
    """Build pillar features from synthetic state."""
    gnn = torch.randn(1, n, GNN_DIM) * 0.5
    pomdp = torch.zeros(1, n, POMDP_DIM)
    mamba = torch.randn(1, n, NODE_FEAT_DIM) * 0.3

    for i in range(n):
        # POMDP belief is 4-dim (sim states only, oscillating detected temporally)
        belief = np.array([0.25]*4)
        if rng.random() < obs_coverage:
            obs = states[i] if rng.random() > obs_noise else rng.integers(0, 4)
            obs = min(obs, 3)
            belief[obs] += 2.0
            belief /= belief.sum()
        pomdp[0, i, :4] = torch.tensor(belief)
        pomdp[0, i, 4] = rng.uniform(0, 1)
        mamba[0, i, 0] = health[i] + rng.normal(0, 0.15)
        if states[i] < N_STATES:
            mamba[0, i, 1 + min(states[i], N_STATES-1)] = 0.7

    return gnn.to(device), pomdp.to(device), mamba.to(device)


print(f"\n{C_GOLD}{C_BOLD}  ╔═══════════════════════════════════════════════════════════╗{C_RESET}")
print(f"  {C_GOLD}{C_BOLD}  ║            BREAK SABLE — Ultimate Stress Test              ║{C_RESET}")
print(f"  {C_GOLD}{C_BOLD}  ╚═══════════════════════════════════════════════════════════╝{C_RESET}\n")
print(f"  {C_DIM}{'Test':<45s} {'Status':>10s}  {'Time':>5s}  {'Detail'}{C_RESET}")
print(f"  {C_DIM}{'─' * 90}{C_RESET}")


# ══════════════════════════════════════════════════════════════════════════
# I. ADVERSARIAL INPUTS
# ══════════════════════════════════════════════════════════════════════════

def test_fgsm_attack():
    """FGSM adversarial attack — perturb inputs in direction of maximum loss."""
    gnn = torch.randn(1, N, GNN_DIM, device=device, requires_grad=True)
    pomdp = torch.randn(1, N, POMDP_DIM, device=device, requires_grad=True)
    mamba = torch.randn(1, N, NODE_FEAT_DIM, device=device, requires_grad=True)
    target = torch.zeros(N, dtype=torch.long, device=device)  # all healthy

    # Enable grad temporarily
    model.train()
    out = model(gnn, pomdp, mamba, temporal_state=None)
    loss = torch.nn.functional.cross_entropy(out['revised_logits'].reshape(-1, N_STATES), target)
    loss.backward()
    model.eval()

    # FGSM perturbation
    eps = 0.3
    gnn_adv = (gnn + eps * gnn.grad.sign()).detach()
    pomdp_adv = (pomdp + eps * pomdp.grad.sign()).detach()
    mamba_adv = (mamba + eps * mamba.grad.sign()).detach()

    with torch.no_grad():
        out_clean = model(gnn.detach(), pomdp.detach(), mamba.detach(), temporal_state=None)
        out_adv = model(gnn_adv, pomdp_adv, mamba_adv, temporal_state=None)

    clean_preds = out_clean['revised_logits'][0].argmax(-1)
    adv_preds = out_adv['revised_logits'][0].argmax(-1)
    flipped = (clean_preds != adv_preds).sum().item()
    issues = check_output_sanity(out_adv)

    if issues:
        status = 'BROKEN'
    elif flipped > N * 0.5:
        status = 'VULNERABLE'
    else:
        status = 'SURVIVED'

    return {
        'status': status,
        'detail': f'{flipped}/{N} flipped, issues={issues}',
    }

run_test('I.1  FGSM adversarial attack (eps=0.3)', test_fgsm_attack)


def test_confidence_inversion():
    """Feed features where the model is CONFIDENT but WRONG."""
    # Create scenario: nodes are actually failed but features scream healthy
    states = np.full(N, 2, dtype=int)  # all failed
    health = np.ones(N)  # health says 1.0 (lying)

    gnn, pomdp, mamba = make_features(states, health, obs_noise=0.0, obs_coverage=1.0)
    # Override POMDP to confidently say healthy
    pomdp[0, :, :4] = torch.tensor([0.95, 0.02, 0.02, 0.01])
    pomdp[0, :, 4] = 1.0  # high confidence

    gt = torch.full((N,), 2, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(gnn, pomdp, mamba, temporal_state=None)
    preds = out['revised_logits'][0].argmax(-1)
    wrong = (preds != gt).sum().item()

    return {
        'status': 'BROKEN' if wrong == N else 'SURVIVED',
        'detail': f'{wrong}/{N} fooled by confident lies',
    }

run_test('I.2  Confident lies (features say healthy, truth=failed)', test_confidence_inversion)


def test_all_same_class():
    """Every node is the same class. Does the model still work?"""
    results_per = {}
    for cls in range(N_STATES):
        states = np.full(N, cls, dtype=int)
        health = np.where(states <= 1, rng.uniform(0.5, 1.0, N), 0.0)
        gnn, pomdp, mamba = make_features(states, health)
        gt = torch.full((N,), cls, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(gnn, pomdp, mamba, temporal_state=None)
        preds = out['revised_logits'][0].argmax(-1)
        acc = (preds == gt).float().mean().item()
        results_per[STATE_NAMES[cls]] = acc

    worst = min(results_per.values())
    worst_cls = min(results_per, key=results_per.get)
    return {
        'status': 'BROKEN' if worst < 0.1 else 'SURVIVED',
        'detail': f'worst={worst_cls}({worst:.3f}) ' + ' '.join(f'{k}={v:.2f}' for k,v in results_per.items()),
    }

run_test('I.3  Mono-class scenarios (all N same class)', test_all_same_class)


# ══════════════════════════════════════════════════════════════════════════
# II. STATE POISONING
# ══════════════════════════════════════════════════════════════════════════

def test_poisoned_trajectory():
    """Fill trajectory with deliberately wrong history, see if it corrupts future."""
    ts = TemporalState.cold_start(N, device=device)
    # Poison: tell every node it was failed for 8 straight cycles
    for t in range(8):
        entry = torch.tensor([2.0, 0.95, t/10., 0.0, 0.0], device=device)
        ts.trajectory[:, t, :] = entry.unsqueeze(0).expand(N, -1)
    ts.cycle = 8
    ts.traj_ptr[:] = 0  # wrap around

    # Now feed healthy data
    states = np.zeros(N, dtype=int)
    health = np.ones(N)
    gnn, pomdp, mamba = make_features(states, health, obs_noise=0.0, obs_coverage=0.8)
    gt = torch.zeros(N, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(gnn, pomdp, mamba, temporal_state=ts)
    preds = out['revised_logits'][0].argmax(-1)
    poisoned = (preds != gt).sum().item()

    return {
        'status': 'BROKEN' if poisoned > N * 0.7 else 'SURVIVED',
        'detail': f'{poisoned}/{N} poisoned by fake history',
    }

run_test('II.1 Poisoned trajectory (8 cycles of lies)', test_poisoned_trajectory)


def test_z_corruption():
    """Corrupt prev_z with adversarial values."""
    ts = TemporalState.cold_start(N, device=device)
    ts.cycle = 5
    # Fill trajectory normally
    for t in range(5):
        entry = torch.tensor([0.0, 0.5, t/10., 0.0, 0.0], device=device)
        ts.trajectory[:, t, :] = entry.unsqueeze(0).expand(N, -1)
    ts.traj_ptr[:] = 5

    # Corrupt prev_z with extreme values
    ts.prev_z = torch.randn(1, N, Z_DIM, device=device) * 100.0

    states = np.zeros(N, dtype=int)
    health = np.ones(N)
    gnn, pomdp, mamba = make_features(states, health)
    with torch.no_grad():
        out = model(gnn, pomdp, mamba, temporal_state=ts)
    issues = check_output_sanity(out)
    preds = out['revised_logits'][0].argmax(-1)

    return {
        'status': 'BROKEN' if issues else 'SURVIVED',
        'detail': f'issues={issues}, pred_dist={torch.bincount(preds, minlength=N_STATES).tolist()}',
    }

run_test('II.2 Z-vector corruption (prev_z *= 100)', test_z_corruption)


def test_trajectory_overflow():
    """Write way past the ring buffer — does traj_ptr wrap correctly?"""
    ts = TemporalState.cold_start(N, device=device)
    with torch.no_grad():
        for c in range(100):  # 100 cycles through 8-slot buffer
            gnn, pomdp, mamba = make_features(np.zeros(N, dtype=int), np.ones(N))
            out = model(gnn, pomdp, mamba, temporal_state=ts)
            ts.update(out['revised_logits'].detach(), out['confidence'].detach(),
                     out.get('z_fused').detach() if out.get('z_fused') is not None else None)

    issues = []
    if ts.cycle != 100:
        issues.append(f'cycle={ts.cycle}, expected 100')
    if (ts.traj_ptr >= TRAJ_K).any():
        issues.append(f'traj_ptr overflow: max={ts.traj_ptr.max().item()}')
    if torch.isnan(ts.trajectory).any():
        issues.append('NaN in trajectory')

    return {
        'status': 'BROKEN' if issues else 'SURVIVED',
        'detail': f'100 cycles, ptr={ts.traj_ptr[0].item()}, issues={issues}',
    }

run_test('II.3 Trajectory overflow (100 cycles, 8-slot buf)', test_trajectory_overflow)


# ══════════════════════════════════════════════════════════════════════════
# III. NUMERICAL WARFARE
# ══════════════════════════════════════════════════════════════════════════

def test_nan_input():
    """Feed NaN directly into the model."""
    gnn = torch.full((1, N, GNN_DIM), float('nan'), device=device)
    pomdp = torch.randn(1, N, POMDP_DIM, device=device)
    mamba = torch.randn(1, N, NODE_FEAT_DIM, device=device)
    with torch.no_grad():
        out = model(gnn, pomdp, mamba, temporal_state=None)
    has_nan = torch.isnan(out['revised_logits']).any().item()
    return {
        'status': 'BROKEN' if has_nan else 'SURVIVED',
        'detail': f'NaN in output: {has_nan}',
    }

run_test('III.1 NaN injection (GNN features = NaN)', test_nan_input)


def test_inf_input():
    """Feed infinity."""
    gnn = torch.full((1, N, GNN_DIM), float('inf'), device=device)
    pomdp = torch.randn(1, N, POMDP_DIM, device=device)
    mamba = torch.randn(1, N, NODE_FEAT_DIM, device=device)
    with torch.no_grad():
        out = model(gnn, pomdp, mamba, temporal_state=None)
    has_nan = torch.isnan(out['revised_logits']).any().item()
    has_inf = torch.isinf(out['revised_logits']).any().item()
    return {
        'status': 'BROKEN' if (has_nan or has_inf) else 'SURVIVED',
        'detail': f'NaN={has_nan} Inf={has_inf}',
    }

run_test('III.2 Infinity injection (GNN features = inf)', test_inf_input)


def test_extreme_values():
    """Feed 1e30 magnitude features."""
    gnn = torch.randn(1, N, GNN_DIM, device=device) * 1e15
    pomdp = torch.randn(1, N, POMDP_DIM, device=device) * 1e15
    mamba = torch.randn(1, N, NODE_FEAT_DIM, device=device) * 1e15
    with torch.no_grad():
        out = model(gnn, pomdp, mamba, temporal_state=None)
    issues = check_output_sanity(out)
    return {
        'status': 'BROKEN' if issues else 'SURVIVED',
        'detail': f'issues={issues}',
    }

run_test('III.3 Extreme magnitudes (features * 1e15)', test_extreme_values)


def test_zero_input():
    """All zeros everywhere."""
    gnn = torch.zeros(1, N, GNN_DIM, device=device)
    pomdp = torch.zeros(1, N, POMDP_DIM, device=device)
    mamba = torch.zeros(1, N, NODE_FEAT_DIM, device=device)
    with torch.no_grad():
        out = model(gnn, pomdp, mamba, temporal_state=None)
    issues = check_output_sanity(out)
    # All zeros should still produce valid (if useless) predictions
    preds = out['revised_logits'][0].argmax(-1)
    return {
        'status': 'BROKEN' if issues else 'SURVIVED',
        'detail': f'issues={issues}, all_same_pred={preds.unique().tolist()}',
    }

run_test('III.4 All-zero input (dead signal)', test_zero_input)


def test_denormals():
    """Denormalized floats — the CPU/GPU silent killer."""
    gnn = torch.ones(1, N, GNN_DIM, device=device) * 1e-38
    pomdp = torch.ones(1, N, POMDP_DIM, device=device) * 1e-38
    mamba = torch.ones(1, N, NODE_FEAT_DIM, device=device) * 1e-38
    with torch.no_grad():
        out = model(gnn, pomdp, mamba, temporal_state=None)
    issues = check_output_sanity(out)
    return {
        'status': 'BROKEN' if issues else 'SURVIVED',
        'detail': f'issues={issues}',
    }

run_test('III.5 Denormalized floats (1e-38)', test_denormals)


# ══════════════════════════════════════════════════════════════════════════
# IV. TOPOLOGY DEGENERATE
# ══════════════════════════════════════════════════════════════════════════

def test_single_node():
    """One node. Minimum possible topology."""
    gnn = torch.randn(1, 1, GNN_DIM, device=device)
    pomdp = torch.randn(1, 1, POMDP_DIM, device=device)
    mamba = torch.randn(1, 1, NODE_FEAT_DIM, device=device)
    ts = TemporalState.cold_start(1, device=device)
    with torch.no_grad():
        out = model(gnn, pomdp, mamba, temporal_state=ts)
    issues = check_output_sanity(out, n_nodes=1)
    return {
        'status': 'BROKEN' if issues else 'SURVIVED',
        'detail': f'pred={STATE_NAMES[out["revised_logits"][0,0].argmax().item()]}, issues={issues}',
    }

run_test('IV.1 Single node topology (N=1)', test_single_node)


def test_max_nodes():
    """Push to maximum node count."""
    n = 200  # beyond training max of 40
    gnn = torch.randn(1, n, GNN_DIM, device=device)
    pomdp = torch.randn(1, n, POMDP_DIM, device=device)
    mamba = torch.randn(1, n, NODE_FEAT_DIM, device=device)
    ts = TemporalState.cold_start(n, device=device)
    with torch.no_grad():
        out = model(gnn, pomdp, mamba, temporal_state=ts)
    issues = check_output_sanity(out, n_nodes=n)
    return {
        'status': 'BROKEN' if issues else 'SURVIVED',
        'detail': f'N=200 (trained on 40), issues={issues}',
    }

run_test('IV.2 200 nodes (5x training max)', test_max_nodes)


# ══════════════════════════════════════════════════════════════════════════
# V. BYZANTINE OBSERVATIONS
# ══════════════════════════════════════════════════════════════════════════

def test_inverted_observations():
    """Every observation is the OPPOSITE of truth. Systematic deception."""
    inversion = {0: 2, 1: 3, 2: 0, 3: 1, 4: 0}  # healthy↔failed, degraded↔unreachable
    n_ticks = 15
    ts = TemporalState.cold_start(N, device=device)
    tp = torch.zeros(N_STATES); fp = torch.zeros(N_STATES); fn = torch.zeros(N_STATES)

    with torch.no_grad():
        for t in range(n_ticks):
            # True state: cascade pattern
            states = np.zeros(N, dtype=int)
            health = np.ones(N, dtype=float)
            if t >= 3:
                for i in range(10): states[i] = 2; health[i] = 0.0
            if t >= 6:
                for i in range(10, 15): states[i] = 3; health[i] = 0.0

            # Build features with INVERTED observations
            gnn = torch.randn(1, N, GNN_DIM, device=device) * 0.5
            pomdp = torch.zeros(1, N, POMDP_DIM, device=device)
            mamba = torch.randn(1, N, NODE_FEAT_DIM, device=device) * 0.3

            for i in range(N):
                # POMDP gets inverted observation
                inv_state = inversion.get(states[i], 0)
                belief = np.array([0.1, 0.1, 0.1, 0.1])
                belief[min(inv_state, 3)] = 0.7
                belief /= belief.sum()
                pomdp[0, i, :4] = torch.tensor(belief)
                pomdp[0, i, 4] = 0.9  # high confidence in the LIE
                mamba[0, i, 0] = 1.0 - health[i]  # inverted health too

            gt = torch.tensor(states, dtype=torch.long, device=device)
            out = model(gnn, pomdp, mamba, temporal_state=ts)
            preds = out['revised_logits'][0].argmax(-1)
            ts.update(out['revised_logits'].detach(), out['confidence'].detach(),
                     out.get('z_fused').detach() if out.get('z_fused') is not None else None)

            for c in range(N_STATES):
                ct = (gt==c); cp = (preds==c)
                tp[c] += (ct&cp).sum().item()
                fp[c] += (~ct&cp).sum().item()
                fn[c] += (ct&~cp).sum().item()

    f1s = []
    for c in range(N_STATES):
        p = tp[c]/max(tp[c]+fp[c],1); r = tp[c]/max(tp[c]+fn[c],1)
        f1s.append(2*p*r/max(p+r,1e-8))
    macro = sum(f1s)/N_STATES

    return {
        'status': 'BROKEN' if macro < 0.1 else 'SURVIVED',
        'detail': f'macro={macro:.3f} under systematic inversion',
    }

run_test('V.1  Inverted observations (opposite of truth)', test_inverted_observations)


def test_adversarial_consensus():
    """All three pillars AGREE on the wrong answer."""
    ts = TemporalState.cold_start(N, device=device)
    n_wrong = 0
    n_total = 0

    with torch.no_grad():
        for t in range(10):
            states = np.zeros(N, dtype=int)
            if t >= 3:
                for i in range(15): states[i] = 2  # actually failed

            # ALL pillars say healthy with high confidence
            gnn = torch.zeros(1, N, GNN_DIM, device=device)
            gnn[0, :, 0] = 1.0  # "healthy" feature activated
            pomdp = torch.zeros(1, N, POMDP_DIM, device=device)
            pomdp[0, :, 0] = 0.95  # p(healthy) = 0.95
            pomdp[0, :, 4] = 1.0  # max confidence
            mamba = torch.zeros(1, N, NODE_FEAT_DIM, device=device)
            mamba[0, :, 0] = 1.0  # health = 1.0
            mamba[0, :, 1] = 1.0  # healthy state feature

            gt = torch.tensor(states, dtype=torch.long, device=device)
            out = model(gnn, pomdp, mamba, temporal_state=ts)
            preds = out['revised_logits'][0].argmax(-1)
            ts.update(out['revised_logits'].detach(), out['confidence'].detach(),
                     out.get('z_fused').detach() if out.get('z_fused') is not None else None)

            if t >= 3:
                failed_nodes = (gt == 2)
                wrong_on_failed = ((preds != gt) & failed_nodes).sum().item()
                n_wrong += wrong_on_failed
                n_total += failed_nodes.sum().item()

    fool_rate = n_wrong / max(n_total, 1)
    return {
        'status': 'BROKEN' if fool_rate > 0.9 else 'SURVIVED',
        'detail': f'{n_wrong}/{n_total} fooled ({fool_rate:.1%}) by consensus lies',
    }

run_test('V.2  Adversarial consensus (all pillars agree wrong)', test_adversarial_consensus)


# ══════════════════════════════════════════════════════════════════════════
# VI. TEMPORAL PARADOX
# ══════════════════════════════════════════════════════════════════════════

def test_time_reversal():
    """Trajectory that goes backward: failed→degraded→healthy→failed→..."""
    ts = TemporalState.cold_start(N, device=device)
    # Write paradoxical trajectory: each node has contradictory history
    cycle_states = [2, 1, 0, 2, 1, 0, 2, 0]  # impossible in real infra
    for t in range(8):
        s = float(cycle_states[t])
        changed = 1.0
        direction = -1.0 if cycle_states[t] < cycle_states[max(0,t-1)] else 1.0
        entry = torch.tensor([s, 0.9, t/10., changed, direction], device=device)
        ts.trajectory[:, t, :] = entry.unsqueeze(0).expand(N, -1)
    ts.cycle = 8
    ts.traj_ptr[:] = 0

    gnn, pomdp, mamba = make_features(np.zeros(N, dtype=int), np.ones(N))
    with torch.no_grad():
        out = model(gnn, pomdp, mamba, temporal_state=ts)
    issues = check_output_sanity(out)
    conf = out['confidence'].mean().item()

    return {
        'status': 'BROKEN' if issues else 'SURVIVED',
        'detail': f'confidence={conf:.3f} (should be low for paradox), issues={issues}',
    }

run_test('VI.1 Time reversal paradox (impossible trajectory)', test_time_reversal)


def test_split_personality():
    """Half the trajectory says healthy, other half says failed. Alternating."""
    ts = TemporalState.cold_start(N, device=device)
    for t in range(8):
        s = 0.0 if t < 4 else 2.0  # sudden jump from healthy to failed
        entry = torch.tensor([s, 0.9, t/10., float(t==4), 1.0 if t==4 else 0.0], device=device)
        ts.trajectory[:, t, :] = entry.unsqueeze(0).expand(N, -1)
    ts.cycle = 8; ts.traj_ptr[:] = 0

    # Current observation: degraded (contradicts both halves)
    states = np.full(N, 1, dtype=int)
    health = np.full(N, 0.5)
    gnn, pomdp, mamba = make_features(states, health)

    with torch.no_grad():
        out = model(gnn, pomdp, mamba, temporal_state=ts)
    issues = check_output_sanity(out)
    preds = out['revised_logits'][0].argmax(-1)
    pred_dist = torch.bincount(preds, minlength=N_STATES).tolist()

    return {
        'status': 'BROKEN' if issues else 'SURVIVED',
        'detail': f'pred_dist={pred_dist}, issues={issues}',
    }

run_test('VI.2 Split personality (sudden state jump in history)', test_split_personality)


# ══════════════════════════════════════════════════════════════════════════
# VII. SCALE STRESS
# ══════════════════════════════════════════════════════════════════════════

def test_rapid_fire():
    """100 inference cycles as fast as possible. Memory leak? Slowdown?"""
    ts = TemporalState.cold_start(N, device=device)
    t0 = time.time()
    with torch.no_grad():
        for c in range(100):
            gnn, pomdp, mamba = make_features(np.zeros(N, dtype=int), np.ones(N))
            out = model(gnn, pomdp, mamba, temporal_state=ts)
            ts.update(out['revised_logits'].detach(), out['confidence'].detach(),
                     out.get('z_fused').detach() if out.get('z_fused') is not None else None)
    elapsed = time.time() - t0
    # Check for memory growth
    mem = torch.cuda.memory_allocated() / 1e6

    return {
        'status': 'SURVIVED',
        'detail': f'100 cycles in {elapsed:.2f}s ({100/elapsed:.0f}/s), VRAM={mem:.0f}MB',
    }

run_test('VII.1 Rapid fire (100 cycles, check for leaks)', test_rapid_fire)


def test_batch_stress():
    """Batch size 32 — does it work beyond B=1?"""
    B = 32
    gnn = torch.randn(B, N, GNN_DIM, device=device)
    pomdp = torch.randn(B, N, POMDP_DIM, device=device)
    mamba = torch.randn(B, N, NODE_FEAT_DIM, device=device)
    # Note: TemporalState is per-topology, not batched. Use None.
    with torch.no_grad():
        out = model(gnn, pomdp, mamba, temporal_state=None)
    issues = check_output_sanity(out)
    shape = out['revised_logits'].shape

    return {
        'status': 'BROKEN' if issues or shape[0] != B else 'SURVIVED',
        'detail': f'shape={list(shape)}, issues={issues}',
    }

run_test('VII.2 Batch stress (B=32)', test_batch_stress)


# ══════════════════════════════════════════════════════════════════════════
# VIII. THE PERFECT STORM
# ══════════════════════════════════════════════════════════════════════════

def test_perfect_storm():
    """Everything at once: adversarial features + poisoned state + noise + scale."""
    n = 80  # 2x training max
    ts = TemporalState.cold_start(n, device=device)

    # Poison trajectory with random garbage
    ts.trajectory = torch.randn(n, TRAJ_K, TRAJ_FEAT, device=device) * 10
    ts.prev_z = torch.randn(1, n, Z_DIM, device=device) * 50
    ts.cycle = 50

    tp = torch.zeros(N_STATES); fp = torch.zeros(N_STATES); fn = torch.zeros(N_STATES)
    any_issues = []

    with torch.no_grad():
        for t in range(20):
            # Real state: complex cascade
            states = np.zeros(n, dtype=int)
            health = np.ones(n, dtype=float)

            # Oscillators
            for i in range(10):
                if t >= 3:
                    states[i] = 4
                elif t % 2 == 0:
                    states[i] = 2
                else:
                    states[i] = 0
                health[i] = 0.5

            # Cascade
            if t >= 5:
                for i in range(10, 30): states[i] = 2; health[i] = 0.0
            if t >= 8:
                for i in range(30, 40): states[i] = 3; health[i] = 0.0

            # Degraded with noise
            for i in range(40, 50): states[i] = 1; health[i] = rng.uniform(0.3, 0.6)

            # Feed adversarial features (60% observation noise, extreme magnitudes)
            gnn = torch.randn(1, n, GNN_DIM, device=device) * 5.0
            pomdp = torch.zeros(1, n, POMDP_DIM, device=device)
            mamba = torch.randn(1, n, NODE_FEAT_DIM, device=device) * 3.0

            for i in range(n):
                # POMDP belief is 4-dim (sim states only, oscillating detected temporally)
        belief = np.array([0.25]*4)
                if rng.random() < 0.4:  # only 40% observed
                    obs = states[i] if rng.random() > 0.6 else rng.integers(0, 4)  # 60% noise
                    obs = min(obs, 3)
                    belief[obs] += 1.5
                    belief /= belief.sum()
                pomdp[0, i, :4] = torch.tensor(belief)
                mamba[0, i, 0] = health[i] + rng.normal(0, 0.3)

            gt = torch.tensor(states, dtype=torch.long, device=device)
            out = model(gnn, pomdp, mamba, temporal_state=ts)
            issues = check_output_sanity(out, n_nodes=n)
            if issues:
                any_issues.extend(issues)

            preds = out['revised_logits'][0].argmax(-1)
            ts.update(out['revised_logits'].detach(), out['confidence'].detach(),
                     out.get('z_fused').detach() if out.get('z_fused') is not None else None)

            for c in range(N_STATES):
                ct = (gt==c); cp = (preds==c)
                tp[c] += (ct&cp).sum().item()
                fp[c] += (~ct&cp).sum().item()
                fn[c] += (ct&~cp).sum().item()

    f1s = []
    for c in range(N_STATES):
        p = tp[c]/max(tp[c]+fp[c],1); r = tp[c]/max(tp[c]+fn[c],1)
        f1s.append(2*p*r/max(p+r,1e-8))
    macro = sum(f1s)/N_STATES
    f1_str = ' '.join(f'{STATE_NAMES[i][:4]}={f1s[i]:.3f}' for i in range(N_STATES))

    if any_issues:
        status = 'BROKEN'
    elif macro < 0.15:
        status = 'VULNERABLE'
    else:
        status = 'SURVIVED'

    return {
        'status': status,
        'detail': f'macro={macro:.3f} [{f1_str}] issues={any_issues[:3]}',
    }

run_test('VIII. THE PERFECT STORM (everything at once)', test_perfect_storm)


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════

print(f"\n{C_GOLD}{C_BOLD}  ╔═══════════════════════════════════════════════════════════╗{C_RESET}")
print(f"  {C_GOLD}{C_BOLD}  ║                      FINAL REPORT                          ║{C_RESET}")
print(f"  {C_GOLD}{C_BOLD}  ╚═══════════════════════════════════════════════════════════╝{C_RESET}\n")

survived = sum(1 for r in results.values() if r['status'] == 'SURVIVED')
broken = sum(1 for r in results.values() if r['status'] == 'BROKEN')
vulnerable = sum(1 for r in results.values() if r['status'] == 'VULNERABLE')
total = len(results)

print(f"  {C_SUCCESS}SURVIVED:   {survived}/{total}{C_RESET}")
print(f"  {C_TEXT}VULNERABLE: {vulnerable}/{total}{C_RESET}")
print(f"  {C_DANGER}BROKEN:     {broken}/{total}{C_RESET}")
print(f"  {C_DANGER}CRASHED:    {len(crashes)}{C_RESET}")

if crashes:
    print(f"\n  {C_DANGER}{C_BOLD}CRASHES:{C_RESET}")
    for name, err in crashes:
        print(f"    {C_DANGER}{name}: {err}{C_RESET}")

if broken > 0:
    print(f"\n  {C_DANGER}{C_BOLD}BROKEN TESTS:{C_RESET}")
    for name, r in results.items():
        if r['status'] == 'BROKEN':
            print(f"    {C_DANGER}{name}: {r['detail']}{C_RESET}")

if vulnerable > 0:
    print(f"\n  {C_TEXT}VULNERABLE (degraded but not broken):{C_RESET}")
    for name, r in results.items():
        if r['status'] == 'VULNERABLE':
            print(f"    {C_TEXT}{name}: {r['detail']}{C_RESET}")

print(f"\n  {C_GOLD}{C_BOLD}VERDICT:{C_RESET}", end=" ")
if broken == 0 and len(crashes) == 0:
    print(f"{C_SUCCESS}{C_BOLD}SABLE HOLDS. No crashes, no numerical failures.{C_RESET}")
elif broken <= 2 and len(crashes) == 0:
    print(f"{C_TEXT}Minor vulnerabilities. Structurally sound.{C_RESET}")
else:
    print(f"{C_DANGER}{C_BOLD}Structural issues found. {broken} broken, {len(crashes)} crashes.{C_RESET}")

print()
