"""
SABLE vs Baselines on Alibaba Cluster Trace 2018
==================================================
Honest shootout: SABLE's edge for infrastructure diagnostics is supposed
to come from cascade reasoning over real topology + telemetry. This script
tests that claim against three dumb-but-defensible baselines:

  1. ThresholdPredictor   — fires when any single metric crosses 90%
  2. IsolationForestPredictor — sklearn anomaly detection on metric vector
  3. RandomPredictor      — uniform random failure prob (chance baseline)
  4. SABLEHealthPredictor — current health score from the adapter (proxy
                             for SABLE inference until engine is wired in)

Forecasting task: at tick T, predict which machines will be in a failed
state at any tick in [T+1, T+lead_window]. Score: precision, recall, F1,
mean lead time on true positives.

Usage (from /mnt/vault/projects/sable):
    source ~/ml-env/bin/activate
    python eval/run_alibaba_baselines.py \\
        --data-dir /mnt/vault/projects/sable/data/alibaba_2018 \\
        --max-machines 500 \\
        --max-ticks 500 \\
        --lead-window 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support

# Run from /mnt/vault/projects/sable (matches existing project convention).
# If invoked as a module from elsewhere, prepend the project root to sys.path.
import os
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from adapters.alibaba import AlibabaAdapter  # noqa: E402
from adapters.base import SystemSnapshot     # noqa: E402


# ── Predictors ───────────────────────────────────────────────────────────


class Predictor:
    name = "base"

    def fit(self, snapshots: list[SystemSnapshot]) -> None:
        pass

    def predict(self, snap: SystemSnapshot) -> dict[str, float]:
        """Return {machine_id: failure_probability in [0,1]}."""
        raise NotImplementedError


class RandomPredictor(Predictor):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def predict(self, snap):
        return {mid: float(self.rng.random()) for mid in snap.nodes}


class ThresholdPredictor(Predictor):
    name = "thresholds"

    def __init__(self, cpu_thr=90, mem_thr=95, disk_thr=90):
        self.cpu_thr = cpu_thr
        self.mem_thr = mem_thr
        self.disk_thr = disk_thr

    def predict(self, snap):
        out = {}
        for mid, n in snap.nodes.items():
            m = n.metrics
            score = max(
                m.get("cpu_utilization", 0) / max(self.cpu_thr, 1),
                m.get("mem_used_pct", 0) / max(self.mem_thr, 1),
                m.get("disk_io_util", 0) / max(self.disk_thr, 1),
            )
            out[mid] = float(min(score, 1.0))
        return out


class IsolationForestPredictor(Predictor):
    name = "isolation_forest"

    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(
            contamination=contamination, random_state=0, n_jobs=-1,
        )
        self._features = ["cpu_utilization", "mem_used_pct",
                          "disk_io_util", "net_in_pct", "net_out_pct"]

    def fit(self, snapshots):
        X = np.array([
            [n.metrics.get(f, 0.0) for f in self._features]
            for snap in snapshots for n in snap.nodes.values()
        ])
        if len(X):
            self.model.fit(X)

    def predict(self, snap):
        mids = list(snap.nodes.keys())
        X = np.array([
            [snap.nodes[m].metrics.get(f, 0.0) for f in self._features]
            for m in mids
        ])
        # decision_function: higher = more normal. Invert + min-max.
        scores = -self.model.decision_function(X)
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min)
        else:
            scores = np.zeros_like(scores)
        return {m: float(s) for m, s in zip(mids, scores)}


class SABLEHealthPredictor(Predictor):
    """Placeholder using the adapter's composite health.
    Replace .predict() with a call into the SABLE engine once wired up.
    """
    name = "sable_health_proxy"

    def predict(self, snap):
        return {
            mid: float(1.0 - (n.health if n.health is not None else 0.5))
            for mid, n in snap.nodes.items()
        }


# ── Eval harness ─────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    n_positives: int
    n_predicted: int
    mean_lead_seconds: float
    seconds: float
    notes: dict = field(default_factory=dict)


def build_future_labels(
    adapter: AlibabaAdapter, lead_window: int,
) -> tuple[list[dict[str, bool]], list[dict[str, int]]]:
    """For each tick T, label[T][m] = True if m is failed in (T, T+lead_window].
    Also return lead_time[T][m] = first tick offset where m failed (else -1).
    """
    per_tick_gt = [adapter.get_ground_truth(t) for t in range(adapter.n_ticks)]
    machines = adapter.machines
    labels = []
    leads = []
    for t in range(adapter.n_ticks):
        end = min(adapter.n_ticks, t + 1 + lead_window)
        lab = {}
        ld = {}
        for m in machines:
            first = -1
            for tt in range(t + 1, end):
                if per_tick_gt[tt].get(m, False):
                    first = tt - t
                    break
            lab[m] = first > 0
            ld[m] = first
        labels.append(lab)
        leads.append(ld)
    return labels, leads


def score_predictor(
    predictor: Predictor,
    snapshots: list[SystemSnapshot],
    labels: list[dict[str, bool]],
    leads: list[dict[str, int]],
    threshold: float,
    tick_seconds: int,
) -> EvalResult:
    t0 = time.time()
    y_true: list[int] = []
    y_pred: list[int] = []
    lead_obs: list[int] = []

    for snap, lab, ld in zip(snapshots, labels, leads):
        probs = predictor.predict(snap)
        for mid in snap.nodes:
            true = bool(lab.get(mid, False))
            pred = probs.get(mid, 0.0) >= threshold
            y_true.append(int(true))
            y_pred.append(int(pred))
            if true and pred and ld.get(mid, -1) > 0:
                lead_obs.append(ld[mid])

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0,
    )
    return EvalResult(
        name=predictor.name,
        threshold=threshold,
        precision=float(p),
        recall=float(r),
        f1=float(f),
        n_positives=int(sum(y_true)),
        n_predicted=int(sum(y_pred)),
        mean_lead_seconds=float(np.mean(lead_obs) * tick_seconds) if lead_obs else 0.0,
        seconds=time.time() - t0,
        notes={"n_samples": len(y_true)},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-machines", type=int, default=500)
    ap.add_argument("--max-ticks", type=int, default=500)
    ap.add_argument("--tick-seconds", type=int, default=300)
    ap.add_argument("--lead-window", type=int, default=12,
                    help="ticks ahead to consider for failure prediction")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="/mnt/vault/projects/sable/eval/results_alibaba.json")
    args = ap.parse_args()

    print(f"[load] adapter from {args.data_dir}")
    t0 = time.time()
    adapter = AlibabaAdapter(args.data_dir)
    adapter.load(
        tick_seconds=args.tick_seconds,
        max_machines=args.max_machines,
        max_ticks=args.max_ticks,
    )
    print(f"[load] {time.time()-t0:.1f}s — "
          f"{len(adapter.machines)} machines, {adapter.n_ticks} ticks, "
          f"{len(adapter.discover_topology())} edges")

    print("[snapshots] materializing all ticks")
    snapshots = [adapter.get_tick(t) for t in range(adapter.n_ticks)]
    print(f"[labels] building forward-looking labels (lead_window={args.lead_window})")
    labels, leads = build_future_labels(adapter, args.lead_window)

    n_train = int(adapter.n_ticks * args.train_frac)
    train_snaps = snapshots[:n_train]
    test_snaps = snapshots[n_train:]
    test_labels = labels[n_train:]
    test_leads = leads[n_train:]
    n_pos_test = sum(1 for L in test_labels for v in L.values() if v)
    print(f"[split] train={n_train} ticks, test={len(test_snaps)} ticks, "
          f"positive labels in test={n_pos_test}")

    if n_pos_test == 0:
        print("⚠  No positive labels in test split. Failures may be sparse — "
              "increase --max-ticks or pick a different slice of the trace.")

    predictors = [
        RandomPredictor(),
        ThresholdPredictor(),
        IsolationForestPredictor(),
        SABLEHealthPredictor(),
    ]

    results = []
    for p in predictors:
        print(f"[fit ] {p.name}")
        p.fit(train_snaps)
        print(f"[eval] {p.name}")
        r = score_predictor(p, test_snaps, test_labels, test_leads,
                            args.threshold, args.tick_seconds)
        results.append(r)
        print(f"       precision={r.precision:.3f} recall={r.recall:.3f} "
              f"f1={r.f1:.3f} lead_s={r.mean_lead_seconds:.0f} "
              f"({r.seconds:.1f}s)")

    out = {
        "args": vars(args),
        "n_machines": len(adapter.machines),
        "n_ticks": adapter.n_ticks,
        "n_edges": len(adapter.discover_topology()),
        "results": [r.__dict__ for r in results],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] results → {args.out}")

    print("\n┌─ Summary ────────────────────────────────────────────────")
    print(f"│ {'predictor':<22} {'P':>6} {'R':>6} {'F1':>6} {'lead_s':>8}")
    print("├──────────────────────────────────────────────────────────")
    for r in sorted(results, key=lambda x: -x.f1):
        print(f"│ {r.name:<22} {r.precision:>6.3f} {r.recall:>6.3f} "
              f"{r.f1:>6.3f} {r.mean_lead_seconds:>8.0f}")
    print("└──────────────────────────────────────────────────────────")


if __name__ == "__main__":
    sys.exit(main() or 0)
