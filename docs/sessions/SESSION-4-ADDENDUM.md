# Session 4 Addendum — Production Demo + Honest Validation

## Docker Production Build

Built and tested. Files at `/mnt/vault/sable/docker/`:
- `server.py` — FastAPI + WebSocket, 8 REST endpoints + live streaming
- `sable_engine.py` — Stateful inference wrapper with NaN/Inf sanitization + recommendations engine
- `dashboard.html` — Single-file b1tr0n1n branded UI with 4 panels + gold FIX button
- `precompute_scenarios.py` — 5 manually-specified cascade scenarios
- `Dockerfile`, `build.sh`, `run.sh` — Container build pipeline

**sm_120 issue:** Docker base image doesn't support Blackwell GPUs. Added CPU fallback but bare metal is the correct deployment for sable. Docker is for distribution to other machines.

---

## Honest Demo Results

Scenarios use harder-than-training conditions: 20% observability, 25% noise, 3-tick delay, fresh seeds.

| Scenario | Accuracy | Confidence | What It Proves |
|----------|----------|------------|----------------|
| Silent Killer | **0.827** | 0.628 | Inference from absence |
| Monday Morning | **0.786** | 0.755 | Compound cascade tracking |
| Slow Poison | **0.779** | 0.643 | Subtle trend detection |
| Whiplash | **0.766** | 0.692 | Causal reasoning |
| Random Chaos | **0.664** | 0.715 | Generalization floor |
| **Average** | **0.764** | **0.687** | |

5ms per inference on CUDA. 141ms cold start.

---

## Recommendations Engine

`SableEngine.get_recommendations()` analyzes trajectories to output:
1. **Root cause** — earliest failed node
2. **Prioritized actions** — INVESTIGATE ROOT CAUSE → RESTART/RECOVER → CHECK CONNECTIVITY → STABILIZE → MONITOR
3. **Per-node reasoning** — why this node, what to do, expected impact

Exposed via `/api/recommendations` and gold FIX button in dashboard.

---

## Known Issues for Session 5

### 1. Confidence Head (Critical)
Average confidence 0.687 is too low. Three causes:
- RevisionGate trained on noisy data → outputs conservative middle values
- Cold start at 0.5 drags average
- Confidence head can't see its own accuracy history

**Fix:** Give confidence head access to running accuracy from trajectory buffer. Retrain with calibration loss so 0.9 confidence = 90% accuracy.

### 2. sable_sim PropagationEngine Bug
Cascade doesn't spread through dependencies on manually-injected failures. 3 failed → propagate → still 3 failed. The propagation loop only processes changes recorded through the state tracking system, but direct component state modification bypasses that tracking.

**Fix:** Investigate `PropagationEngine.propagate()` and `SystemState.record_change()` — injected failures need to be recorded as StateChanges at tick 0 for propagation to find them.

### 3. Docker sm_120 Support
Need custom base image with PyTorch 2.7+ for Blackwell GPU support in container. Check NVIDIA NGC for `nvcr.io/nvidia/pytorch:25.XX-py3` with sm_120.

---

## File Changes (Addendum)

### New
```
docker/sable_engine.py       — Engine wrapper + recommendations
docker/server.py             — FastAPI server
docker/dashboard.html        — Branded dashboard with FIX button
docker/precompute_scenarios.py — Manual state evolution scenarios
docker/Dockerfile, build.sh, run.sh
```

### Modified
```
docker/precompute_scenarios.py — Rewrote from sable_sim propagation to manual state functions
docker/server.py              — Added inference_ms timing, /api/recommendations
docker/sable_engine.py        — Added get_recommendations(), CUDA fallback
```
