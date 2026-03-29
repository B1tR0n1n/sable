# SABLE_STABLE — Current System State
**As of:** 2026-03-27
**Location:** `/mnt/vault/sable/` on machine `sable`
**Hardware:** AMD Ryzen 9 9950X3D, NVIDIA RTX 5090 (32GB VRAM), 64GB DDR5

---

## What SABLE Is

A three-pillar cognitive architecture for infrastructure diagnostics. Given infrastructure topology and telemetry, it classifies every node into one of 5 states (healthy, degraded, failed, unreachable, oscillating), tracks how those states evolve over time, and recommends what to fix first.

**Total parameters:** 5.4M active
**Inference speed:** 5ms per tick on RTX 5090
**VRAM footprint:** 22MB steady state
**State space:** 5 classes (healthy, degraded, failed, unreachable, oscillating)

---

## Architecture

```
Pillar 1: GNN                Pillar 2: POMDP            Pillar 3: Mamba
GATv2Conv encoder             POMCP tree search          SelectiveSSM
128 hidden, 3 layers, 4 heads CPU Bayesian beliefs       4 layers, 256 dim
Trained on:                   No learned params          Trained on:
• FB15k-237 (237 relations)   Belief state persists      • PowerGraph IEEE39 (28K)
• ogbl-biokg (93K nodes)      across inference cycles    • PowerGraph IEEE118 (122K)
• Topology Zoo (191 real)                                Stateful h_init/h_final
• Microservice graphs (20)
       │                           │                          │
       └───────────┬───────────────┘                          │
                   │                                          │
         ┌────────▼──────────────────────────────────────────▼────┐
         │                Staged Fusion v3                         │
         │   4 expert heads (GNN, POMDP, Mamba, Fusion)            │
         │   Sharp router: hard argmax at eval, router supervision │
         │   Router input: 4×5 logits + 3 norms = 23-dim          │
         │   297K parameters                                       │
         └────────────────────┬───────────────────────────────────┘
                              │
         ┌────────────────────▼───────────────────────────────────┐
         │                Temporal Chain                            │
         │   ContextMixer (50K) → Fusion → RevisionGate (24K)     │
         │   Verdict t → Input t+1 via trajectory ring buffer     │
         │   NaN/Inf sanitization at entry                         │
         │   74K parameters                                        │
         └────────────────────┬───────────────────────────────────┘
                              │
         ┌────────────────────▼───────────────────────────────────┐
         │                Production Server                        │
         │   FastAPI + WebSocket + b1tr0n1n branded dashboard      │
         │   5 pre-computed scenarios, recommendations engine      │
         │   asyncio.Lock on state, run_in_executor for inference  │
         └────────────────────────────────────────────────────────┘
```

---

## Metrics

### Training Eval (on training-distribution data)
| Class | F1 |
|-------|-----|
| Healthy | 0.999 |
| Degraded | 1.000 |
| Failed | 0.995 |
| Unreachable | 1.000 |
| Oscillating | 0.972 |
| **Macro** | **0.993** |

### Demo Scenarios (training-aligned features, unseen topologies)
| Scenario | Accuracy | Description |
|----------|----------|-------------|
| Monday Morning | 0.957 | 25 ticks, progressive cascade with recovery |
| Silent Killer | 0.955 | Hub fails silently, downstream inference |
| Cascade Whiplash | 0.963 | Root recovers, dependents stay dead |
| Slow Poison | 0.985 | Gradual 5%/tick degradation |
| Random Chaos | 0.912 | Multiple staged failures |

### Adversarial Stress Test (harder-than-training fog-of-war, fresh seeds)
| Scenario | Macro F1 |
|----------|----------|
| Oscillator | 0.711 |
| Slow Poison | 0.547 |
| Whiplash | 0.587 |
| FP Storm (40% noise) | 0.390 |
| Silent Killer | 0.520 |
| Avalanche | 0.521 |
| Monday Morning | 0.836 |
| **Average** | **0.587** |

### Break Test (20 adversarial attack scenarios)
| Result | Count |
|--------|-------|
| Survived | 16 |
| Vulnerable | 1 |
| Broken | 3 (trust boundary — compromised inputs) |
| Crashed | 0 |

**678 inferences/sec sustained. Zero memory leaks. Zero numerical failures.**

---

## File Inventory

### Core Architecture
```
sable_sim/core/states.py              — N_STATES=5, STATE_NAMES, STATE_MAP (single source of truth)
sable_sim/core/component.py           — ComponentState enum (4 sim states)
sable_sim/core/state.py               — SystemState, StateChange
sable_sim/core/graph.py               — InfrastructureGraph
sable_sim/simulation/propagation.py   — PropagationEngine (cascade physics)
sable_sim/simulation/failure_injection.py — FailureInjector
sable_sim/simulation/fog.py           — FogOfWar (partial observability)
```

### Pillar 1: GNN
```
pillar1/cortex_gnn_model.py           — SableGNN (GATv2Conv backbone, multi-task heads)
pillar1/build_infra_dataset.py        — Real topology → PyG (Topology Zoo + Microservices)
pillar1/train_infra_gnn.py            — Train GNN on infrastructure topologies
pillar1/train_fb15k.py                — Pre-train on FB15k-237
pillar1/train_biokg.py                — Pre-train on ogbl-biokg
pillar1/domain_portability_test.py    — InfrastructureAdapter (component → GNN features)
pillar1/create_infra_gnn.py           — Transfer pretrained backbone to infra dims
pillar1/cortex_gnn_export.py          — CORTEX knowledge graph → PyG export
pillar1/gnn_server.py                 — HTTP inference server (port 5070)
```

### Pillar 2: POMDP
```
pillar2/pomcp.py                      — POMCPSolver, BeliefState, ActionGenerator
pillar2/hard_scenarios.py             — Diagnostic test scenarios
```

### Pillar 3: Mamba
```
pillar3/sable_mamba.py                — SelectiveSSM, MambaBlock (stateful h_init/h_final)
pillar3/sable_mamba_final.py          — SableMambaFinal (cascade predictor, forward_stateful)
pillar3/generate_temporal_data.py     — sable_sim → temporal training data
pillar3/generate_temporal_data_v3.py  — v3 cascade outcome format
pillar3/train_powergraph.py           — Train on PowerGraph real cascade data
```

### Fusion
```
fusion/staged_fusion_v3.py            — SharpRoutedFusion (hard argmax, router supervision)
fusion/temporal_chain.py              — TemporalState, ContextMixer, RevisionGate, TemporalChainFusion
fusion/shared_latent_space.py         — SharedStateSpace, projections, generate_fusion_data
fusion/train_temporal.py              — Train temporal chain (BPTT, noise hardening)
fusion/generate_temporal_sequences.py — Per-tick pillar features from sable_sim
fusion/adversarial_test.py            — 7-scenario stress test
fusion/break_sable.py                 — 20-scenario destruction test
```

### Production Server
```
docker/server.py                      — FastAPI + WebSocket (asyncio.Lock, run_in_executor)
docker/sable_engine.py                — SableEngine (stateful inference, recommendations)
docker/dashboard.html                 — Single-file b1tr0n1n branded UI
docker/precompute_scenarios.py        — Training-aligned scenario generation
docker/build.sh                       — Docker build pipeline
docker/run.sh                         — Docker launch
docker/Dockerfile                     — Container definition
```

### Orchestrator
```
orchestrator/sable_core.py            — Three-pillar integration
orchestrator/complex_scenario.py      — Monday Morning Meltdown (42-node)
orchestrator/cortex_diagnostic.py     — CORTEX knowledge graph reasoning
orchestrator/cortex_repair.py         — Graph structure repair
```

### Checkpoints
```
pillar1/checkpoints/best_model.pt             — Infrastructure GNN (5-class, 1.4M params)
pillar1/checkpoints_fb15k/best_fb15k.pt       — FB15k-237 pretrained (Link AUC 0.957)
pillar1/checkpoints_biokg/best_biokg.pt       — ogbl-biokg pretrained (Link AUC 0.960)
pillar3/checkpoints_powergraph_ieee39/        — PowerGraph IEEE39 (Affected F1 0.999)
pillar3/checkpoints_powergraph_ieee118/       — PowerGraph IEEE118 (Affected F1 0.9995)
fusion/checkpoints/staged_fusion_v3.pt        — Sharp routed fusion (297K params)
fusion/checkpoints/temporal_chain.pt          — Temporal chain (74K params)
docker/checkpoints/fusion.pt                  — Deployed fusion weights
docker/checkpoints/temporal.pt                — Deployed temporal weights
```

### Training Data
```
pillar1/data/infra_graphs.pt                  — 5,730 scenarios from real topologies (152 MB)
pillar1/data/infra_topo/topology_zoo/         — 261 GML files from Topology Zoo
pillar1/data/infra_topo/microservices/        — 20 GraphML files from MicroserviceDataset
pillar3/data/PowerGraph/dataset_cascades/     — IEEE24, IEEE39, IEEE118, UK cascade data
fusion/infra_fusion_data.pt                   — 10K aligned fusion scenarios
fusion/temporal_sequences.pt                  — 4K temporal sequences (3.6 GB)
docker/scenarios/*.pt                         — 5 pre-computed demo scenarios (3.3 MB)
```

---

## How to Run

### Dashboard (bare metal — uses RTX 5090)
```bash
source /home/b1tr0n1n/ml-env/bin/activate
cd /mnt/vault/sable/docker
python server.py
# Open http://localhost:8080
```

### Dashboard (Docker — CPU fallback, no sm_120 support yet)
```bash
cd /mnt/vault/sable/docker
./build.sh
./run.sh
# Open http://localhost:8080
```

### Regenerate Demo Scenarios
```bash
source /home/b1tr0n1n/ml-env/bin/activate
cd /mnt/vault/sable/docker
python precompute_scenarios.py
```

### Retrain Full Pipeline
```bash
source /home/b1tr0n1n/ml-env/bin/activate

# 1. Regenerate infrastructure graph data
cd /mnt/vault/sable/pillar1
python build_infra_dataset.py                          # 0.8s, 5,730 scenarios

# 2. Train GNN on infrastructure
python train_infra_gnn.py --epochs 80                  # ~12 min

# 3. Regenerate fusion data
cd /mnt/vault/sable
python -c "
import sys; sys.path.insert(0,'fusion'); sys.path.insert(0,'pillar1'); sys.path.insert(0,'pillar2'); sys.path.insert(0,'pillar3')
from shared_latent_space import generate_fusion_data
import torch
train_data, val_data = generate_fusion_data(count=10000, device='cpu')
torch.save({'train': train_data, 'val': val_data}, 'fusion/infra_fusion_data.pt')
"                                                       # ~9s

# 4. Train staged fusion
cd /mnt/vault/sable/fusion
python staged_fusion_v3.py --device cuda               # ~30s

# 5. Generate temporal sequences
python generate_temporal_sequences.py --count 4000 --ticks 20  # ~3s

# 6. Train temporal chain
python train_temporal.py --epochs 100 --window 4 --device cuda  # ~8 min

# 7. Copy checkpoints to docker
cp checkpoints/staged_fusion_v3.pt ../docker/checkpoints/fusion.pt
cp checkpoints/temporal_chain.pt ../docker/checkpoints/temporal.pt

# 8. Regenerate demo scenarios
cd /mnt/vault/sable/docker
python precompute_scenarios.py
```

### Run Adversarial Stress Test
```bash
source /home/b1tr0n1n/ml-env/bin/activate
cd /mnt/vault/sable/fusion
python adversarial_test.py
```

### Run Destruction Test
```bash
source /home/b1tr0n1n/ml-env/bin/activate
cd /mnt/vault/sable/fusion
python break_sable.py
```

---

## API Endpoints

```
GET  /                    — Dashboard HTML
GET  /api/status          — Engine state (device, scenario, cycle, n_nodes)
GET  /api/scenarios       — List available demo scenarios (cached)
POST /api/scenario        — Load scenario: {"name": "monday_morning"}
POST /api/tick            — Advance one inference cycle (returns full state)
POST /api/autoplay        — Start/stop automatic ticking: {"action": "start", "speed": 2.0}
POST /api/reset           — Restart current scenario from tick 0
GET  /api/node/{id}       — Full node detail (trajectory, state changes)
GET  /api/recommendations — Prioritized fix actions with root cause analysis
WS   /ws                  — Live streaming during autoplay
```

### Tick Response Format
```json
{
  "cycle": 10,
  "nodes": [
    {
      "id": 0,
      "state": "failed",
      "state_idx": 2,
      "confidence": 0.987,
      "probs": {"healthy": 0.003, "degraded": 0.005, "failed": 0.987, "unreachable": 0.004, "oscillating": 0.001},
      "transition": {"improving": 0.05, "stable": 0.10, "deteriorating": 0.85},
      "trend": "deteriorating",
      "routing": {"GNN": 0.0, "POMDP": 0.01, "Mamba": 0.01, "Fusion": 0.98}
    }
  ],
  "class_counts": {"healthy": 22, "degraded": 3, "failed": 5, "unreachable": 3, "oscillating": 2},
  "routing": {"healthy": {"expert": "Fusion", "weight": 0.98}},
  "avg_confidence": 0.934,
  "accuracy": 0.943,
  "inference_ms": 5.2
}
```

### Recommendations Response Format
```json
{
  "summary": "13/35 nodes affected: 5 failed, 3 unreachable, 2 degraded, 3 oscillating. Root cause likely Node 11.",
  "total_affected": 13,
  "root_cause": 11,
  "actions": [
    {"priority": 1, "action": "INVESTIGATE ROOT CAUSE", "target": "Node 11",
     "reason": "First failure at tick 0. 5 total failed nodes likely depend on this.",
     "recommendation": "Restart service, check hardware, review recent changes."},
    {"priority": 2, "action": "RESTART/RECOVER", "target": "Node 03", ...},
    {"priority": 3, "action": "CHECK CONNECTIVITY", "target": "Node 10", ...},
    {"priority": 4, "action": "STABILIZE", "target": "Node 06", ...},
    {"priority": 5, "action": "MONITOR/INVESTIGATE", "target": "Node 05", ...}
  ]
}
```

---

## Dashboard

Single HTML file (`docker/dashboard.html`). Four panels:

1. **Topology Map** — Node grid, state-colored cells, confidence as opacity, gold flash on transitions
2. **System Status** — Class distribution bars, router decisions per class, confidence stats
3. **Cascade Timeline** — Heatmap of node states over ticks, sorted by first change
4. **Node Detail** — Click any node for probabilities, trajectory, trend, expert routing

Controls: Play/Pause, Step, Speed (0.5x-5x), Scenario dropdown, Reset, **FIX** button (gold).

Brand: b1tr0n1n design system. Background `#0a0908`, gold `#c9a227`, Cormorant Garamond for prose, JetBrains Mono for data. No gradients, no shadows. Gold is earned.

---

## Security

- Supabase service_role key removed from all source files. Moved to `os.environ.get("SUPABASE_KEY")`.
- NaN/Inf sanitization at temporal chain entry point.
- `torch.load(weights_only=False)` used throughout — safe for self-generated checkpoints, document risk for external.
- asyncio.Lock on all shared mutable state in server.
- WebSocket has exponential backoff reconnect + onerror handler.

---

## Known Limitations

1. **Confidence on out-of-distribution data** — Model correctly reports uncertainty on unseen feature distributions. Max softmax probability is the honest signal. No post-hoc scaling.
2. **FP Storm degraded** — 40% observation noise kills degraded detection (0.000 F1). Extreme noise is a fundamental limit.
3. **sable_sim cascade speed** — Cascades resolve in 1-3 ticks. Real infrastructure fails over hours. Demo scenarios use controlled injection for longer dynamics.
4. **Docker sm_120** — No official PyTorch Docker image supports Blackwell GPUs yet. Container falls back to CPU. Bare metal is the correct deployment for sable.
5. **Oscillating mono-class** — If all 40 nodes oscillate simultaneously, model can't classify. Not a real scenario.

---

## Backups

```
/mnt/vault/sable-backup-session3     — Before Session 4 (5-class migration)
/mnt/vault/sable-backup-session4a    — Before training distribution fix
/mnt/vault/sable-backup-session4b    — Before Docker build
/mnt/vault/sable-backup-pre-audit    — Before code audit fixes
/mnt/vault/sable-backup-pre-session5 — Before Session 5 (confidence + propagation)
```

---

## Session History

| Session | Focus | Key Deliverable |
|---------|-------|-----------------|
| 1 | Foundation | GNN + POMDP + Mamba built, sable_sim operational |
| 2 | Integration | Orchestrator, Monday Morning Meltdown (42-node), dashboard |
| 3 | Real data | FB15k-237, ogbl-biokg, Topology Zoo, PowerGraph, staged fusion, temporal chain |
| 4 | Hardening | 5-class migration, adversarial testing, noise hardening, Docker build |
| 5 | Production | Code audit (34 fixes), confidence calibration, demo scenario alignment, recommendations engine |
