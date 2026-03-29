# SABLE

Three-pillar cognitive architecture for infrastructure diagnostics.

A 5.4M parameter system that classifies infrastructure node states through three specialized neural pillars fused with a temporal feedback loop. Runs at 5ms per inference on a single RTX 5090.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         Fusion Layer             │
                    │   Cross-Attention + Routed Expert│
                    │   + Temporal Chain               │
                    └──────┬──────┬──────┬────────────┘
                           │      │      │
              ┌────────────┘      │      └────────────┐
              ▼                   ▼                    ▼
     ┌────────────────┐ ┌────────────────┐ ┌──────────────────┐
     │   Pillar 1     ��� │   Pillar 2     │ │    Pillar 3      │
     │   GNN          │ │   POMDP        │ │    Mamba/SSM     │
     │                │ │                │ │                   │
     │   Structural   │ │   Decision     │ │   Temporal        │
     │   Reasoning    │ │   Planning     │ │   Prediction      │
     └────────────────┘ └────���───────────┘ └──────────────────┘
              │                   │                    │
              └───────────────────┴──────���─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      sable_sim             │
                    │  Infrastructure Simulator  │
                    │  20 component types        │
                    │  9 dependency types         │
                    │  Cascade propagation       │
                    │  Fog-of-war observability  │
                    └───────────────────────────┘
```

**Pillar 1 — GNN (Structural Reasoning):** EdgeConditionedGAT over infrastructure topology. 1044-dim node features (1024 embedding + 20 native infrastructure type one-hot). Learns dependency propagation patterns, contradiction detection, link prediction.

**Pillar 2 — POMDP (Decision Planning):** Monte Carlo tree search under partial observability. Given fog-of-war constraints (75% monitoring coverage, 2-tick delay, 5% false positives), plans diagnostic actions to maximize information gain toward root cause.

**Pillar 3 — Mamba/SSM (Temporal Prediction):** Selective state space model for cascade forecasting. Takes 2-tick system snapshots, predicts 30-tick cascade outcomes — which nodes will be affected, predicted severity, state trajectories.

**Fusion:** Cross-attention weighted combination of three pillar perspectives with routed expert selection per node. Temporal chain maintains state across inference cycles for multi-step reasoning.

## Project Structure

```
sable/
├── sable_sim/          # Infrastructure simulator (foundation)
├── pillar1/            # GNN — structural reasoning
├── pillar2/            # POMDP — decision planning
├── pillar3/            # Mamba — temporal prediction
├── fusion/             # Three-pillar fusion + temporal chain
├── orchestrator/       # Integration layer
├── adapters/           # Telemetry adapters (Prometheus, health scoring, encoding)
│   └── topologies/     # Infrastructure topology configs (YAML)
├── docker/             # Engine demo server + dashboard
├── dashboard/          # CORTEX knowledge graph dashboard (React)
├── docs/               # Documentation + session history
└── archive/            # Deprecated scripts
```

## Running the Engine Demo

```bash
cd docker
source ~/ml-env/bin/activate
python server.py
```

Open http://localhost:8080. Select a scenario, hit play, watch the cascade unfold. Click nodes to inspect state probabilities, trajectories, and routing decisions. Hit the fix button for prioritized remediation recommendations.

## Telemetry Adapters

The `adapters/` package converts real monitoring data into SABLE's pillar input formats:

```python
from adapters import HealthScorer, PillarEncoder, SystemSnapshot
from adapters.prometheus import PrometheusAdapter, PrometheusConfig

adapter = PrometheusAdapter(PrometheusConfig(url="http://prometheus:9090"))
scorer = HealthScorer()
encoder = PillarEncoder()

snapshot = adapter.poll_with_health(scorer)
inputs = encoder.encode(snapshot)
# inputs.gnn  → [N, 1044] node features
# inputs.pomdp → dict[node_id → 8-dim belief]
# inputs.mamba → [1, 2, 1040] temporal input
```

## Requirements

- Python 3.11+
- PyTorch 2.x with CUDA
- torch-geometric
- numpy, networkx, requests
- FastAPI + uvicorn (for server)

## Hardware

Developed and tested on:
- AMD Ryzen 9 9950X3D
- NVIDIA RTX 5090 (32GB VRAM)
- 64GB DDR5-6000
