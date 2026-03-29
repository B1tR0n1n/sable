# Project PARALLAX — Session 3 Summary
**Date:** 2026-03-24/25
**Duration:** ~5 hours
**Focus:** Real data training, staged fusion, temporal chaining

---

## Mission

Train both pillars on real datasets (not synthetic/wrong-domain), validate staged fusion with all three pillars at genuine strength, then build the temporal chaining layer — the verdict feedback loop that gives the system memory across inference cycles.

---

## Phase 1: Real Data Training

### Problem
- GNN (Pillar 1) was trained on CORTEX knowledge graph (philosophy, AI, games) — 4.8% accuracy on infrastructure. Wrong domain entirely.
- Mamba (Pillar 3) was trained on synthetic sable_sim data — decent but not real cascade physics.

### Research
Evaluated 10+ datasets per pillar. Selected based on task alignment, accessibility, and structural fit.

### GNN Training Pipeline

| Stage | Dataset | Nodes | Edges | Relations | Key Metric | Time |
|-------|---------|-------|-------|-----------|------------|------|
| Validate | FB15k-237 (PyG) | 14,541 | 272,115 | 237 | Link AUC **0.957**, Type Acc **0.915** | 2 min |
| Pretrain | ogbl-biokg (OGB) | 93,773 | 5,088,434 | 51 | Link AUC **0.960**, MRR **0.186** | 2 min |
| Fine-tune | Topology Zoo + Microservices | 191 real topologies | 5,730 scenarios | 6 | Macro F1 **0.770**, Failed F1 **1.000**, Link AUC **0.989** | 23 min |

**Topology Zoo:** 261 real ISP/enterprise network topologies in GML format. Downloaded from topology-zoo.org. Nodes are routers/switches with geographic data, edges have LinkType attributes (OC-192, Ethernet, DWDM, etc.).

**Microservice Dataset:** 20 real service dependency graphs from open-source architectures (eShopOnContainers, Spinnaker, etc.) in GraphML format.

**Architecture adaptation:** SableGNN with 10-dim node features (8 type one-hot + degree_norm + health) and 7-dim edge features (6 type one-hot + weight). Hidden dim 128, 3 GATv2Conv layers, 4 heads. Trained with link prediction + node state prediction multi-task loss.

### Mamba Training Pipeline

| Stage | Dataset | Scenarios | Nodes | Key Metric | Time |
|-------|---------|-----------|-------|------------|------|
| Primary | PowerGraph IEEE39 | 28,000 | 39 buses | Affected F1 **0.999**, Macro F1 **0.974** | 28 sec |
| Scale | PowerGraph IEEE118 | 122,500 | 118 buses | Affected F1 **0.9995**, Macro F1 **0.989** | 3.5 min |

**PowerGraph (NeurIPS 2024):** Real power grid cascading failure simulations. Data in MATLAB v7.3 (HDF5) format — Bf.mat (bus features), Ef.mat/Ef_nc.mat (pre/post cascade edge features), blist.mat (topology), labels (binary, 4-class, regression). Downloaded from Figshare.

**Data mapping:** Tick 0 = pre-cascade bus features + edge aggregation. Tick 1 = post-cascade with tripped lines zeroed. Per-node state derived from edge health changes.

---

## Phase 2: Staged Fusion

### Fusion Data Pipeline
`generate_fusion_data()` in shared_latent_space.py generates 10K aligned scenarios in ~9 seconds. Each scenario runs all three pillars on a sable_sim infrastructure cascade:
- GNN: encodes topology graph → 128-dim node embeddings (padded to 256)
- POMDP: belief state → 8-dim per node
- Mamba: raw state features → 25-dim per node
- Ground truth: 4-class per node (healthy/degraded/failed/unreachable)

### Fusion v1 → v3 Evolution

| Version | Change | Macro F1 | vs Best Expert |
|---------|--------|----------|----------------|
| v1 | Basic staged (old GNN) | 0.536 | +4.6% |
| v2 | Deeper expert heads | 0.544 | +4.7% |
| v2 + infra GNN | Real GNN embeddings | 0.566 | +8.1% |
| **v3** | **Sharp router + hard routing + router supervision** | **0.653** | **+16.8%** |

### v3 Router Analysis
The router learned correct specialization:

| Class | Router Choice | Routed F1 | Fusion-only F1 |
|-------|--------------|-----------|----------------|
| Healthy | Fusion (83%) | 0.959 | 0.861 |
| Degraded | Fusion (100%) | 0.964 | 0.964 |
| Failed | Fusion (98%) | 0.450 | 0.450 |
| Unreachable | POMDP (54%) | 0.239 | 0.103 |

**Key fix in v3:** Hard routing at eval (argmax, not softmax blend). Router supervision loss that teaches the router which expert is actually correct per-node. Removed balance loss — let the router specialize aggressively.

**Result:** Routed (0.653) > Fusion-only (0.595) > Best Expert (0.485). The router adds value — it routes unreachable to POMDP which is the only expert that can detect it.

---

## Phase 3: Temporal Chaining

### Architecture

The temporal chain wires the fusion output back as input for the next inference cycle. Verdict at t becomes context at t+1.

**TemporalState** (~160 KB per topology):
- `mamba_h`: SSM hidden states per MambaBlock layer — list of (1, 512, 16)
- `prev_z`: fused latent Z from previous cycle — (1, N, 128)
- `trajectory`: ring buffer of last 8 verdicts per node — (N, 8, 5) where each entry = [state, confidence, cycle_norm, changed, direction]
- `belief_state`: POMDP belief vectors — (N, 8)

**ContextMixer** (before fusion): Augments pillar inputs with trajectory summary + prev_z via residual gating. Gate initialized near-zero (bias=-3) so model starts equivalent to stateless. Three TemporalGate modules with 32-dim bottleneck. **~50K params.**

**RevisionGate** (after fusion): Compares current verdict against trajectory history. Produces revised logits, confidence, and transition direction (improving/stable/deteriorating). Blends revision with current based on learned confidence. **~22K params.**

**TemporalChainFusion**: Wraps SharpRoutedFusion without modifying it.
```
ContextMixer(pillar_outputs, temporal_state)
    → SharpRoutedFusion(augmented_outputs)   ← existing, untouched
        → RevisionGate(verdict, trajectory)
            → Updated TemporalState for t+1
```

**Total new parameters: 74,186.** System: 9.2M → 9.27M.

### Mamba State Persistence
Modified `SelectiveSSM._selective_scan()` to accept `h_init` and return `h_final`. Added `forward_stateful()` to SableMambaFinal. Zero new parameters — pure plumbing. Hidden state `h: (B, d_inner, d_state)` now carries across inference cycles instead of resetting to zeros.

### Training
- Data: 3,000 temporal sequences from sable_sim (12 ticks max, avg 3 ticks)
- Hardened fog-of-war: 30% observability, 15% noise, 2-tick observation delay, health noise σ=0.15, 20-25% health masking
- Training: 4-step BPTT windows, freeze base fusion, train only temporal components
- Loss: L_state + 0.3 * L_transition
- AdamW lr=5e-4, cosine schedule, early stopped at epoch 65

### Validation Results (Hardened)

| Configuration | Healthy | Degraded | Failed | Unreachable | Macro F1 |
|--------------|---------|----------|--------|-------------|----------|
| Base fusion (no temporal) | 0.980 | 0.977 | 0.667 | 0.129 | **0.688** |
| Temporal — cold start t=0 | 1.000 | 1.000 | 0.999 | 0.000* | **0.750** |
| Temporal — autoregressive | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |

*Unreachable has 0 samples at t=0 — it only emerges from cascade propagation.

**Delta: +0.312 macro F1 from temporal chaining over base fusion.**

### Honest Assessment

The 1.000 autoregressive score is real in three senses:
1. Cold start (0.750) ≠ full chain (1.000) — temporal context genuinely helps
2. Unreachable goes from 0.000 (can't exist at t=0) to 1.000 — the chain tracks cascade emergence
3. +0.312 over base fusion is legitimate improvement

The 1.000 is inflated by:
1. Short cascades (avg 3 ticks) — limited time for errors to compound
2. Class imbalance (91% healthy) — POMDP priors aligned with reality by default
3. POMDP belief observations still carry signal even at 30% observability
4. Task is easier than real-world adversarial conditions

**The architecture works. The engineering isn't done.** On harder scenarios (longer cascades, oscillating nodes, adversarial conditions), the score will drop. The temporal chain adds genuine value — the question is how much degrades under real operational stress.

---

## Files Created/Modified

### New Files
```
pillar1/train_fb15k.py              — GNN pretraining on FB15k-237
pillar1/train_biokg.py              — GNN pretraining on ogbl-biokg (with torch.load patch)
pillar1/build_infra_dataset.py      — Real topology → PyG converter (Topology Zoo + Microservices)
pillar1/train_infra_gnn.py          — GNN fine-tuning on infrastructure topologies
pillar1/create_infra_gnn.py         — Transfer pretrained backbone to infra dimensions
pillar1/data/infra_topo/            — Downloaded Topology Zoo (261 GML) + Microservice Dataset (20 GraphML)
pillar1/data/infra_graphs.pt        — 5,730 cascade scenarios from real topologies (152 MB)

pillar3/train_powergraph.py         — Mamba pretraining on PowerGraph cascades
pillar3/data/PowerGraph/            — Downloaded PowerGraph dataset (IEEE24/39/118/UK)
pillar3/powergraph_ieee39_ieee39.pt — Converted IEEE39 dataset (75.8 MB)

fusion/staged_fusion_v2.py          — Improved staged fusion with deeper experts
fusion/staged_fusion_v3.py          — Sharp router + hard routing + router supervision
fusion/temporal_chain.py            — TemporalState, ContextMixer, RevisionGate, TemporalChainFusion
fusion/generate_temporal_sequences.py — Per-tick pillar features from sable_sim cascades
fusion/train_temporal.py            — Train temporal chain on BPTT windows
fusion/infra_fusion_data.pt         — 10K aligned fusion scenarios
fusion/temporal_sequences.pt        — 3K temporal sequences (1.6 GB)
```

### Modified Files
```
pillar3/sable_mamba.py              — SelectiveSSM: h_init/h_final params, MambaBlock: state passthrough
pillar3/sable_mamba_final.py        — Added forward_stateful() with SSM state persistence
fusion/shared_latent_space.py       — Updated generate_fusion_data for infra-native GNN features
```

### Checkpoints
```
pillar1/checkpoints_fb15k/best_fb15k.pt         — FB15k-237 pretrained GNN
pillar1/checkpoints_biokg/best_biokg.pt          — ogbl-biokg pretrained GNN
pillar1/checkpoints/best_model.pt                — Infrastructure fine-tuned GNN
pillar3/checkpoints_powergraph_ieee39/            — PowerGraph IEEE39 Mamba
pillar3/checkpoints_powergraph_ieee118/           — PowerGraph IEEE118 Mamba
fusion/checkpoints/staged_fusion_v3.pt            — Sharp routed fusion
fusion/checkpoints/temporal_chain.pt              — Temporal chain (trained)
```

---

## Architecture State

```
                    SABLE Cognitive Architecture
                    ============================

    Pillar 1: GNN                  Pillar 2: POMDP              Pillar 3: Mamba
    ─────────────                  ──────────────               ──────────────
    GATv2Conv encoder              POMCP tree search            SelectiveSSM
    Trained on:                    CPU-based                    Trained on:
    • FB15k-237 (237 rels)         Bayesian belief updates      • PowerGraph IEEE39
    • ogbl-biokg (93K nodes)       Observation history          • PowerGraph IEEE118
    • Topology Zoo (191 topos)     Structural contradiction     • sable_sim cascades
    • Microservice graphs          detection
                                                                Now with h_init/h_final
    Macro F1: 0.770                                             state persistence
    Link AUC: 0.989
    Failed F1: 1.000
         │                              │                            │
         └──────────────┬───────────────┘                            │
                        │                                            │
                   ┌────▼────────────────────────────────────────────▼───┐
                   │              Staged Fusion v3                       │
                   │  Sharp Router + Hard Routing + Router Supervision   │
                   │  Macro: 0.653 | Routed > Fusion > Best Expert      │
                   └────────────────────┬───────────────────────────────┘
                                        │
                   ┌────────────────────▼───────────────────────────────┐
                   │              Temporal Chain                         │
                   │  ContextMixer → Fusion → RevisionGate              │
                   │  Verdict t → Input t+1 | Trajectory tracking       │
                   │  74K params | +0.312 macro over base               │
                   │  Autoregressive: 1.000 (hardened fog-of-war)       │
                   └────────────────────────────────────────────────────┘
```

---

## Total Parameter Count

| Component | Parameters |
|-----------|-----------|
| GNN encoder (infra) | 1,366,924 |
| Mamba cascade predictor | 3,658,993 |
| POMDP solver | 0 (CPU tree search) |
| Staged Fusion v3 | 297,236 |
| Temporal Chain | 74,186 |
| **Total** | **~5.4M active** |

All fits on a single RTX 5090 with room to spare.

---

## Next Steps

1. **Adversarial validation** — longer cascades (20+ ticks), oscillating nodes, recovery scenarios, adversarial observation patterns. Push until the 1.000 breaks, then see where it settles.
2. **Monday Morning Meltdown** — run the 42-node enterprise scenario through the full temporal chain. Track the cascade progression across diagnostic steps.
3. **Orchestrator integration** — wire `diagnose_temporal()` into sable_core.py with live Mamba state persistence and POMDP belief continuity.
4. **Real-time inference loop** — the temporal chain is designed for streaming inference. Build the loop that continuously ingests telemetry, updates temporal state, and emits verdicts.
