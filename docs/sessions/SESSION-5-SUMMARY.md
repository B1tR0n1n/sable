# Session 5 - Project Parallax Pt. 3

**Date:** 2026-03-29

## What Was Built

### Telemetry Adapter Package (`adapters/`)
- Prometheus adapter with auto-discovery, SNMP/Windows/Linux query dispatch
- Health scorer: raw metrics to health [0,1] + state classification
- Pillar encoder: SystemSnapshot to GNN (1044), POMDP (8-dim), Mamba (26-dim)
- Graph builder: topology from YAML or Netbox API
- SMD adapter: real server telemetry (28 machines, 38 metrics) into SABLE format
- Audited and fixed 15 bugs across the adapter package

### 1044 GNN Upgrade
- Replaced 6 CORTEX node types with 20 native infrastructure types
- No more lossy compression (CORE_SWITCH and FIREWALL get their own dimensions)
- Weight transfer from FB15k backbone, only input_proj reinitialized

### MC Dropout
- 5-sample stochastic forward passes at inference
- Returns per-node agreement + variance
- Measures actual model uncertainty, not softmax confidence
- Dashboard toggle button

### Operator Feedback API
- `/api/feedback` endpoint with SQLite store
- Dashboard UX: click node, select correct state
- Confusion stats, finetune readiness flag at 100+ corrections

### Cascade Dynamics
- Three PropagationEngine profiles: fast, slow, realistic
- Propagation delay, partial recovery, secondary failures
- Mixed profiles in temporal data generation

### Full Retrain
- GNN: 80% accuracy, 98% AUC on Topology Zoo + Microservice graphs
- Mamba: 49% affected F1 from 2 ticks of signal
- Fusion + Temporal Chain: 99.8% macro F1

### Real Server Telemetry (SMD)
- 28 production servers, 38 metrics each, 5000 ticks
- 33 scenarios generated (8 normal + 19 incidents + 6 cascades)
- Found and fixed label leakage (features contained ground truth)
- Honest base accuracy without adaptation: 60%

### LoRA Fine-Tuning
- 48,280 trainable parameters (11.4% of model)
- Base model completely frozen
- Trained on full system: fusion + temporal chain, sequential tick processing
- Fixed key loading order (must build full TemporalChainFusion before applying LoRA)
- Result: 92.4% accuracy on real server telemetry
- Temporal ramp: 82% at tick 0, 96% by tick 5

### LoRA Adapter Toggle
- Runtime enable/disable, no model reload
- ADAPTER ON: 90.4% avg accuracy
- ADAPTER OFF: 57.2% avg accuracy
- The before/after demo in one button

### Nemotron Integration
- Chat interface in dashboard - operator asks questions about findings
- SABLE provides structured diagnostic data as context
- Nemotron reasons about likely causes, fix steps, downstream risks
- Grounded in SABLE data but can apply infrastructure expertise
- SABLE: 618MB GPU, 5ms inference. Nemotron: 13.8GB, ~3 seconds

### Project Reorganization
- Docs moved to `docs/`, dead code to `archive/` subdirs
- `__init__.py` added to all packages
- Docker engine mirror deleted (10K lines of stale copies)
- Sonar audit: 200+ issues fixed across 47 files

## Key Bugs Caught

1. **Label leakage in SMD encoding** - Mamba features contained the ground truth state. Model was reading the answer, not predicting. Fixed by using different health formula for features vs ground truth.

2. **LoRA key mismatch** - LoRA applied to base fusion before wrapping in TemporalChainFusion. Key names like `base_fusion.gnn_expert...` didn't exist yet. Fixed by building full model first, then applying LoRA.

3. **Temporal chain fighting LoRA** - Training fusion LoRA and temporal chain LoRA separately caused them to overcorrect each other. Fixed by training the full system end-to-end with sequential tick processing.

## Architecture Insights

**Confidence/accuracy gap is the sales pitch.** Base model on unfamiliar data: 98% confident, 57% accurate. With LoRA: 98% confident, 93% accurate. The LoRA doesn't change confidence - it changes what the model is confident about.

**SABLE + LLM division of labor.** SABLE does structured reasoning on structured data (5ms). LLMs do natural language communication (~3s). Both doing what their architecture is built for. Frontier models can't do graph reasoning with temporal state - wrong architecture, not wrong capability.

**Temporal chain needs enough data.** With 4 scenarios (1 validation), it overfits. With 33 scenarios (9 validation), it generalizes. The temporal context mixer and revision gate need variety to learn when to trust the trajectory vs the current prediction.

## Files Changed
- 47+ files modified in sonar audit alone
- New: `adapters/` (6 files), `adapters/smd.py`, `fusion/lora_finetune.py`, `docker/nemotron_bridge.py`, `docker/precompute_smd.py`
- Modified: `sable_engine.py`, `server.py`, `dashboard.html`, `temporal_chain.py`, `propagation.py`, `generate_temporal_data.py`, `domain_portability_test.py`, `create_infra_gnn.py`
- Deleted: `docker/engine/` (36 files, 10K lines)

## How to Run

```bash
# SABLE engine (port 8080)
cd /mnt/vault/sable/docker && source ~/ml-env/bin/activate && python3 server.py

# Nemotron chat (port 8081) - optional
~/llama.cpp/build/bin/llama-server -m /mnt/vault/models/nemotron-nano/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf --ctx-size 4096 --temp 0.4 -ngl 999 --port 8081
```

Open http://localhost:8080. Select an SMD scenario. Hit play. Toggle ADAPTER to show before/after. Hit ASK to chat with Nemotron about findings.

## What's Next

Everything on the pre-MSP checklist from SABLE_PLANFULL.md is done. The remaining work requires the MSP environment:
1. Point Prometheus adapter at a real instance
2. Write topology YAML for customer environment
3. Fine-tune with LoRA on their labeled data (5 seconds, 48K params)
4. Deploy
