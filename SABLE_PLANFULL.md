# SABLE — Full Production Roadmap

## Session Status: Parallax Pt. 3 (2026-03-27)

---

## COMPLETED THIS SESSION

### 1044 GNN Upgrade
- `domain_portability_test.py` — `InfrastructureAdapter` now uses native 20-type one-hot (was 6 CORTEX types compressed to 6)
- `create_infra_gnn.py` — targets 1044 dims (1024 embedding + 20 infrastructure type one-hot)
- `create_infra_gnn_1044.py` — standalone upgrade script preserved as reference
- Docker mirror synced (`docker/engine/pillar1/domain_portability_test.py`)
- Verified: adapter outputs `(1044,)` with correct per-type indices, no more CORE_SWITCH/FIREWALL collision

### Telemetry Adapter Package (`sable/adapters/`)
- `base.py` — `TelemetryAdapter` interface, `NodeSnapshot`, `EdgeSnapshot`, `SystemSnapshot`, `Criticality`
- `health_scorer.py` — raw metrics → health [0,1] + state classification, threshold-based with per-customer config, oscillation detection, binary metric handling
- `prometheus.py` — Prometheus/VictoriaMetrics adapter with auto-discovery from targets API, SNMP/Windows/Linux query dispatch, node type inference from job labels + sysDescr
- `encode.py` — `PillarEncoder` converting `SystemSnapshot` → GNN [N, 1044], POMDP [N, 8] belief, Mamba [1, 2, 1040] temporal inputs
- `graph_builder.py` — topology ingestion from YAML/JSON config or Netbox API
- Full pipeline verified end-to-end: Prometheus → scorer → encoder → pillar tensors

### Audit + Bug Fixes (15 items)
1. Health cliff at degraded boundary → smooth interpolation (1.0 → 0.8 → 0.0)
2. State/health disagreement → state derived from composite health, single source of truth
3. Metric clobbering on multi-series PromQL → `max by (instance)` aggregation + first-write guard
4. False unreachable when `up` query fails → only when zero metrics received for a node
5. Non-deterministic GNN embeddings per poll → `md5(node_id)`-seeded perturbation
6. YAML import crash without PyYAML → lazy import with clear error message
7. `sys.path.insert` at module level → moved to lazy-load property on `PillarEncoder.gnn_adapter`
8. Windows queries never dispatched → exporter-type classification + separate query routing
9. Node ID collision silent → warning logged on duplicate mappings
10. POMDP belief too confident for unhealthy nodes → `_health_to_belief()` with smooth state distributions
11. `node_ids` property non-deterministic → returns `sorted()`
12. Dead code `DEP_TO_RELATION`, `CRIT_TO_CONFIDENCE` → removed
13. Redundant `metric_points` field → removed from `NodeSnapshot`
14. Unused `COMPONENT_TYPES` import in health_scorer → removed
15. Unused imports in prometheus (`MetricPoint`, `COMPONENT_TYPES`, `COMPONENT_TYPE_INDEX`) → removed

### Sample Topology
- `adapters/topologies/msp_demo.yaml` — 32 nodes, 57 edges, all 20 component types, 8 dependency types, 9 tiers
- Three-tier network (core/distribution/access), compute (hypervisors + VMs), storage (SAN), services (DNS/AD/DHCP/CA/monitoring), VDI, ERP application
- Models a real small-to-mid enterprise environment

### Engine UI Topology Integration
- `docker/server.py` — `/api/topology` endpoint, topology loaded at startup, `enrich_tick()` and `enrich_recommendations()` inject infrastructure labels/types/IDs into all engine outputs
- `docker/dashboard.html`:
  - Node grid shows component labels + types colored by functional role
  - Node detail panel shows topology metadata + full dependency list with direction arrows and criticality coloring
  - Timeline uses infrastructure IDs (e.g., `core-sw-1`) instead of bare numbers
  - Recommendations overlay shows real component names + types (e.g., "INVESTIGATE ROOT CAUSE → Core Switch 1 [CORE_SWITCH]")
- CORTEX dashboard (`dashboard/`) untouched — reverted to exact original state

---

## REMAINING — 6-STEP PRODUCTION PLAN

### Step 1: Ship Current Demo as Proof of Architecture

**Retrain GNN on 1044 dims**
- The checkpoint is still 1030. Run `create_infra_gnn.py` (now targeting 1044), then retrain on Topology Zoo + Microservice dataset via `train_infra_gnn.py`
- Attention weights transfer from 1030 — only `input_proj` resets (1024 embedding columns copy, 20 type columns reinitialize)
- Effort: half a day

**Retrain fusion pipeline against new GNN checkpoint**
- Fusion reads `in_dim` from checkpoint config dynamically, but fusion training data was generated with 1030 features
- Need a fresh `generate_fusion_data` run + fusion retrain with 1044-dim GNN outputs
- Effort: half a day (data gen + training)

**Pre-compute new scenarios for Docker demo**
- `precompute_scenarios.py` needs to generate scenarios with 1044-dim GNN features
- Current scenarios in `docker/scenarios/` (monday_morning, cascade_whiplash, etc.) are 1030-dim
- Effort: 1 hour once models are retrained

**Pitch artifact**
- One-pager or short deck translating the demo into business value
- Target audience: VP of Ops at an MSP customer, not technical
- Content: what Parallax does, what class of failures it catches that current tools don't, why the architecture is differentiated (three-pillar fusion, temporal cascade prediction, honest confidence)
- Not marketing — a clear operational value statement
- Effort: 2-3 hours

### Step 2: Land One Design Partner with Real Monitoring Data
- Keith handling through MSP role — direct access to multi-tenant infrastructure
- Requirements: NDA at minimum (possibly BAA given healthcare context), read access to monitoring stack for one environment (even a lab), champion inside the org frustrated with current tooling
- No engineering work needed — this is relationship/business development

### Step 3: Build the Telemetry Adapter — DONE
- Prometheus adapter: built and audited
- Health scorer: built and audited
- Pillar encoder: built and audited
- Graph builder (YAML + Netbox): built and audited

**Remaining sub-items:**
- **SNMP adapter** (`adapters/snmp.py`) — for network gear not covered by Prometheus snmp_exporter natively. Lower priority since Prometheus + snmp_exporter covers most MSP environments. Build when you see the actual monitoring stack.
- **Integration test against real Prometheus** — can't test until you have access to one. The adapter is structurally complete but untested against live PromQL responses.

### Step 4: Fine-Tune on Real Data (1 week)
- Blocked on Step 2 (design partner with real data)
- Fine-tuning pipeline infrastructure exists: LoRA on fusion heads + temporal chain with frozen backbone
- **When data arrives, build:** data loader that reads from telemetry adapter's `SystemSnapshot` format, converts to training batches, runs fine-tuning pass
- Target: 2 weeks of real monitoring data, validate on a third held-out week
- Effort: 1 week once data is available

### Step 5: Monte Carlo Dropout for Honest Confidence (1 day)
- **Not started. No data dependency — can build now.**
- Add `mc_samples` parameter to `SableEngine.infer()`
- Run N forward passes (default 5) with dropout enabled at inference time
- Return mean prediction + variance as calibrated confidence
- Already have dropout in fusion layers — just need the inference-time flag
- ~25ms instead of 5ms per inference (still real-time)
- Implement as `SableEngine.infer(mc_samples=5)`
- Calibration pass against held-out incidents needed once real data arrives (Step 4)
- Effort: half a day for implementation, few extra days for calibration with real data

### Step 6: Operator Feedback Loop for Continuous Improvement (1 week)
- **Not started. No data dependency for schema/API design — can build now.**

**Feedback schema:**
```json
{
  "node_id": "core-sw-1",
  "correct_state": "degraded",
  "timestamp": 1711584000,
  "operator_id": "kburns"
}
```

**Components to build:**
- `/api/feedback` endpoint in `docker/server.py`
- Local correction store (SQLite or flat JSON file)
- Fine-tuning trigger: when corrections reach threshold (100+), run LoRA pass on fusion heads + temporal chain with frozen backbone
- Dashboard UX: click a node, override its state, submit correction
- Effort: API endpoint (1 day), fine-tuning loop (1 day), dashboard UX (1 day), operational workflow design (ongoing)

---

## REMAINING — MODEL IMPROVEMENT (NOT IN 6-STEP BUT NEEDED)

### Diverse Training Topologies
- Topology Zoo (261 real ISP networks) and Microservice Dataset (20 real service graphs) already available in `pillar1/data/infra_topo/`
- Generate 10x more cascade scenarios on those real topologies — different failure points, cascade depths, recovery patterns
- Train on the full diversity of real graph structures, not just random generation from `build_random_topology`
- **Can do now** — no real data dependency
- Effort: 1-2 days. Infrastructure is built. Data generation run + retraining pass.

### Longer Cascade Sequences
- Mamba trained on 3-tick sequences. Real infrastructure failures play out over hours.
- Modify `PropagationEngine` for slower cascade dynamics:
  - Delayed propagation (not instant)
  - Gradual health decay (not binary state flips)
  - Intermittent recovery
  - Secondary failure after partial recovery
- Generate 15-20 tick cascade sequences with 10-15 tick recovery phases
- Retrain temporal chain on longer-horizon patterns
- **Can do now** — no real data dependency
- Effort: ~1 week. Engine changes moderate, retraining is a day.

### Docker Image Update
- Docker engine mirrors need syncing with 1044-dim changes
- `precompute_scenarios.py` needs to generate scenarios with 1044-dim GNN features
- Docker build/run scripts may need updating for new dependencies
- Effort: 1-2 hours once models are retrained

### Dashboard Polish
- Topology overlay works but the 32-node MSP demo has more nodes (32) than the current scenarios (29). Need scenario-topology alignment or dynamic node mapping.
- No force-directed graph visualization in engine UI yet — still a grid layout. Force graph would show cascade propagation paths visually and make the demo more compelling.
- Mobile/responsive layout not addressed (low priority for internal tool).

---

## PRIORITY ORDER

What moves the needle fastest, accounting for dependencies:

| Priority | Task | Effort | Dependency | Impact |
|----------|------|--------|------------|--------|
| 1 | MC dropout (Step 5) | 0.5 day | None | Makes confidence honest. High demo value. |
| 2 | Retrain GNN + fusion on 1044 | 1 day | None | Completes dim upgrade end-to-end. |
| 3 | Longer cascade sequences | 1 week | None | Directly improves temporal model quality for real-world use. |
| 4 | Diverse topologies retrain | 1-2 days | None | Improves generalization across infrastructure patterns. |
| 5 | Feedback API schema + endpoint | 1 day | None | Builds the loop before you need it. |
| 6 | Pre-compute new 1044 scenarios | 1 hour | #2 | Updates Docker demo. |
| 7 | Pitch artifact | 2-3 hours | None | Needed before design partner conversations. |
| 8 | SNMP adapter | 2-3 days | MSP access | Build when you see the actual stack. |
| 9 | Fine-tuning data loader | 2-3 days | Design partner data | Build when real data arrives. |
| 10 | MC dropout calibration | 2-3 days | Design partner data | Requires held-out real incidents. |
| 11 | Dashboard force graph | 2-3 days | None | Visual polish for demo. |

---

## TIMELINE

**Before MSP start (can do now):**
- MC dropout (0.5 day)
- Retrain GNN + fusion on 1044 (1 day)
- Pre-compute new scenarios (1 hour)
- Diverse topologies retrain (1-2 days)
- Longer cascade sequences (1 week)
- Feedback API schema + endpoint (1 day)
- Pitch artifact (2-3 hours)

**Total pre-MSP effort: ~2 weeks**

**After MSP start, before design partner:**
- SNMP adapter (if needed based on their stack)
- Pitch artifact refinement with MSP-specific context

**After design partner agreement (5-7 weeks to production pilot):**
- Week 1-2: Integration test against real Prometheus, build any missing adapters
- Week 2-3: Fine-tuning data loader + first fine-tune pass
- Week 3: MC dropout calibration on real data
- Week 4: Feedback loop UX + deployment
- Week 5-7: Iterate on operator feedback, continuous improvement

---

*Total: 5-7 weeks from design partner agreement to production pilot. The architecture is built. The training pipeline is built. The telemetry adapter is built. What's missing is real data and the last-mile calibration.*
