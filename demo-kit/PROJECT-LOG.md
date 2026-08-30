# SABLE — Audit, Rebuild & Demo Kit

**A complete record of the work: what SABLE is, what was wrong with it, how it was rebuilt honest, what it can actually do, and everything shipped.**

Field Systems Division · b1tr0n1n · 2026-08-30

---

## TL;DR

SABLE is an **infrastructure diagnosis engine**: give it a dependency map + whatever your monitoring sees (always partial), and it infers the state of the whole estate — especially the nodes you *aren't* monitoring — by propagating observed failures through the graph.

It was found reporting **inflated 82–96% accuracy that was a lie** (label leakage), with a **structural reasoning pillar fed all-zeros** (dead), and node/edge types silently discarded. All of that was audited, fixed, and re-measured honestly. It now does something real, if modest: **catches ~40% of failures in unmonitored nodes that monitoring alone catches 0% of, ~85% accuracy tracking a live incident.** Every number in this kit is measured on de-leaked data.

---

## 1. What SABLE does

**Input:** a dependency graph (what runs on what) + partial observations (which nodes your monitoring reports healthy/degraded/failed; most are unwatched).
**Output:** a predicted state for every node, with the failures in **unmonitored** nodes flagged, a ranked "check this first" list, and a likely root cause.

**Architecture (three pillars, fused):**
- **GNN** — encodes topology/structure: how a failure propagates from a dependency to its dependents.
- **POMDP belief** — folds in the partial monitoring (fog-of-war).
- **Mamba / temporal chain** — tracks how the incident evolves over time.
- **LoRA adapter** — per-environment fine-tuning on top.

**Two modes:** snapshot triage (one point in time) and live-stream monitoring (tracks an incident tick by tick).

---

## 2. The audit — what was wrong (all verified against code + data)

| Finding | Detail | Evidence |
|---|---|---|
| **Label leakage** | The true node state was one-hot-encoded directly in the Mamba input. A trivial `argmax` recovered **100%** of the labels on synthetic scenarios. The demoed 82–96% measured the model copying an input feature. | Empirically: `argmax(mamba[:,1:6]) == ground_truth` = 100%. |
| **Dead GNN** | The GNN was fed an **all-zero tensor** its entire existence — an edge-lookup direction bug (`get_dependency(comp.id, dep_id)` reversed) meant the edge list was always empty and `gnn.encode()` never ran. The "three-pillar" system was really two pillars + a corpse. | GNN embeddings: `0/256` non-zero columns. Expert F1 = 0.02. |
| **Node & edge types discarded** | Both `COMP_TYPE_MAP` and `DEP_TYPE_MAP` were keyed on `"ComponentType.X"` but `str(enum)` yields bare `"X"` — every node and edge mapped to `"unknown"`. The GNN never saw types. | Lookup miss → fallback to `unknown`. |
| **Oscillating class unlearnable** | 0.00% of the fusion training data was the oscillating class. | Class distribution: `oscillating = 0`. |
| **Temporal + LoRA trained on leaked data** | The temporal chain and LoRA adapter (claimed "F1 0.548") were fine-tuned on the leaked pipeline — invalid for an honest model. | Boot log F1 was a leaked-data artifact. |
| **Orchestrator CLI broken** | Imported a module (`cortex_gnn_model`) that had been deleted in a reorg. | ModuleNotFoundError on a clean run. |

---

## 3. The rebuild — fixes and honest results

Every fix was measured on **de-leaked, partial-observability data** (observation-based inputs, ~91% observation ceiling, genuine hidden nodes).

| Milestone | Routed macro-F1 | Hidden-failure catch | GNN F1 | Notes |
|---|---|---|---|---|
| Leaked demo (fake) | ~0.55 claimed | — | 0.02 | 82–96% accuracy was leakage |
| De-leaked baseline | 0.35 | 27% | 0.02 | honest floor; GNN still dead |
| + GNN un-zeroed (edge bug fixed) | 0.49 | 39.5% | 0.34 | structural reasoning alive |
| + typed nodes/edges + oscillating | 0.64 | ~40% | 0.33 | oscillating F1 0.00 → 0.998 |
| + GNN reachability objective | 0.63 | 40.8% | 0.36 | **flat — honest negative result** |
| + temporal + LoRA (de-leaked) | — | — | — | **full stack ~85% on streams** (vs ~60% base); LoRA F1 0.548 fake → **0.31 honest** |

**What earned its keep:** de-leaking (the honesty), the GNN edge-bug fix (biggest single-metric jump), typed features, the oscillating fix, and the temporal+LoRA retrain (biggest accuracy lift, ~60% → ~85% on live streams).

**What didn't:** the GNN reachability objective came back flat — the GNN was already propagation-trained via its state objective, and it isn't the current ceiling. Recorded as an honest dead end, not spun.

**Fixes that were the right call to *drop*:** the old LoRA (leaked); "fine-tune the GNN to raise its standalone F1" (would erode the pillar division of labor — the GNN's job is topology, not state classification).

---

## 4. What's honest now — capability and limits

**Capability (measured):**
- **~40%** of failures caught in unmonitored nodes vs **0%** for trusting monitoring alone — the core value.
- **~85%** accuracy tracking a live incident over time (full stack); ~60% base fusion alone.
- Milliseconds per diagnosis. Detects flapping (oscillating). Handles all five states.
- Routed fusion beats every single pillar — the three-pillar design genuinely adds value now.

**Limits (say these before a buyer does):**
- It's a **triage aid, not an oracle** — a ranked "look here," often at modest confidence (e.g. 29%), not an automated verdict.
- It **lives on topology quality**. Auto-discovery (below) mitigates this, but garbage map in → garbage inference out.
- Confidences are honest and modest. It augments monitoring; it doesn't replace it.

---

## 5. Deliverables — what shipped and how to run it

All in the repo, all committed. The honest model checkpoints are deployed; leaked originals kept as `*.leaked.bak.pt`.

### Demo kit (`demo-kit/`)
| File | What it is | Run |
|---|---|---|
| `index.html` | **The engine dashboard** (node grid, routing, confidence, live accuracy, per-tick playback) as a self-contained file, playing **honest de-leaked runs**. | double-click (offline) |
| `topology-view.html` | Single-screen topology diagram — observed alerts vs SABLE-inferred hidden failures (Monday Morning cascade). | double-click (offline) |
| `serve.py` + `live.html` | **Live interactive mode.** Pick/scan a topology, click nodes to set what monitoring sees, Diagnose against the real model. Includes **⟳ Scan Network**. | `python3 serve.py` → http://localhost:8760 |
| `PROJECT-LOG.md` | This document. | — |
| `RUN-OF-SHOW.md` / `START-HERE.txt` | Pitch flow + quick start. | — |

### Topology ingest (`adapters/topology_ingest.py`)
Feed a dependency graph JSON + observations → diagnosis. Infers hidden-node state by propagating observed failures.
`python3 adapters/topology_ingest.py adapters/monday_morning.json`

### Network auto-discovery (`adapters/network_discovery.py`)
Scans a live network and **builds the SABLE topology automatically** — closes the "needs a clean map" gap.
- **nmap** sweep + service/version fingerprinting → role inference (gateway/switch/DNS/DC/hypervisor/storage/app).
- **SNMP/LLDP/CDP** neighbour walk → real L2 switch edges (needs a community string; consumer gear falls back to an L3 star).
`python3 adapters/network_discovery.py --snmp public --diagnose`
Verified live: discovered the local lab (Ubiquiti gateway, workstation, phone, TV), and SABLE inferred the gateway-failure blast radius across the hidden devices.

### Engine (`docker/`)
The full FastAPI engine + dashboard, now loading the **honest** checkpoints. Boot with `sable.sh` (workstation).

---

## 6. Positioning — is there anything like this, and where it fits

**Yes, it's a mature field** — AIOps / topology-based root-cause analysis. Real players: **Dynatrace (Davis AI + Smartscape)** (closest — dependency graph + auto-RCA), **BigPanda, Moogsoft** (alert correlation), Datadog Watchdog, Splunk ITSI, and MSP-oriented ScienceLogic/LogicMonitor/OpsRamp. Inferring failures through a dependency graph is *sold*, not novel.

**SABLE's honest niche:** the incumbents are **agent-based** — they instrument everything, so they *know* node state. SABLE's premise is the opposite: work from a **known topology + partial observations** and infer nodes you **can't or don't instrument**. That matters in exactly one place — **environments you can't fully instrument**: an MSP's inherited heterogeneous client estates, OT/ICS, edge, third-party dependencies. Where Dynatrace says "install the agent," SABLE says "you can't — infer it structurally."

**The MSP wedge:** MSPs uniquely already maintain dependency data (IT Glue / Hudu), and can't afford Dynatrace per-client across 40 SMBs. SABLE slots **between the RMM and the PSA ticket**: reads the doc tool + the alerts, and tells tier-1 "the root cause is the switch underneath these app alerts, and here are 3 unmonitored things also hit." **Auto-discovery is the onboarding wedge** — point it at a client subnet, no CMDB required.

**The honest strategic read:** as a standalone product it's a hard road (crowded, well-funded incumbents, topology dependency). As a **feature in an MSP stack** it's plausible. As a **demonstration of engineering judgment** — architecting a three-pillar model, catching that it was lying via leakage, and rebuilding it honest — it's rare and genuinely valuable. That last use is the highest-value one: **the market needs engineers who can do this, more than it needs another AIOps startup.**

---

## 7. Safety & git

- **Safety tag:** `pre-cleanup-20260830` (full pre-work snapshot; `git reset --hard` restores).
- **Backups:** `fusion/infra_fusion_data.leaked.bak.pt`, `pillar1/checkpoints/best_model.linkpred.bak.pt`, `docker/checkpoints/*.leaked.bak.pt`, `fusion/checkpoints/*.leaked.bak.pt`.
- **Repo:** cleaned and organized; a stray 721 MB sub-project and vendored junk were gitignored, working tree committed in logical commits.
- **`feed/`** is a separate sub-project (React Native app) — gitignored here, should be its own repo.

---

## 8. Honest bottom line

SABLE is a **real, working, honestly-measured** infrastructure diagnosis engine that fills monitoring blind spots by reasoning over dependencies. It is **not** a category-defining product, and its numbers are modest — but they're **true**, which is the whole point. The rebuild turned a system that lied about 96% into one that honestly delivers ~40% of the invisible, with auto-discovery, a live dashboard, and a defensible niche.

The strongest asset here isn't the tool. It's the demonstrated ability to take a leaking, broken ML pipeline and make it honest — and that's the thing worth putting in front of anyone hiring for AI infrastructure work.

---

## 9. Second-pass audit + full retrain (2026-08-30, later)

A three-reviewer code audit (ML core, serving layer, security) was run over the whole tree to check the "all fixed, all de-leaked" claim above. It held for the fusion/eval path that produced the headline numbers, but found real gaps that were then fixed and re-measured.

### What the audit found and fixed
- **Two live leakage regressions.** `orchestrator/sable_core.py` and `pillar3/generate_temporal_data.py`'s own generator still called `encode_system_state()` on true state (no belief) — re-triggerable via each file's documented CLI. Fixed: `encode_system_state` is now **default-safe** (raises without a `belief`; explicit `allow_true_state=True` only for labelled leaked-baseline probes), and both call sites thread a real fog-of-war belief.
- **A second, distinct GNN leak.** Separate from the "dead GNN" bug in §2: the pillar-1 pretraining data (`build_infra_dataset.py`) fed the GNN a health scalar that was `0.0` iff a node was a root failure — so "failed" was recoverable from one input, and the standalone GNN F1 was partly a shortcut, not topology reasoning. Fixed with observation-gating (noise + masking). Verified: failed nodes with `health<0.05` went from **100% → 47%**, distinct health values `2 → 1001`.
- **Silent fabrication path.** `sable_engine.py` returned `True` and served confident predictions even with missing checkpoints (random weights). Now raises, and `/api/status` exposes `model_healthy`.
- **Security pass.** Both servers now bind `127.0.0.1` by default (opt-in `SABLE_HOST`/`SABLE_TOKEN` for LAN) with token auth on mutating endpoints; path traversal closed in `serve.py`; `/api/scan` single-flight lock; 28 dashboard `innerHTML` XSS sinks escaped; scanned hostnames sanitized at the source. Verified live: server binds localhost only.
- Also: `temporal_chain` stability-feature logic bug, torch seeding for reproducibility, `hash()`-salt scenario-seed bug, LoRA single-scenario split guard.

### The retrain (forced by the GNN fog-gating fix)
Because the fusion checkpoints train on the frozen GNN's embeddings, fixing the GNN forced a **full-stack retrain**: regenerate GNN dataset → retrain GNN → regenerate fusion data + temporal sequences → retrain staged-fusion, temporal-chain, LoRA → regenerate demo scenarios. All checkpoints deployed and verified; pre-retrain checkpoints backed up (`checkpoint_backup_preretrain_20260830/`) for rollback.

**Honest results on the de-leaked GNN stack:**

| Component | Macro-F1 | Notes |
|---|---|---|
| GNN (pillar 1) | **0.726** | failed 0.755, unreachable 0.850 — real structural reasoning now, **not** the old health shortcut |
| Staged fusion (Routed) | **0.648** | beats fusion-only +0.128, best expert +0.056 — router still earns its keep |
| Temporal chain | **0.630** | failed 0.810 |
| LoRA | **0.326** | consistent with the honest 0.31 above |
| Live-incident tracking | **~76–97%** | per-scenario, ~7 ms/tick on the 5090 |

**The load-bearing result:** closing the GNN leak did **not** tank the fusion numbers — the stack landed at 0.63–0.65, on par with the previously-reported 0.63. So the fusion router was already doing real work; only the standalone pillar-1 GNN F1 had been flattered by the shortcut, and that number is now genuinely **0.73** without the leak. Every figure here is measured on the fully-consistent, de-leaked, deployed stack.

### Synthetic vs. real telemetry — the honest gap
The SMD real-server-telemetry scenarios (`smd_*`, 28 machines, OmniAnomaly ServerMachineDataset) were also regenerated on the new GNN and run live through the deployed engine. On real telemetry the stack tracks incidents at **~57–68% per-tick node-state accuracy** — meaningfully **lower** than the 76–97% on synthetic simulator scenarios, which is expected: real metrics are noisy and the labels come from the anomaly-detection scorer, not a clean simulator. This is the number to quote for real-world behaviour, and it does **not** reproduce the "90.4% on real server telemetry" figure that appears in older notes — that claim should be treated as stale/differently-measured until re-derived on this stack. Inference holds at ~7 ms/tick either way.
