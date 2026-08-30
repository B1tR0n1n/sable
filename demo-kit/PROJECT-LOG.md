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
