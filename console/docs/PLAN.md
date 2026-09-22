# PLAN.md — OVERLORD × SABLE Closed-Loop Console

> Build plan for Claude Code. Work one phase at a time. At the end of each phase, stop,
> summarize what changed, list open questions, and wait for approval before continuing.

---

## 0. What we are building

A closed remediation loop for infrastructure:

```
DETECT (SABLE) → PROPOSE (planner) → GATE (policy) → EXECUTE (OVERLORD) → VERIFY (SABLE) → RECEIPT
                                                                               │
                                                  fail → OVERLORD rollback ◄───┘
```

- **SABLE** (`github.com/B1tR0n1n/sable`, Python) is the perception layer. It ingests telemetry
  through adapters (`adapters/base.py` → `SystemSnapshot`), diagnoses root cause over a dependency
  graph, and already serves results via FastAPI (`docker/server.py`, `docker/sable_engine.py`).
- **OVERLORD** (`github.com/B1tR0n1n/overlord`) is the actuation layer: transactional, scoped,
  recorded execution. It gains a built-in agent loop calling Anthropic/OpenAI APIs, with tools
  executing inside the transactional session.
- **Claude** is the language and reasoning layer. It writes summaries and reports, runs
  operator chat with read-only tool access to SABLE, and proposes plans from the action
  catalog. It is an analyst, never an actor. SABLE's local Nemotron bridge remains as the
  on-box alternative for privacy-sensitive sites.
- **The product** is a third component that joins them through three data contracts
  (Finding, Plan, Receipt) and a single operator console.

**Three roles, never blurred:** SABLE's models are the source of truth, Claude is the analyst,
and OVERLORD is the only component that changes anything.

**MVP domain: infrastructure remediation**, matching what SABLE's engine models today
(switches, servers, DNS, domain controllers, storage, and so on, per `COMPONENT_TYPES`).
Microsoft 365 identity hygiene is a later adapter, not part of the MVP.

---

## 1. Ground rules (apply to every phase)

1. **Do not modify SABLE's model code or weights.** That covers `pillar1/`, `pillar2/`,
   `pillar3/`, `fusion/`, and checkpoints. Integrate by wrapping engine output, not by
   changing the models.
2. **Keep the repos separate.** The product lives in a new repo (working name `console`)
   that depends on both. Changes inside `sable` or `overlord` must be small, additive, and
   listed in the phase summary.
3. **Contracts first.** No integration code before the schemas in Phase 1 are approved.
4. **The LLM never invents actions.** Plans may only reference actions in the action catalog
   (Phase 3). The LLM may choose and parameterize catalog actions, write summaries, and rank
   options. It may not produce arbitrary commands.
5. **Default-deny autonomy.** Until policy says otherwise, every plan requires human approval.
6. **Lab only.** All execution targets the local lab (Phase 8). No real credentials, no
   customer systems, and no employer infrastructure, ever. Secrets come from environment
   variables and never enter the repo.
7. **Everything recorded.** Every executed step produces a record. If something can't be
   recorded, it doesn't execute.
8. **Claude reads, it never acts.** Every tool exposed to Claude inside SABLE is read-only. Any
   state change goes through a Plan, the gate, and OVERLORD. Claude never overrules SABLE's
   diagnosis without citing conflicting data, and generated text is always flagged as generated.
9. **The LLM provider is a configuration choice.** `SABLE_LLM=claude` uses the Claude API;
   anything else uses the local Nemotron bridge. Don't delete or break the local path, because
   some customers won't allow telemetry to leave the box. The API key comes only from
   `ANTHROPIC_API_KEY`.
10. **Tests with every phase.** Use pytest for Python and Vitest for the console. Each phase
   lists acceptance criteria, and the phase is done only when they pass.

---

## Phase 0 — Reconnaissance (no code changes)

**Goal:** understand both codebases before designing against them.

Tasks:
- Read the OVERLORD repo. Document its session and transaction model: how a session opens,
  how scope and permissions are declared, how actions are recorded, what rollback exists
  today, and the state of the built-in agent loop.
- Read SABLE's `docker/sable_engine.py`, `docker/server.py`, `adapters/base.py`,
  `adapters/health_scorer.py`, `orchestrator/sable_core.py`, and `orchestrator/cortex_repair.py`.
  Document exactly what a diagnosis output contains: root-cause node, affected nodes,
  confidence and its source (including MC-dropout uncertainty, if exposed), and timing.
- Read `docker/nemotron_bridge.py` and the `/api/nemotron/*` routes in `docker/server.py`.
  Confirm that the report methods build their prompts and then call `_complete()`. Also find the
  internal functions behind `/api/recommendations`, `/api/node/{idx}`, and `/api/topology`, since
  Phase 2B's tools will call them directly.
- Note runtime constraints. SABLE's engine initializes with `device="cuda"`, so confirm
  whether a CPU fallback exists or is needed for the lab.

Deliverable: `console/docs/INTEGRATION-NOTES.md` covering both engines' real interfaces,
gaps against this plan, and proposed adjustments.

**Acceptance:** the notes cite file paths and function signatures for every claim. Stop for
review.

---

## Phase 1 — Contracts

**Goal:** define the three objects that everything else speaks.

Create the `console/contracts/` package with Pydantic v2 models plus exported JSON Schema.

### Finding (emitted by SABLE)
```
id, created_at, site_id
detection_mode: "live_feed" | "unmonitored_gap"
root_cause: { node_id, component_type, state }
affected_nodes: [ { node_id, component_type, state } ]
evidence: [ { metric, value, unit, threshold?, timestamp } ]
confidence: { score: 0..1, method: "mc_dropout" | "engine_native" | "heuristic", samples? }
severity: "low" | "medium" | "high" | "critical"
summary: str            # may be LLM-written; flag with summary_generated: bool
engine_version: str
```

### Plan (produced by the planner, annotated by the gate)
```
id, finding_id, created_at
steps: [ {
  step_id, action_id,             # action_id MUST exist in the catalog
  target_node, params,
  reversibility: "reversible" | "compensable" | "irreversible",
  compensation: { action_id, params } | null,
  precondition: str,              # a check name from the catalog
  timeout_s
} ]
blast_radius: { nodes: [node_id], count }
verification: { predicate: str, window_s: int }
gate: { decision: "auto" | "delay" | "human" | "human_plus" | "report_only",
        rule_id, reason, delay_s? } | null
```
The plan's overall reversibility is its **worst** step.

### Receipt (produced by OVERLORD after execution and verification)
```
id, plan_id, session_id
approvals: [ { actor, decision, timestamp } ]
steps: [ { step_id, started_at, ended_at, status, output_digest } ]
snapshots: [ { node_id, snapshot_ref } ]
verification: { status: "pass" | "fail" | "inconclusive", observed, checked_at }
rollback: { performed: bool, steps: [...] }
prev_receipt_hash, receipt_hash   # hash chain for tamper evidence
```

**Acceptance:** round-trip serialization tests pass; JSON Schemas are exported to
`contracts/schema/`; one fixture example exists per object. Stop for review.

---

## Phase 2 — SABLE finding emitter

**Goal:** SABLE diagnoses become Findings.

- Add `console/sable_bridge/`, which calls SABLE's engine (or its existing API) and maps output
  to `Finding`.
- Set `detection_mode` from whether the diagnosis came from live telemetry or from an
  unmonitored gap. This distinction drives the gate later.
- Expose `GET /findings`, `GET /findings/{id}`, and a WebSocket stream of new findings.
- Deduplicate: a persisting condition updates its open Finding rather than creating a new one.

**Acceptance:** replaying a precomputed SABLE scenario (`docker/precompute_scenarios.py` data)
produces stable, deduplicated Findings. Tests cover the mapping for each component type in the
scenario.

---

## Phase 2B — Claude language layer in SABLE

**Goal:** Claude becomes SABLE's analyst: reports, Finding summaries, and a tool-using operator
chat. The local Nemotron path is preserved.

**Reference implementation provided:** `claude_bridge.py` ships alongside this plan. It defines
`ClaudeBridge(NemotronBridge)`, a drop-in replacement with the same public interface.
- It inherits `explain_tick`, `explain_recommendations`, and `explain_scenario_complete`
  unchanged, and overrides `_complete()` to call the Claude Messages API. The report prompts need
  no changes.
- It overrides `chat()` with a tool-use loop (capped at 6 rounds) over three read-only tools:
  `get_recommendations`, `get_node(idx)`, and `get_topology`.
- It adds `register_tools(...)`, which the server calls with read-only accessors once the engine
  is initialized.
- The default model is `claude-sonnet-5`, overridable with `SABLE_CLAUDE_MODEL`.
- API errors and tool failures return marked strings and never crash the server.

Treat it as a reviewed starting point, not verified code. It was written against the public repo
without running it.

Tasks:
1. Copy `claude_bridge.py` into `sable/docker/` and add `anthropic` to SABLE's dependencies.
2. In `docker/server.py`, choose the bridge based on the environment. Keep the variable name
   `nemotron` so every existing `/api/nemotron/*` route and the dashboard keep working:
   ```python
   if os.getenv("SABLE_LLM") == "claude":
       from claude_bridge import ClaudeBridge
       nemotron = ClaudeBridge()
   else:
       nemotron = NemotronBridge()
   ```
3. After engine initialization, call `nemotron.register_tools(...)` with the functions found in
   Phase 0. Only call it when the bridge is a `ClaudeBridge`. Each tool must return plain
   JSON-serializable dicts and must not mutate engine state.
4. Update `/api/nemotron/status` output so the dashboard shows which provider is active. It
   should keep working with both providers.
5. In the console (not in SABLE), use the same provider switch to generate `Finding.summary`, and
   set `summary_generated: true`.
6. Expose Claude to the Phase 3 LLM-assisted planner through OVERLORD's agent loop. The planner's
   contract is unchanged: it may only select and parameterize catalog actions, and its output is
   validated against the schema.

Acceptance:
- With `SABLE_LLM` unset, behavior is identical to today's Nemotron path.
- With `SABLE_LLM=claude` and a valid key, all three report endpoints return text for a
  precomputed scenario.
- Chat makes at least one tool call when asked about something not in the seeded context. For
  example, "what is upstream of [NODE]?" should trigger `get_topology` or `get_node`.
- A test confirms that no registered tool changes engine state: snapshot state, run every tool,
  and compare.
- A test confirms that a missing key produces the "unavailable" message rather than an exception.
- A test confirms that an unknown tool name returns `is_error` to Claude.

Stop for review.

---

## Phase 3 — Action catalog and planner

**Goal:** turn Findings into Plans using only known actions.

- Create `console/catalog/actions.yaml`. Each entry has: `action_id`, the applicable
  `component_types`, the params schema, `reversibility`, the `compensation` action,
  `precondition` checks, the `verification` predicate, and an executor binding for OVERLORD.
- Start with the smallest useful set for the lab:
  - `restart_service` (compensable, since the prior running state is restorable)
  - `set_config_value` (reversible, via snapshot)
  - `failover_to_replica` (compensable, via fail-back)
  - `clear_dns_cache` (reversible in effect, low risk)
  - Destructive actions are excluded from the MVP catalog entirely.
- The planner has two layers:
  1. A **deterministic template planner**: a finding pattern maps to a step template. This
     ships first.
  2. An **LLM-assisted planner** (via OVERLORD's agent loop): it may select and order catalog
     actions and fill params, and its output is validated against the catalog and schema.
     Invalid output is rejected, never repaired silently.
- Compute blast radius from SABLE's dependency graph: the target plus downstream dependents.

**Acceptance:** every lab fault scenario yields a valid Plan from the template planner.
LLM-planner output that references an unknown action fails validation, with a test proving it.

---

## Phase 4 — OVERLORD executor

**Goal:** execute Plans transactionally and emit Receipts.

- Add an executor adapter that opens an OVERLORD session with the **minimum scope** the plan's
  actions require.
- For each step: check the precondition, snapshot, execute, and record, with a timeout per step.
- On step failure, run compensations in reverse order for completed steps, then mark the
  receipt `rollback.performed = true`.
- Irreversible steps (none in the MVP catalog) must refuse to run without a recorded approval.

**Acceptance:** fault-injection tests cover a failure at each step position and prove correct
reverse-order compensation. Every run, successful or not, produces a Receipt.

---

## Phase 5 — Verification loop

**Goal:** prove fixes worked.

- After execution, wait `verification.window_s`, then have SABLE re-evaluate the affected nodes.
- A pass closes the Finding with the Receipt attached.
- A fail triggers OVERLORD rollback and reopens the Finding with the failed attempt attached.
- An inconclusive result (for example, telemetry is stale) escalates to a human and never
  counts as a pass.
- Finalize the receipt hash chain here.

**Acceptance:** an end-to-end test in the lab (Phase 8) covers injecting a fault, fixing it,
passing verification, and getting a closed Finding. A second test uses a deliberately wrong fix
and confirms that verification fails, rollback runs, and the Finding reopens.

---

## Phase 6 — Autonomy gate

**Goal:** decide how much autonomy each plan gets.

- Create `console/policy/policy.yaml` with a matrix of confidence band × plan reversibility →
  decision. Confidence bands (high, medium, low) are configurable thresholds.

| confidence \ reversibility | reversible | compensable | irreversible |
|---|---|---|---|
| high   | auto   | delay  | human      |
| medium | human  | human  | human_plus |
| low    | report_only | report_only | report_only |

- **Shipped default: every cell is `human`.** The matrix above is the target configuration the
  operator opts into.
- An `unmonitored_gap` finding is capped at one band below its score.
- A `delay` decision starts a visible countdown. The operator can execute now, hold, or reject.
- Every gate decision is written into the Plan with the `rule_id` that produced it.

**Acceptance:** table-driven tests cover every cell, plus the unmonitored-gap cap and
default-deny behavior.

---

## Phase 7 — Console UI

**Goal:** a single operator screen, matching the approved design artboard
(OVERLORD × SABLE Console).

- Use React + Vite in `console/ui/`, served by a FastAPI `console/server/` that orchestrates the
  loop and proxies to both engines.
- Screens for the MVP:
  - **Console:** a loop-state strip, the findings list, the selected finding's plan with its
    gate decision, the autonomy matrix, the live session log, and recent receipts.
  - **Receipt detail:** full step log, snapshots, verification, and hash, with export to PDF or
    JSON.
  - **Analyst panel:** Claude chat, scoped to the selected Finding. It shows which tools Claude
    called, visually separates SABLE data from Claude's inference, and includes a
    provider indicator (Claude API or local) in the header.
- Replace the design artboard's sample content (identity-style findings) with infrastructure
  faults from the lab.
- Follow the b1tr0n1n style: backgrounds `#0a0908`/`#0f0e0b`, text `#c8bda0`/`#ede5d0`, gold
  `#c9a227` as the only accent, red `#a63d2f` and green `#4a7a45` for semantics only. Headings
  and prose use Cormorant Garamond; labels, metadata, and logs use JetBrains Mono. Use a subtle
  80px grid background, and no light mode.
- Use live updates over WebSocket. Approve, hold, and reject are real buttons, and each writes an
  approval record.

**Acceptance:** the full lab loop (inject → finding → plan → approve → execute → verify →
receipt) can be driven entirely from the UI.

---

## Phase 8 — Lab environment (build alongside Phases 2–5)

**Goal:** a safe, reproducible world to break and heal.

- `console/lab/docker-compose.yml` runs a small topology with DNS, an app service, a replica,
  a reverse proxy, Prometheus, and exporters, mirroring `adapters/topologies/msp_demo.yaml`
  where practical.
- `console/lab/faults/` holds scripts that inject faults: stop a service, corrupt a config
  value, kill the primary, or poison the DNS cache.
- SABLE reads the lab through its existing Prometheus adapter (`adapters/prometheus.py`).
- A single command runs the end-to-end suite: `make lab-e2e`.

**Acceptance:** `make lab-e2e` passes from a clean checkout.

---

## MVP definition of done

- Lab faults are detected, planned, gated, executed, verified, and receipted end-to-end.
- A failed fix is automatically rolled back and reopened.
- No action outside the catalog can execute.
- The default policy requires human approval for everything.
- The console drives the full loop and exports receipts.
- Claude writes reports and summaries and answers operator questions using read-only tools, and
  the local provider still works with a single environment change.

## Explicitly out of scope for the MVP

Multi-tenant SaaS hosting, M365/Graph adapters, billing, RBAC beyond a single operator,
destructive catalog actions, and changes to SABLE's models.

## Revisit as it grows

- Move from a single site to multi-site, with per-site policy.
- Add a Microsoft Graph adapter and identity action catalog as a second domain.
- Add signed receipts (not just hash-chained) for third-party attestation.
- Package self-hosted deployment for MSPs.
- Add prompt caching for the large seeded SABLE context, and track token cost per Finding.
- Add an eval set of scenarios with known root causes, to score whether Claude's explanations
  agree with SABLE and to catch regressions when models change.
- Add per-site data-egress policy that decides which fields may be sent to the API and which
  stay local.
