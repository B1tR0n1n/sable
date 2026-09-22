# Phase summaries

What each phase of `PLAN.md` changed, what it found, and what is open.
Acceptance criteria were met by tests in `console/tests` unless noted.

## Phase 0 — Reconnaissance
`docs/INTEGRATION-NOTES.md`, every claim cited `path:line`. Key findings:
SABLE's recommendations carry no confidence (only per-node state
confidence on the tick); the served root cause is the first node to leave
`healthy` (temporal, not causal); precomputed scenarios use random
topologies (labels are positional cosmetics); no `detection_mode`; no
tests, no dependency file, Dockerfile missing files; OVERLORD's rollback is
pre-commit discard, filesystem-scoped. Decisions taken: confidence =
root-cause node's state-classification confidence (`engine_native`, or
`mc_dropout` when sampling is on); `unmonitored_gap` when the root cause is
unreachable **or depends on** an unreachable node; loop rollback =
compensation through OVERLORD; replay for mapping tests only, topology
validated in the lab.

## OVERLORD (0.30.0) — executor surface
`revert <sid>` (undo a committed session as a new reviewable session,
including a removed directory), `cause` on provenance, `net_allow`/`limits`
through the SDK, a brokered `audit` op (namespaced external events on the
keyed chain), a tool-less `complete` op with hashed audit, a commit-time
content policy gate (secrets / binaries / dependency files / size). Two
bugs fixed on the way: deleting a directory was un-committable; the
installer omitted a module.

## Phase 1 — Contracts
Field lists per the plan; additive fields marked. `Plan.reversibility` =
worst step; compensable steps must name a compensation; `Receipt` hash
chain with `audit_ref` outside the hash (the audit entry commits to the
receipt hash — one direction must be outside). Schemas + fixtures pinned.

## Phase 2 — SABLE finding emitter
Client, mapper, store, emitter, router. Open: SABLE exposes no engine
version (`sable@<sha>` from `SABLE_GIT_SHA`); evidence metric names are
node-prefixed. Live replay is exercised in the lab.

## Phase 2B — Claude language layer
`docker/claude_bridge.py`, three additive hunks in `docker/server.py`,
Dockerfile + `docker/requirements.txt`. `SABLE_LLM=claude` selects it;
unset → Nemotron, unchanged. Tools are read-only with bounds checks.

## Phase 3 — Catalog and planner
Five actions (no destructive ones), load-time cross-checks, docker-compose
bindings at declared grants. Templates propose what SABLE can verify: a
degraded app/proxy → golden-config restore (reversible) with restart as
fallback; a dead primary → restart. **Failover is not a template** — it
moves the proxy but leaves the primary and the app down, so `node_healthy`
could never pass; the LLM planner may still propose it. Invalid LLM output
is rejected with reasons, never repaired. Open: `set_config_value` needs a
`service` param (the follow-up restart); stricter-than-spec precondition /
predicate declaration checks.

## Phase 4 — Executor
One session per step at the binding's scope; precondition → action →
follow-up → commit; failure rolls back and compensates completed steps in
reverse (`overlord revert` for reversible, catalog compensation for
compensable); irreversible refuses without approval; receipts chained and
mirrored. Fault injection at every position tested.

## Phase 5 — Verification
Pure `evaluate()`; pass closes, fail compensates + reopens, inconclusive
(no fresh tick, stale, unreachable, missing) escalates and never passes.

## Phase 6 — Autonomy gate
Frozen validated policy; shipped default all `human` (`default-deny:`
rule ids); target matrix opt-in; gap cap; irreversible never auto.

## Phase 7 — Console
Server (`Loop` + FastAPI per `API.md`) and UI (React + Vite, 40 tests).
Open: SABLE's chat API does not return the tool calls Claude made, so the
UI's tool chips are empty until SABLE exposes them (`claude_bridge.chat`
could return them — additive).

## Phase 8 — Lab
Built and statically validated (25 tests); **not run** in the build
environment (no Docker daemon). First live run is the acceptance test:
`make -C console lab-e2e`. Runtime unknowns listed in `lab/README.md`
(nginx template escapes, dnsmasq HUP path, blackbox DNS validation, image
healthchecks).

## Not in the MVP
Multi-site policy, Graph/M365 adapters, signed (not just chained)
receipts, prompt caching, an eval set of scenarios, per-site egress policy.
