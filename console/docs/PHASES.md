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

## Post-first-run fixes
What the first live run against the lab turned up, and what changed:

- **Findings lifecycle.** A finding born from a transient blip (a heal-all
  recreate) stayed `open` for ever and later faults deduped into it. Now
  the loop resolves an open finding whose root cause reads healthy for
  `CONSOLE_RESOLVE_TICKS` (3) consecutive ticks with nothing executing —
  status `resolved` (no action, no receipt), pending plan withdrawn, plan
  mapping cleared. A `reopened` finding was never re-planned (`plan_of` was
  never cleared); now a failed verification proposes a fresh plan, once per
  failed receipt, capped at `CONSOLE_MAX_ATTEMPTS` (2) — then the finding is
  `escalated` (event + warn line + audit mirror) and the loop stops proposing.
  `FindingStatus` grew `resolved` and `escalated` (schema regenerated); the
  UI treats `resolved` as closed and tags `escalated`. Tests:
  `tests/test_lifecycle.py`.
- **Auth on the console API.** `CONSOLE_TOKEN` gates every mutating route
  behind `Authorization: Bearer <token>` (constant-time compare, 401 with the
  documented error shape); reads and `/ws` stay open; unset → open, one
  startup warning. The UI reads the token from `localStorage["console_token"]`
  or a `?token=`/`#token=` on the link (stored, stripped). Tests:
  `tests/test_auth.py`, `ui/src/__tests__/api.test.ts`.
- **Verification window.** dns came back healthy while the lab app still read
  `degraded`/`oscillating` at 30s — the app's dependency probe settles slower.
  `restart_service` now declares `verification_window_s: 60` in the catalog
  (`ActionSpec.verification_window_s`, planner default still 30); the verifier
  treats `oscillating` on an affected (non-target) node as inconclusive and
  re-checks once after a 15s settle against a NEWER tick — still oscillating
  then escalates (the fix landed; the flapping dependents need eyes), an
  oscillating target is a fail as before. Contract shapes unchanged; the
  receipt's `observed.recheck` records both looks.
- **Model lag vs telemetry (live, 21:31–21:35).** After any disturbance the
  trained model held the dependent `app` in `oscillating`/`degraded` for
  ~150s while the scorer's ground truth (raw Prometheus) read healthy the
  whole time; at the 60s window the verifier read `app=degraded`, failed,
  compensated (restarted the just-fixed service) and reopened — two of four
  lab scenarios failed on this alone. SABLE broadcasts only the aggregate
  `accuracy`; the per-node reading lives in `engine.history` and surfaces
  as `truth` on `GET /api/node/{idx}`, so `sable_bridge/ground_truth.py`
  fetches it once per tick when a verifier asks (`Loop.latest_tick`), and
  the stub emits `ground_truth` on the tick + `truth` in its node report.
  Verifier semantics: pass still needs the model healthy; model non-healthy
  with telemetry healthy is a disagreement → inconclusive, no compensation,
  re-checked every 15s against a newer tick up to
  `CONSOLE_VERIFY_MAX_SETTLE_S` (180) — first model-healthy look passes,
  the deadline escalates; only model-and-telemetry non-healthy (or a
  `failed` target, or `unreachable` with telemetry agreeing) is a fail.
  Unknown telemetry (replay ticks) is no evidence against the model.
  `observed.recheck = {count, seconds, first, last}`. Tests:
  `tests/test_ground_truth.py`, `tests/test_verify.py`.
- `requests` added to `console/requirements.txt` (the stub's Nemotron bridge
  imports it; `make test` failed on a fresh box).

## Not in the MVP
Multi-site policy, Graph/M365 adapters, signed (not just chained)
receipts, prompt caching, an eval set of scenarios, per-site egress policy.
