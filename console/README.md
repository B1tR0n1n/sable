# OVERLORD × SABLE Console

A closed remediation loop for infrastructure:

```
DETECT (SABLE) → PROPOSE (planner) → GATE (policy) → EXECUTE (OVERLORD) → VERIFY (SABLE) → RECEIPT
                                                                               │
                                                fail → compensate via OVERLORD ◄┘
```

Three roles, never blurred: **SABLE's models are the source of truth**, the
LLM is an **analyst that only proposes**, and **OVERLORD is the only thing
that changes anything** — transactionally, at a declared scope, on the record.

Everything speaks three contracts — `Finding`, `Plan`, `Receipt`
(`contracts/`, JSON Schemas in `contracts/schema/`). The build plan is
`docs/PLAN.md`; what was found in both engines before building is
`docs/INTEGRATION-NOTES.md`; the server's API is `docs/API.md`; what each
phase delivered and left open is `docs/PHASES.md`.

## Layout

| path | phase | what |
|---|---|---|
| `contracts/` | 1 | Finding / Plan / Receipt (pydantic v2, `extra="forbid"`), receipt hash chain |
| `topology.py` | 1 | the dependency graph (source depends on target); dependents, blast radius |
| `sable_bridge/` | 2 | SABLE client (REST + `/ws`), tick → Finding mapper, dedup store, emitter, findings API |
| `../docker/claude_bridge.py` | 2B | Claude as SABLE's analyst (`SABLE_LLM=claude`); local Nemotron path unchanged |
| `catalog/` | 3 | the action catalog (`actions.yaml`): params, reversibility, compensation, checks, bindings |
| `planner/` | 3 | template planner; LLM planner through OVERLORD's tool-less `complete()`; strict validation |
| `executor/` | 4 | one OVERLORD session per step, minimum scope, `cause` on provenance, reverse-order compensation, receipts |
| `verify/` | 5 | pass → close; fail → compensate + reopen; inconclusive → escalate, never a pass |
| `policy/` | 6 | the autonomy matrix; shipped default is every cell `human`; `policy.target.yaml` is the opt-in |
| `server/` | 7 | the loop as one process + the API + WebSocket; serves the UI |
| `ui/` | 7 | the operator screen (React + Vite, b1tr0n1n identity) |
| `lab/` | 8 | docker-compose world, faults, Prometheus config for SABLE, `e2e.py` |
| `tests/` | all | pytest; `make -C console test` |

## Running it live (WSL2 / Linux)

Four processes on one host. Read `lab/README.md` first — it names the two
SABLE facts that matter (its topology directory is hard-coded and mapped by
position; its live monitor needs the lab's `node_map`).

```bash
# 0. once
pip install -r console/requirements.txt          # console
pip install -r docker/requirements.txt           # SABLE server deps (+ torch from its image or your env)
(cd console/ui && npm install && npm run build)  # the UI → console/ui/dist, served by the server

# 1. the lab
make -C console lab-up                           # dns app db db-replica proxy prometheus blackbox
make -C console lab-status

# 2. SABLE, watching the lab, with Claude as the analyst
cp console/lab/topology.yaml adapters/topologies/00-lab.yaml   # sorts first → SABLE maps the lab (see lab/README.md)
export ANTHROPIC_API_KEY=…  SABLE_LLM=claude
python3 docker/server.py                          # :8080
python3 -m console.lab.live_monitor_lab            # Prometheus → /api/live_tick

# 3. OVERLORD's broker (the executor talks to this socket)
overlord daemon                                   # ~/.overlord/overlordd.sock by default
#    optional per-target policy (jail/net ceilings, commit-time content checks):
#    ~/.overlord/policy.json  →  {"targets": {"<lab dir>": {"net": "none", "checks": {"secrets": "block"}}}}

# 4. the console
export SABLE_URL=http://127.0.0.1:8080  CONSOLE_LAB=1  SITE_ID=lab
python3 -m console.server                         # http://127.0.0.1:7780
```

Then break something and watch the loop:

```bash
make -C console lab-fault FAULT=stop_service      # dns goes down → finding → plan (restart_service) → gate: human
# approve in the UI, or:
curl -s localhost:7780/api/findings?status=open | jq '.[0].id'
curl -s -X POST localhost:7780/api/plans/<plan_id>/approve -d '{"actor":"keith","decision":"approve"}' -H 'content-type: application/json'
# → executes through OVERLORD → verifies against a fresh SABLE tick → receipt (pass) → finding closed
make -C console lab-fault FAULT=corrupt_config    # app degraded → set_config_value back to golden → an OVERLORD-reversible file edit
```

The whole loop, unattended, positive and negative scenarios:

```bash
make -C console lab-e2e        # starts the console itself (CONSOLE_LAB=1); needs SABLE + OVERLORD running
```

## Autonomy

Shipped default: **every plan needs a human.** The target matrix
(`policy/policy.target.yaml` — high-confidence reversible plans run
automatically, compensable ones after a countdown, low confidence is
report-only) is opt-in: `CONSOLE_POLICY=console/policy/policy.target.yaml`,
or "Load target matrix" in the UI. An `unmonitored_gap` finding (the
diagnosis rests on absent telemetry) is always capped one band below its
score; an irreversible plan is never automatic.

## What is recorded

Every step runs in an OVERLORD session with `cause={plan_id, step_id,
action_id, finding_id}`, so `overlord log <sid>` names the plan step on
every path it touched; the session is the snapshot (`overlord revert <sid>`
undoes a committed step as a new reviewable session). Receipts form a
sha256 chain (`GET /api/receipts/verify`) and each closing entry is mirrored
onto OVERLORD's keyed, witnessed audit chain (`overlord audit verify`), as
are approvals and policy changes.

## Tests

```bash
make -C console test                    # 244 pytest cases, no network, no torch, no docker
(cd console/ui && npm test)             # 40 Vitest cases
```
