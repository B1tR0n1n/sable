# Console server API (Phase 7 contract)

The console server (`console/server/`, FastAPI, default `127.0.0.1:7780`)
orchestrates the loop and is the ONLY thing the UI talks to. It proxies to
SABLE (`SABLE_URL`, default `http://127.0.0.1:8080`) and drives OVERLORD
through its daemon socket (`OVERLORD_SOCKET`). Every object on the wire is a
Phase 1 contract (Finding / Plan / Receipt) serialised with `model_dump(mode="json")`.

## REST

| Method | Path | Body → Result |
|---|---|---|
| GET | `/api/state` | loop-state strip: `{sable:{url,ok,provider,device}, overlord:{socket,ok,version}, counts:{findings_open,plans_pending,receipts}, policy:{matrix,bands,default_deny}}` |
| GET | `/api/findings?status=open` | `[Finding]` newest first |
| GET | `/api/findings/{id}` | `Finding` |
| GET | `/api/findings/{id}/plan` | current `Plan` (gate annotated) or 404 |
| POST | `/api/findings/{id}/plan` | `{planner:"template"\|"llm"}` → `Plan` (gated); 422 if the planner's output failed validation (never repaired) |
| GET | `/api/plans/{id}` | `Plan` |
| POST | `/api/plans/{id}/approve` | `{actor, decision:"approve"\|"reject"\|"hold"\|"execute_now"}` → `{approval, plan, started:bool}`; approve/execute_now start execution when the gate allows it |
| GET | `/api/receipts?finding_id=` | `[Receipt]` newest first |
| GET | `/api/receipts/{id}` | `Receipt` |
| GET | `/api/receipts/{id}/export?format=json\|md` | the receipt as a file download |
| GET | `/api/receipts/verify` | `{ok, broken_at}` — the receipt hash chain |
| GET | `/api/policy` | the autonomy matrix and bands |
| PUT | `/api/policy` | replace the matrix (operator opt-in; audited) |
| GET | `/api/topology` | proxied from SABLE |
| GET | `/api/log?limit=200` | recent session-log lines `[{ts, level, source, text}]` |
| POST | `/api/analyst/chat` | `{finding_id, message, history}` → `{reply, tool_calls:[{name,input}], provider, generated:true}` |
| POST | `/api/lab/fault` | `{name}` → runs `console/lab/faults/<name>.sh` (dev only; `CONSOLE_LAB=1`) |

Errors: `{"error": "<message>"}` with 4xx/5xx.

## Authentication

Set `CONSOLE_TOKEN` and every mutating route (`POST`/`PUT`: plan, approve,
policy, analyst chat, lab fault) requires `Authorization: Bearer <token>`;
a missing or wrong bearer is `401 {"error": "..."}`. `GET` routes and `/ws`
stay open (read-only). Unset — the lab default — everything is open and the
server logs one warning line at startup:
`console: CONSOLE_TOKEN unset — mutating routes are unauthenticated`.
The UI sends the header when a token is stored (`localStorage["console_token"]`);
a link carrying `?token=…` or `#token=…` stores it on load and strips it from
the address bar.

## WebSocket `/ws`

One JSON message per event, `{"type": T, "ts": ISO, ...payload}`; on connect
the server sends `{"type":"hello","state":<GET /api/state>}`.

| type | payload |
|---|---|
| `finding` | `{finding}` (new or updated) |
| `plan` | `{plan}` (created or gate changed) |
| `approval` | `{plan_id, approval}` |
| `countdown` | `{plan_id, remaining_s}` (a `delay` decision, once per second) |
| `step` | `{plan_id, step}` (`StepResult`) |
| `verification` | `{plan_id, receipt_id, result}` |
| `receipt` | `{receipt}` |
| `log` | `{line}` |
| `state` | `{state}` (the strip, on any change) |
| `escalation` | `{finding_id, plan_id?, receipt_id?, decision:"human_plus", reason}` — an inconclusive verification, or the re-plan cap |

## Loop semantics the server implements

1. **Finding** arrives from the SABLE bridge (Phase 2) → stored/deduplicated → `finding` event.
2. **Plan**: the template planner runs automatically for a new finding; `llm` on request. The gate (Phase 6) annotates it → `plan` event.
3. **Gate**: `auto` → execute now; `delay` → countdown, operator may `execute_now`/`hold`/`reject`; `human`/`human_plus` → wait for `approve`; `report_only` → never executes.
4. **Execute** (Phase 4) through OVERLORD; each step → `step` event; log lines → `log`.
5. **Verify** (Phase 5) after `verification.window_s` (the catalog may pin it per action; `restart_service` waits 60s). Two readings per node: the model's state (SABLE — a pass needs it `healthy` across the scope) and the health scorer's ground truth from raw telemetry (`tick.ground_truth`, fetched from SABLE's `/api/node/{idx}` when the broadcast omits it; the stub carries it). pass → finding `closed`, receipt attached; fail (model non-healthy **and** telemetry non-healthy or unknown; a `failed` target) → compensations run, finding `reopened`; inconclusive → `human_plus` escalation, never a pass. Model non-healthy while telemetry is healthy is a **disagreement**, not a failure: nothing is compensated, the verifier re-checks every 15s against a newer tick for up to `CONSOLE_VERIFY_MAX_SETTLE_S` (180s) — the first look with the model healthy is a pass, the deadline is inconclusive; `observed.recheck` records `{count, seconds, first, last}`.
6. **Receipt** sealed into the hash chain and mirrored onto OVERLORD's audit chain (`receipt.close`) → `receipt` event.
7. **Reopened → re-plan**: a `reopened` finding gets a fresh template plan (gated like any other) — up to `CONSOLE_MAX_ATTEMPTS` (2) failed receipts, then it becomes `escalated` (`escalation` event, log line) and the loop stops proposing; an operator may still `POST …/plan`.
8. **Resolved without action**: an open/reopened/escalated finding whose root cause reads `healthy` for `CONSOLE_RESOLVE_TICKS` (3) consecutive ticks, with no plan executing, becomes `resolved`; its pending plan is withdrawn (a countdown stops, approvals no longer start it) and `GET …/plan` is 404. A resolved or closed finding never absorbs a later occurrence — the same fault again is a new finding.

Finding `status` values: `open`, `reopened`, `escalated` (still open — `counts.findings_open`), `closed` (verified fix, receipt attached), `resolved` (no action taken). The UI lists `closed` and `resolved` under the closed filter.
