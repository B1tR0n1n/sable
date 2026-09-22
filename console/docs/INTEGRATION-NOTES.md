# INTEGRATION-NOTES — OVERLORD × SABLE (Phase 0 reconnaissance)

> Read-only reconnaissance of both engines against PLAN.md. Every claim cites `path:line`. No code was changed in either repository.

## Part A — OVERLORD (actuation layer)

Read at `B1tR0n1n/overlord` HEAD `46b2c45`, `VERSION = "0.29.0"` (`overlord.py:2994`).
Dependency-free Python; the engine is one module (`overlord.py`) plus siblings
(`agent.py`, `providers.py`, `review.py`, `mcp.py`, `audit.py`, `netproxy.py`,
`policycheck.py`, `sdk/overlord_client.py`).

### A1. How a session opens

- `open_session(target, backend, grants, trace=None, wait=False, stack=False, capture=False, agent=None, owner=None) -> LiveSession` — `overlord.py:1614`.
  Realpaths and validates the target, picks a backend (`detect_backend()`, kernel or fuse),
  **refuses to open if a pending session already exists for the target** unless
  `stack=True` (`:1629-1634`), takes a per-target lock (`acquire_target_lock(target, wait)`, `:540`, called `:1635`),
  snapshots the tree (`snapshot_manifest(target)` → `{rel: [size, mtime_ns]}`, `:551`),
  optionally keeps a base copy for three-way merge (`grants["merge_base"]`, `:1645`),
  then launches the holder process that owns the mount namespace (`_launch_holder`, `:1660`).
- Session record (`meta.json`, shape at `:1653-1660`):
  `{id, target, cmd, execs[], backend, grants{}, trace, agent, owner, layers[{n, started}], started, status}`.
  `status` is a three-state machine: `open` → `pending` (closed, reviewable) → `committed`;
  rollback deletes the record. An `open` session whose holder died is reconciled to
  `pending` on next load (`reconcile_session`, `:1605`).
- One-shot: `execute_session(target, cmd, backend, grants, ...) -> (sid, exit_code, changes)` — `:2265`.
- Reopen a pending session to keep working on its stack: `reopen_session(sid, wait=False, capture=False)` — `:1663`.

### A2. How scope and permissions are declared (grants)

- Grant keys (`load_grants(args)`, `:1088-1113`): `net` ∈ `host|none|proxy`, `jail` bool,
  `timeout` seconds, `merge_base` bool, `connector_shell` bool, `net_allow` list of
  host / `*.suffix` patterns, `limits` dict (`LIMIT_KEYS`, `:1066` — cpu/mem/pids/disk),
  and `connectors` (MCP server names, set by the agent loop, `agent.py:470-474`).
- **Policy is a ceiling, never a loosening**: `resolve_policy(target, requested) -> (effective, rule)`
  — `:3019`. Reads `~/.overlord/policy.json` (`POLICY_FILE`, `:2996`; re-read per request,
  `load_policy`, `:2999`), matches the longest target prefix else `default`
  (`_policy_rule`, `:3008`), and **refuses a target with no rule** (`:3029`). A rule can force
  `jail`, force `net=none`, cap `timeout`, floor `limits`, strip `connector_shell`, and cap
  which MCP connectors may be granted (`_policy_connectors`, `:3221`).
- Enforcement: the jail is `pivot_root` + user namespace + seccomp + dropped caps (kernel
  backend only); `timeout` is a `threading.Timer` deadline that kills the process group
  (`:1288-1293`, `_kill` `:1387`, `expired` `:1394`); `net=proxy` is an empty network
  namespace whose only egress is the in-process recording proxy
  (`netproxy.py`; wired at `:1977-1978`: `Allow(grants["net_allow"])` + `Recorder(path=<session>/egress.jsonl)`).
- Commit-time policy (same file, per rule): `protect` globs force a countersignature
  (`protected_hits`, `:2540`); `checks` scan diff content — secrets / binaries / dependency
  manifests / size — with per-check action `block|countersign|warn`
  (`policy_gate`, `:2549`; `policycheck.evaluate`, `policycheck.py`); `require_review`,
  `allow_force` (`_api_commit`, `:3069-3078`).

### A3. How actions are recorded

Everything a session does lands under `~/.overlord/sessions/<sid>/` (`SESSIONS_DIR`, `:94`):

| file | writer | shape |
|---|---|---|
| `meta.json` | throughout | see A1; `execs[]` = `{id, cmd, label, layer, exit_code, timed_out, ...}` (`:1428-1430`); `layers[]` = one entry per writing command, stamped `exec, label, cmd` and optionally **`cause`** (`:1423-1426`) |
| `manifest.json` | open | pre-snapshot `{rel: [size, mtime_ns]}` — drift detection at commit |
| `provenance.jsonl` | close / commit | `build_provenance` (`:855`): per path `{ts, kind, path, layer?, before_sha256?, after_sha256?, after_size?}` + `caused_by` copied from the layer's `cause` (`_write_provenance`, `:1548-1565`); `before_retained`/`after_retained` flags after commit (`:2491-2497`) |
| `output.log` | exec | combined stdout/stderr |
| `egress.jsonl` | net=proxy | one line per connection `{ts, host, port, method, allowed, up, down[, error]}` (`netproxy.Recorder.note`) |
| `syscalls.jsonl` / `raw.strace` | `--trace` | parsed syscall events (`_finalize_session`, `:1571-1580`) |
| `review.jsonl` | `overlord review` | reviewer events + verdict (`review.py:262-268`) |
| `layers/` | exec | the overlay stack, one upper dir per savepoint |

- **`cause` is the hook for plan attribution.** `LiveSession.exec(cmd, timeout=None, capture=True, cwd=None, on_output=None, label=None, cause=None) -> (rc, output_bytes)` — `:1399`. `cause` is a free-form dict stamped on the layer the command writes into and surfaces on every affected path's provenance record as `caused_by`. The agent loop uses it for `{turn, tool, summary}`; an executor can use it for `{plan_id, step_id, action_id}` with no engine change.
- Content retention for blame: `store_object(path)` (`:879`; `OBJECTS_DIR` `:98`, `OBJECT_MAX` `:99`) keeps before/after copies content-addressed under `objects/<sha256>` (≤ `OBJECT_MAX`); `blame_path(path)` (`:2773`) answers which session/turn/tool/instruction wrote each line.
- **Tamper-evident audit chain** (`audit.py`): `record(action, **fields)` (`:122`) appends
  `{ts, action, actor, seq, prev, v, hash, ...fields}` to `~/.overlord/audit.jsonl`, each hash
  MAC-keyed with `~/.overlord/audit.key` (`audit_key`, `:50`; `_link`, `:75`). `verify()`
  (`:178`) walks the chain and reports `keyed` / downgrade; `head()` (`:218`); an off-box
  witness can hold the head (`send_checkpoint` `:283`, `verify_against_witness` `:319`,
  auto-checkpoint on configured actions `:336`). Webhooks subscribe to actions
  (`notify.dispatch`, `:147`).
  Actions the engine emits today: `session.open` / `session.reopen` (`overlord.py:2000`), `session.needs_review`
  (`:1542`), `session.commit` (`:2512`), `session.commit_refused` (`:2443`),
  `session.rollback` (`:2587`), `session.rewind` (`:2073`), `session.fork` (`:2139`),
  plus `connector.call / connector.decision / connector.result` from the agent loop.

### A4. What rollback exists today — read this carefully

- `rollback_session(sid) -> target` — `:2572`. Kills the holder if the session is still
  open, **deletes the session directory**, audits `session.rollback`. That is the whole
  operation: it discards a *pending* overlay. **The target was never touched**, because
  nothing reaches the real tree before `commit_session`.
- `rewind_session(sid, to)` (`:2064`) drops layers above a savepoint on a still-open or
  pending session; `fork_session(sid, at)` (`:2085`) copies a stack prefix into a new session.
- `commit_session(sid, merge=False, force=False, only=None, drop=None, countersigned=False) -> dict`
  — `:2393`. Order of gates: pending check → countersignature freshness
  (`review.review_state`, `review.py:95`, fingerprint-bound to exactly what would be
  replayed) → protected paths → content policy gate → whiteout sanity → drift conflicts
  (`find_conflicts`, `:733`) → provenance → replay (`apply_layers`) → retain objects →
  `status=committed` → audit. Refusals return `{"committed": False, ...}` with
  `conflicts` / `rejected` / `policy` populated; `--force` overrides conflicts, a rejection,
  and policy `block` (audited as `forced` / `overrode_rejection` / `policy_forced`), but
  never protected paths.
- **Therefore OVERLORD's "rollback" is pre-commit discard, scoped to the filesystem.**
  There is no post-commit revert operation. The material to build one exists
  (`before_sha256` + retained before-objects in provenance), but no `overlord revert <sid>`
  op is implemented. And for side effects outside the tree — a restarted service, a
  failed-over replica — OVERLORD contains and records the command but cannot undo its
  effect. See gaps G1/G2.

### A5. The built-in agent loop

- `run_agent(live, provider, task, max_turns=40, emit=None, should_stop=None, resume=False, note=None, connectors=None, approval=None, approve=None) -> str`
  — `agent.py:449`. Drives a model against an open `LiveSession`; every tool call runs
  *inside the transaction* via `live.exec(..., cause={turn, tool, summary})`.
  Emits events `task, assistant_delta, assistant, tool_call, tool_result, approval,
  connectors, skills, memory_suggestion, skill_use, resume, done{reason: end_turn|max_turns|budget|limit|cancelled|refusal|max_tokens}, error`
  (`agent.py:530-655`).
- **Fixed tool set** `TOOLS` (`agent.py:66`): `list_dir, read_file, write_file, shell`
  + `remember` + `skill`. There is no parameter to restrict or replace the tool set.
- Providers: `make_provider(provider, model=None, key=None, ...)` (`agent.py:170`) →
  `providers.py`: `anthropic`, `openai` (Responses API by default, `--api chat` to switch),
  `openai-compatible` (any base URL — local models), `scripted` (deterministic, for tests).
  Keys from env or the engine's 0600 key store.
- MCP connectors (`mcp.py`): `Registry(names, allow_shell)` (`:366`); calls run **on the host,
  outside the transaction**, gated by `approval ∈ ask|auto|readonly` (`:50`) with
  `is_read_only(tool)` (`:350`) and a per-call `call_fingerprint` (`:337`); every call,
  decision and result hash is audited.
- Countersignature: `review.run_review(sid, provider, max_turns, emit, allow_same=False)`
  (`review.py:236`) builds a dossier of the pending diff, **requires a model independent of
  the one that did the work** (`independent`, `:231`), and records a verdict
  (`approve|reject` + reason + `truncated`) into `meta["reviews"]` and `review.jsonl`.
  `review_state` (`:95`) binds the verdict to a fingerprint of the exact replay set.

### A6. Programmatic surface

- Daemon: `overlord daemon` serves a Unix socket; `DAEMON_OPS` (`overlord.py:3299`):
  `ping, run, open, close, diff, log, commit, rollback, sessions, agent_cancel, transcript,
  savepoints, rewind, blame, fork, models, compare`; streaming `STREAMING_OPS` (`:3322`):
  `exec, agent, resume, review`. `_api_open` (`:3170`) accepts an arbitrary `grants` dict
  and runs it through `resolve_policy` before opening — so the daemon is the policy broker.
- SDK (`sdk/overlord_client.py`): `OverlordClient(socket_path=None, timeout=None)`;
  `.open(target, jail=False, net="host", timeout=None, merge_base=False, trace=None, wait=False, stack=False, agent=None) -> LiveSession` (`:149`);
  `LiveSession.exec(cmd, timeout=None, cwd=None, label=None, on_output=None) -> (exit_code, output, changes)` (`:272`),
  `.shell(script)` (`:284`), `.diff()`, `.savepoints()`, `.rewind(to)`, `.close() -> Session`;
  `Session.commit(merge, force, only, drop, countersigned)` (`:50`), `.review(...)` (`:69`),
  `.rollback()` (`:83`), `.fork(at)`, `.rewind(to)`. Also `client.run(...)`, `.agent(...)`,
  `.blame(path)`, `.transcript(sid)`, `.sessions()`.

### A7. Runtime constraints

- Linux only. `jail` and `net` grants require the **kernel backend** (unprivileged user
  namespaces; on Ubuntu 24.04+ the shipped AppArmor profile grants them to the `overlord`
  launcher). The **fuse backend** (`fuse-overlayfs`) is cooperative: overlay only, no jail,
  no net grants, best-effort savepoints. `overlord doctor` reports which is live.
- Inside a Docker container OVERLORD needs `--device /dev/fuse --cap-add SYS_ADMIN` (fuse
  backend); the kernel backend inside a container needs unprivileged userns on the host.
  This matters for Phase 8: if the executor runs *in* the lab compose network, plan for the
  fuse backend unless the host permits nested userns.
- WSL2: the per-session `systemd-run --user` scope is skipped automatically (`_systemd_user_ok`);
  limits fall back to rlimits (`_systemd_user_ok`, `:1693`).

### A8. Gaps against the plan (OVERLORD side) and proposed adjustments

- **G1 — "OVERLORD rollback" in the loop diagram is not what OVERLORD provides.**
  OVERLORD's rollback discards a pending overlay before commit; it is filesystem-scoped and
  cannot undo a service restart or a failover. *Adjustment:* define `Receipt.rollback` as
  **catalog compensations executed through OVERLORD in reverse order** (which Phase 4
  already specifies), and reserve `overlord rollback` for the narrow case where a step's
  effect is purely a file change still pending in its session. Keep the diagram honest:
  `fail → compensate (via OVERLORD) ◄`.
- **G2 — no post-commit revert.** If Phase 5 wants to undo an already-committed file
  change, a small additive `overlord revert <sid>` (replay `before_sha256` objects from
  provenance as a new session) is buildable from existing material. Proposed as a
  Phase 4/5 addition to OVERLORD, ~small, listed in that phase's summary.
- **G3 — the LLM-assisted planner should not use `run_agent`.** `run_agent` hands the model
  `shell`/`write_file` and has no way to restrict the tool set (`agent.py:66`, `:449`).
  A planner that must only emit a validated Plan should call the provider directly
  (`providers.py` completion, or a JSON-only prompt) and validate against the catalog and
  schema — no execution tools at all. This also satisfies ground rule 4 structurally.
  *Optional additive change:* a `tools=` parameter on `run_agent` if the planner ever needs
  read-only lookups.
- **G4 — SDK `open()` does not expose `net_allow`, `limits`, or `connectors`** (`sdk/overlord_client.py:149`),
  though `_api_open` accepts any grants dict (`overlord.py:3173`). One-line additive SDK
  change; needed if the executor wants `net=proxy` with an allowlist per action.
- **G5 — step attribution is free-form.** Use `LiveSession.exec(..., cause={...})` (engine)
  — the SDK's `exec` exposes `label` but **not `cause`** (`:272`). Additive SDK change:
  pass `cause` through `_api_exec` (`overlord.py:3182` — currently forwards `label`, `cwd`,
  `timeout` only). Until then, `label=f"{plan_id}/{step_id}/{action_id}"` is recorded on
  the exec and layer.
- **G6 — Receipt hash chain already exists in a different place.** `audit.record` gives
  every entry `prev`/`hash` under a MAC key with off-box witnessing. *Adjustment:* emit
  receipt events (`receipt.step`, `receipt.verify`, `receipt.close`) through
  `audit.record` so receipts inherit the verified chain and `overlord audit verify`; store
  the audit `seq`/`hash` of the closing entry in `Receipt.receipt_hash` rather than
  inventing a parallel chain. `prev_receipt_hash` then = the previous receipt's closing
  audit hash.
- **G7 — approvals.** OVERLORD has two approval primitives that map to the gate:
  the countersignature (`review`) for `human_plus`-style second-model sign-off, and the
  MCP approval gate (`ask|auto|readonly`) for connector actions. A *human* approval record
  is not a first-class object today — it is the operator running `commit`. The console's
  `approvals[]` should be written by the console and mirrored into the audit chain
  (`audit.record("plan.approval", actor=..., decision=...)`).
- **G8 — minimum scope per plan.** Map the catalog's executor binding to grants:
  `jail=True`, `net=none` unless the action declares egress (then `net=proxy` +
  `net_allow` from the binding), `timeout=step.timeout_s`, and a `policy.json` rule for the
  lab target with `checks` enabled. The daemon enforces the ceiling, so the executor
  cannot request more than policy allows.

## Part B — SABLE (perception layer)

Read at `B1tR0n1n/sable` (shallow clone, default branch). Python ≥3.10 by syntax
(`dict | None` at `docker/sable_engine.py:119`); README says 3.11+ (`README.md:95`).
**No `requirements.txt`, `pyproject.toml`, `setup.py`, or test suite exist** — the only
dependency declaration is the Dockerfile (`docker/Dockerfile:1,6`).

### B1. What a diagnosis contains

Two engines exist; **only one is served.**

- **Served:** `class SableEngine` / `__init__(self, device="cuda")` — `docker/sable_engine.py:40-59`.
  Per-tick `infer(gnn, pomdp, mamba, ground_truth=None, mc_samples=0) -> dict` — `:117-120`.
  Tick dict (`:205-225`): `{cycle, nodes[], class_counts{state: n}, routing{}, avg_confidence, min_confidence{node,value}, max_confidence{node,value}, accuracy?, mc_samples?, mc_avg_agreement?, mc_avg_variance?}`.
  Per-node (`:179-196`): `{id: int, state, state_idx, confidence, probs{state: p}, transition{improving,stable,deteriorating}, trend, routing{GNN,POMDP,Mamba,Fusion}, mc_agreement?, mc_variance?}`.
  States: `STATE_NAMES = ["healthy","degraded","failed","unreachable","oscillating"]` — `sable_sim/core/states.py:15-29`.
- **Recommendations:** `get_recommendations(self) -> dict` — `docker/sable_engine.py:325`;
  returns `{summary, total_affected, actions[{priority, action, target, reason, recommendation}], root_cause: int|None}` (`:448-453`);
  needs ≥2 cycles (`:331-332`). Server enrichment adds `root_cause_label / root_cause_id / root_cause_type` and per-action `target_id / target_type` (`enrich_recommendations`, `docker/server.py:120-160`).
- **Root cause = earliest node to leave `healthy`.** `first_bad` per node (`docker/sable_engine.py:348`), bucketed by state (`:358-361`), `failed.sort(key=first_affected_tick)` (`:364`), `root_cause = failed[0]` (`:452`). **No topology reasoning is used** — this is a temporal heuristic, not causal inference. Affected = every non-healthy node (`total_affected`, `:433`), each `{node, state, first_affected_tick, n_state_changes, ticks_in_current_state}` (`:350-356`).
- **Confidence — where it really lives.** `get_recommendations` carries **no confidence field at all** (`:325-453`). Confidence exists only per node on the tick:
  - default: max softmax probability, `probs.max(dim=-1)` — `docker/sable_engine.py:157`;
  - MC dropout, when `mc_samples > 0`: `_mc_dropout_confidence(...)` — `:246-293`; `confidence = 0.6·agreement + 0.4·(1 − norm_variance)` (`:292-293`); exposed per node as `mc_agreement`/`mc_variance` (`:193-195`) and at tick level (`:222-225`). Toggled at runtime by `POST /api/mc_dropout` → global `mc_dropout_samples` clamped 0..20 (`docker/server.py:504-509`); visible in `/api/status` (`:215-216`).
- **Timing:** engine adds none; server sets `result["inference_ms"]` (`docker/server.py:412`). Live path sets `result["source"]="live"` (`:480`). **No engine-version field** — only `FastAPI(title="SABLE Engine", version="1.0")` (`:24`).
- **Not served:** `orchestrator/sable_core.py` `SABLEOrchestrator.diagnose(true_state, max_steps=5) -> DiagnosticResult` (`:184-186`; result fields `:103-124`: `root_cause_candidates`, `predicted_affected`, `predicted_severity`, `cascade_risk`, `confidence` = pillar-agreement heuristic `min(0.95, 0.5 + 0.15·overlap)` `:315-321`). CLI only. `orchestrator/cortex_repair.py` is **CORTEX thought-graph link repair, not infrastructure diagnosis** (`find_repairs`, `:173-202`; it *writes* via `insert_link`, `:105-126`).
- `SystemSnapshot` (`adapters/base.py:94-116`): `nodes: dict[str, NodeSnapshot]`, `edges: list[EdgeSnapshot]`, `timestamp`, `source`. `NodeSnapshot` (`:62-80`): `node_id, component_type, metrics{}, timestamp, labels{}, health?, state?, reachable=True, stale_seconds=0.0`. `EdgeSnapshot` (`:83-91`): `source_id, target_id, dep_type, criticality(HARD|SOFT|REDUNDANT), confidence=0.8, metadata`.
- `HealthScorer.score(node) -> (health, state)` — `adapters/health_scorer.py:102-150`; `unreachable` short-circuit (`:110-111`); **empty metrics ⇒ `(1.0, "healthy")`** ("assume healthy", `:113-114`); thresholds `≥0.8 healthy / ≥0.3 degraded / else failed` (`:138-143`).

### B2. Nemotron bridge — the real interface `claude_bridge.py` must match

`docker/nemotron_bridge.py` (288 lines):
- `__init__(self, llama_url=DEFAULT_LLAMA_URL, timeout=30.0)` — `:31-34`; sets `self.llama_url`, `self.timeout`, `self._session = requests.Session()`. **There is no `self.model`.**
- `is_available(self) -> bool` — `:36-42`; GETs `{llama_url}/health`, hard-coded `timeout=3`.
- `explain_tick(tick_result) -> str` — `:44`, ends `return self._complete(prompt)` — **`:93`**.
- `explain_recommendations(recs, max_tokens=300) -> str` — `:95`; reads the *enriched* keys `root_cause_label/root_cause_type` (`:101-105`); `result = self._complete(prompt, max_tokens=max_tokens)` — **`:126`**.
- `explain_scenario_complete(tick_history, recs) -> str` — `:129`; `return self._complete(prompt)` — **`:186`**.
- **Confirmed:** all three report methods build a prompt string and call `_complete()`. Inheriting them is sound.
- `chat(self, user_message, sable_context, history=None) -> str` — **`:188`**; `history` is a plain `[{role, content}]` list inserted verbatim (`:215-218`); **does not go through `_complete`** (POSTs `/v1/chat/completions`, `:221-231`); so `chat` must be overridden — it is.
- `_complete(self, prompt, max_tokens=512) -> str` — `:240`; trims to last sentence, capitalises (`:258-262`).
- Only external reader of bridge attributes: `docker/server.py:627` → `{"available": nemotron.is_available(), "url": nemotron.llama_url}`.

### B3. Server routes and the internals behind the read-only tools

- Globals at module scope: `engine = SableEngine(device="cuda")` — **`docker/server.py:27`**; `nemotron = NemotronBridge()` — **`:38`** (no env var read).
- Post-init hook point: `@app.on_event("startup") async def startup()` — `:163-191`: `engine.load_checkpoints` (`:167`) → `_load_topology_metadata()` (`:171`) → feedback DB (`:174`) → default scenario + `engine.reset_state` (`:178-189`). **After line 189** the model is loaded, `_topo_nodes/_topo_edges` are populated, and the engine is reset — that is where `register_tools` belongs.
- `/api/nemotron/*`: `status` (`:624-627`), `report` → `to_thread(nemotron.explain_recommendations, recs)` (`:630-643`), `chat(body)` → `history = body.get("history", [])` (`:650`) → `to_thread(nemotron.chat, user_message, recs, history)` (`:659-661`), `after_action` → `explain_scenario_complete` (`:669-694`; passes empty `class_counts` per tick, `:676-682`).
- **Chat history is client-owned**, not stored: browser `chatHistory` (`docker/dashboard.html:1477`), sends `history: chatHistory.slice(-10)` (`:1521-1523`). No server-side conversation store.
- Read-only internals (these become Claude's tools):
  - `GET /api/recommendations` → `async def recommendations()` = `enrich_recommendations(engine.get_recommendations())` — `:299-301`.
  - `GET /api/node/{idx}` → `async def get_node_detail(idx: int)` — `:290-296`: `engine.get_node_report(idx)` (`docker/sable_engine.py:300-323`) + `label/topo_id/component_type`. **No bounds check on `idx`** (contrast `/api/feedback`, `:552-553`).
  - `GET /api/topology` → `async def get_topology()` — `:319-333`: re-reads the first `*.y*ml` in `TOPOLOGY_DIR` (`:46`) each call → `{name, nodes[], edges[]}`.
  - Helpers: `node_label(idx)` `:86`, `node_id(idx)` `:94`, `node_type(idx)` `:101`, `_load_topology_metadata()` `:73-83`, `load_scenario(name)` with `_SAFE_NAME` guard `:41, :61-70`.
- Concurrency: routes take an `asyncio` `state_lock`; `chat`/`report` run the bridge in `asyncio.to_thread`. Tools called from that thread read `engine.history` without the lock (it is an asyncio lock, not usable from a thread).
- Bind: `uvicorn.run(app, host="127.0.0.1", port=8080)` — `:698`; `docker/run.sh` maps `-p 8080:8080` — a loopback bind inside a container is not reachable through that mapping.

### B4. Runtime

- CPU fallback **exists and is silent**: `docker/sable_engine.py:44-52` probes `torch.zeros(1, device="cuda")` and falls back to `cpu`; reported by `get_summary()["device"]` (`:462`).
- Dockerfile: `FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (`docker/Dockerfile:1`); `pip install fastapi uvicorn websockets torch-geometric h5py` (`:6`). It **does not** install `pyyaml`/`requests` (both imported: `docker/server.py:17`, `docker/nemotron_bridge.py:21`) and **does not COPY `nemotron_bridge.py`** (`:9-16`) although `server.py:22` imports it unconditionally. `anthropic` is nowhere in the repo.

### B5. Data, scenarios, topology

- `docker/precompute_scenarios.py` writes one `torch.save` dict per scenario to `docker/scenarios/<name>.pt` (`:463, :476-477`): `{name, description, n_nodes, n_ticks, gnn (1,T,N,D), pomdp, mamba, ground_truth (T,N)[, noise_profile]}` (`:352-359`, `:304-312`). Five bases (`monday_morning, silent_killer, cascade_whiplash, slow_poison, random_chaos`, `:479-500`) + noisy variants (`:510-518`). **`docker/scenarios/` is gitignored — the checkout has no scenario data**; it must be generated.
- **Every precomputed scenario uses a random topology**: `build_random_topology(rng, 20, 40)` — `:277, :326`. The server maps engine node index → infrastructure identity **positionally** into the first YAML in `TOPOLOGY_DIR` (`node_label/node_id/node_type`, `docker/server.py:86-106`). Therefore, for replayed scenarios, the `msp_demo.yaml` labels on findings are **cosmetic** — they do not describe the graph the scenario was generated on. Only the live path (`POST /api/live_tick`, `:418`) has a truthful mapping.
- `/api/scenarios` **hard-filters to `source == "smd"`** (`docker/server.py:237`); simulated scenarios (no `source` key ⇒ `"sable_sim"`, `:234`) are hidden from the API even when generated. `docker/precompute_smd.py` produces SMD-derived scenarios with `"source": "smd"` (`:249-261`).
- `COMPONENT_TYPES` (20) — `adapters/base.py:26-32`; `DEPENDENCY_TYPES` (9) — `:37-41`; `Criticality` — `:44-47`.
- `adapters/topologies/msp_demo.yaml`: `nodes[{id, type, label, tier}]` (31 entries; header says 32, `:4`) and `edges[{source, target, type, criticality}]` (`:7`, `:174`).
- Dependency graph — three representations, **none used by `get_recommendations`**: (1) YAML → `load_topology_from_file(path) -> list[EdgeSnapshot]` (`adapters/graph_builder.py:31`); (2) server flat lists `_topo_nodes/_topo_edges` (`docker/server.py:35-36`) — loaded, **never used for adjacency**; (3) simulator `InfrastructureGraph.get_dependents(component_id)` = predecessors, "source depends on target" (`sable_sim/core/graph.py:59-70`). **No server-side downstream-dependents computation exists.**
- `adapters/prometheus.py`: `PrometheusAdapter(config: PrometheusConfig)` (`:184-197`); `poll() -> SystemSnapshot(source="prometheus")` (`:199-229`); `discover_topology()` **always returns `[]`** (`:291-297`) — topology must come from YAML. `_mark_unreachable`: `up < 1` ⇒ `reachable=False`; no metrics ⇒ `reachable=False, stale_seconds=999` (`:280-289`). Default PromQL: cpu, mem, disk, disk io, net error rate, `up` (`:124-149`).

### B6. Detection mode — does not exist yet

No first-class `live_feed` vs `unmonitored_gap` concept. What exists: tick `source="live"` only on `/api/live_tick` (`docker/server.py:480`; replay ticks are unlabelled, `:400-413`); scenario-file `source` (`smd` / default `sable_sim`); `NodeSnapshot.reachable/stale_seconds` (`adapters/base.py:79-80`) collapsing into the `unreachable` state (`adapters/health_scorer.py:110-111`); and a simulator-only `FogOfWar` `unobservable_components` list (`sable_sim/simulation/fog.py:111-166`) that both call sites discard (`docker/precompute_scenarios.py:138`, `orchestrator/sable_core.py:216`).

### B7. Tests

None. Zero `pytest`/`unittest` imports; no `tests/`, `conftest.py`, or CI. Files named `*_test.py` are argparse demo/benchmark scripts (`orchestrator/fusion_test.py`, `fusion/adversarial_test.py`, `pillar1/domain_portability_test.py` — the last is also the real home of `InfrastructureAdapter`, imported by `adapters/encode.py:123`).

---

## Part C — `claude_bridge.py` against the real `NemotronBridge`

The reference file was written without running SABLE; checked line by line against B2/B3.

**Holds as written**
- `super().__init__(timeout=timeout)` matches `__init__(llama_url=..., timeout=30.0)` (`nemotron_bridge.py:31`); `llama_url` keeps its default.
- Setting `self.llama_url = f"anthropic:{model}"` is exactly right: the *only* external reader is `/api/nemotron/status` (`server.py:627`), so the dashboard's provider indicator works with **no server change**.
- Overriding `is_available()` is **required**, not optional — the base would `GET anthropic:claude-sonnet-5/health`.
- Overriding `_complete(prompt, max_tokens=512)` covers all three report methods (`:93, :126, :186`).
- Overriding `chat(user_message, sable_context, history=None)` is required (base `chat` bypasses `_complete`); the signature matches `:188`; the server's `history` is `[{role, content}]` strings from the browser, so the `role in ("user","assistant")` filter is correct and drops any stray `system`.
- `sable_context` is the enriched recommendations dict — the seeded system prompt is the right data.

**Needs adjustment before Phase 2B**
1. **Dockerfile.** `nemotron_bridge.py` is not copied and `anthropic`, `pyyaml`, `requests` are not installed (`docker/Dockerfile:6, :9-16`). Phase 2B task 1 must add `COPY claude_bridge.py nemotron_bridge.py` and the three packages, and create a `requirements.txt` (none exists).
2. **Tools must call sync internals, not the async routes.** `register_tools` should receive: `lambda: enrich_recommendations(engine.get_recommendations())`; a sync node detail = `engine.get_node_report(idx)` + `label/topo_id/component_type` (the body of `get_node_detail`, `server.py:290-296`) **with a bounds check** `0 <= idx < engine.n_nodes` (the route has none); and topology from `_topo_nodes/_topo_edges` (already loaded, `:35-36`) rather than re-reading YAML per call.
3. **Register after `startup()` line 189**, guarded by `isinstance(nemotron, ClaudeBridge)`.
4. **History normalisation.** If a browser request failed after pushing the user turn, `chatHistory` can carry two consecutive `user` entries. Merge consecutive same-role turns before sending; cheap insurance against a 400.
5. **`stop_reason == "max_tokens"`** returns truncated text silently; append a marker or raise `max_tokens` for reports (1024 is fine for chat).
6. **Thread safety.** Tools run in `asyncio.to_thread` and read `engine.history` without `state_lock`. Reads are safe in CPython but may straddle a tick; tools should copy what they read and never write — which also satisfies the plan's "no tool mutates engine state" test.
7. **Model choice.** `claude-sonnet-5` is a sound default for short operator chat. If reports need deeper reasoning, `claude-opus-5` with adaptive thinking is the current recommended default; make it an env choice, not a code change. No streaming is needed at 400–1024 tokens.
8. **`_complete` post-processing.** The base trims to the last complete sentence (`:258-262`); the Claude override only strips. Harmless, but reports may end mid-sentence on `max_tokens` — see 5.

---

## Part D — Gaps against the plan (both engines) and proposed adjustments

- **D1 (critical) — `Finding.confidence` has no source in SABLE's recommendations.** Proposal: take the root-cause node's per-node `confidence` from the latest tick; `method = "mc_dropout"` when `mc_dropout_samples > 0` (use `mc_agreement`, and carry `samples`), else `"engine_native"` (max softmax). **Be explicit in the notes and the UI that this is confidence in the node's *state classification*, not in the causal claim** — the root cause is "first to fail" (B1). The Phase 6 gate will be gating on classification confidence. Alternative: gate on MC agreement only and require `mc_dropout` on for any `auto` decision.
- **D2 (critical) — precomputed scenarios do not carry a real topology.** Their node labels are positional cosmetics over a random graph (B5). Phase 2's acceptance ("replaying a precomputed scenario produces stable Findings") is achievable, but blast radius, dependents, and any "upstream of X" chat answer will be **fiction on replayed data**. Proposal: use replay only for Finding-mapping/dedup tests; make the lab's live Prometheus path (Phase 8) the source for anything topological, with a lab YAML that mirrors the compose stack. `discover_topology()` returns `[]`, so YAML is the only topology source either way.
- **D3 — `detection_mode` must be synthesised.** Proposal: `live_feed` when the tick's `source == "live"`; `unmonitored_gap` when the root-cause node is `unreachable` or `stale_seconds > stale_threshold` (the diagnosis rests on *absence* of telemetry) — and note the scorer's "no metrics ⇒ healthy" default (B1), which can mask a gap as health.
- **D4 — no server-side dependents computation.** Blast radius should be computed in the console from `/api/topology` edges (`source depends on target` ⇒ dependents = predecessors), not added to SABLE.
- **D5 — `engine_version`.** Nothing exists; use the SABLE git SHA + checkpoint file hashes (`engine.load_checkpoints`, `server.py:167`).
- **D6 — `/api/scenarios` hides simulated scenarios** (`source != "smd"`). Phase 2 either calls `load_scenario(name)` directly or an additive query param lifts the filter. Scenarios must be generated first (`docker/scenarios/` is gitignored).
- **D7 — SABLE has no tests or dependency file.** Phase 2B's pytest acceptance means bootstrapping `tests/`, `conftest.py`, and `requirements.txt` in SABLE (additive). Engine tests need the checkpoints or a stub; bridge tests should mock `anthropic.Anthropic`.
- **D8 — OVERLORD rollback ≠ loop rollback** (Part A, G1/G2): compensation via OVERLORD, not `overlord rollback`.
- **D9 — planner ≠ `run_agent`** (G3): direct provider call + schema validation.
- **D10 — SDK passthrough** (G4/G5): `net_allow`, `limits`, `cause` on `open`/`exec` — small additive OVERLORD change for Phase 4.
- **D11 — Receipt chain** (G6): ride the existing keyed audit chain.
- **D12 — container bind.** SABLE binds `127.0.0.1` inside Docker (`server.py:698`); the lab needs `0.0.0.0` (env-driven, additive) or host networking.
- **D13 — `cortex_repair.py` is out of scope**: it is not diagnosis and it *writes* to Supabase; keep it out of the console's SABLE surface.

## Part E — Open questions for approval before Phase 1

1. **Where does `console` live?** A new repo (`B1tR0n1n/console`?) — I cannot create it from this session; you create it (or authorise push access) and I scaffold. Until then, notes are delivered as a file.
2. **Confidence semantics (D1):** accept "root-cause node's state-classification confidence, `engine_native` unless MC dropout is on"? Or require MC dropout for any `auto` gate cell?
3. **Replay vs live (D2):** accept that replayed scenarios are for mapping/dedup tests only, and topological features are validated in the lab?
4. **`detection_mode` rule (D3):** approve the `unreachable`/stale ⇒ `unmonitored_gap` mapping?
5. **Rollback wording (D8):** approve redefining the loop's "OVERLORD rollback" as catalog compensations executed through OVERLORD?
6. **Pre-approve small additive changes:** OVERLORD SDK passthrough (`cause`, `net_allow`, `limits`), optional `overlord revert`; SABLE Dockerfile/requirements fix, `register_tools` call, `0.0.0.0` bind env, node `idx` bounds check.
