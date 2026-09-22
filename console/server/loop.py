"""The closed loop, as one object.

    DETECT   SABLE ticks arrive (on_tick) → the emitter maps them to Findings
             in the store; a NEW open finding is planned at once.
    PROPOSE  TemplatePlanner by default; LLMPlanner on request, through
             OVERLORD's tool-less complete().
    GATE     the Policy annotates the plan: auto / delay / human /
             human_plus / report_only.
    EXECUTE  auto → now; delay → a countdown the operator can cut short,
             hold or reject; human/human_plus → an approve decision;
             report_only → never. Execution runs in a worker thread.
    VERIFY   the Verifier waits the window, takes the freshest tick, and
             closes / compensates+reopens / escalates.
    RECEIPT  sealed onto the chain, mirrored onto OVERLORD's audit chain.

Lifecycle around the loop: a finding whose root cause SABLE reports healthy
for `resolve_after_ticks` ticks in a row, with nothing executing for it,
resolves itself (no action, no receipt) and its pending plan is withdrawn.
A finding reopened by a failed verification gets a fresh plan — up to
`max_attempts` failed receipts, after which it escalates to a human instead
of looping fix / fail / fix.

Every state change is an event (emit) that the server broadcasts.
Nothing here imports FastAPI.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Any, Callable, Optional

from console.catalog import Catalog
from console.contracts import Approval, Finding, Plan, Receipt, now_utc
from console.executor import Executor, ReceiptChain
from console.planner import LLMPlanner, NoTemplate, PlanValidationError, TemplatePlanner
from console.policy import Policy
from console.sable_bridge import FindingEmitter, FindingStore, attach_ground_truth
from console.topology import Topology
from console.verify import Verifier, node_states

HERE = Path(__file__).resolve().parent
CONSOLE_DIR = HERE.parent


@dataclass
class Config:
    sable_url: str = os.environ.get("SABLE_URL", "http://127.0.0.1:8080")
    overlord_socket: Optional[str] = os.environ.get("OVERLORD_SOCKET")
    data_dir: Path = Path(os.environ.get("CONSOLE_DATA", str(Path.home() / ".overlord-console")))
    site_id: str = os.environ.get("SITE_ID", "lab")
    lab_dir: Path = Path(os.environ.get("LAB_DIR", str(CONSOLE_DIR / "lab")))
    lab_enabled: bool = os.environ.get("CONSOLE_LAB", "") == "1"
    policy_path: Optional[str] = os.environ.get("CONSOLE_POLICY")
    llm_provider: str = os.environ.get("CONSOLE_LLM_PROVIDER", "anthropic")
    llm_model: Optional[str] = os.environ.get("CONSOLE_LLM_MODEL")
    disabled_actions: tuple[str, ...] = tuple(
        a for a in os.environ.get("CONSOLE_DISABLE_ACTIONS", "").split(",") if a)
    verify_stale_after_s: int = int(os.environ.get("CONSOLE_VERIFY_STALE_S", "120"))
    verify_max_settle_s: int = int(os.environ.get("CONSOLE_VERIFY_MAX_SETTLE_S", "180"))   # model-lag re-check budget
    # lifecycle: healthy ticks before a no-action resolve; failed receipts before escalation
    resolve_after_ticks: int = int(os.environ.get("CONSOLE_RESOLVE_TICKS", "3"))
    max_attempts: int = int(os.environ.get("CONSOLE_MAX_ATTEMPTS", "2"))
    log_lines: int = 2000


@dataclass
class _PlanState:
    plan: Plan
    finding_id: str
    approvals: list[Approval] = field(default_factory=list)
    # proposed | countdown | held | rejected | executing | done | superseded | withdrawn
    status: str = "proposed"
    remaining_s: Optional[int] = None
    receipt_id: Optional[str] = None


class Loop:
    def __init__(self, cfg: Config, sable, overlord, catalog: Optional[Catalog] = None,
                 policy: Optional[Policy] = None, topology: Optional[Topology] = None,
                 store: Optional[FindingStore] = None, chain: Optional[ReceiptChain] = None,
                 emit: Optional[Callable[[dict], None]] = None, sleep: Callable[[float], Any] = time.sleep,
                 clock: Callable[[], datetime] = now_utc):
        self.cfg, self.sable, self.overlord = cfg, sable, overlord
        self.catalog = catalog or Catalog.load()
        self.policy = policy or (Policy.load(cfg.policy_path) if cfg.policy_path else Policy.load())
        self._emit = emit or (lambda ev: None)
        self._sleep, self._clock = sleep, clock
        # state that note()/emit() need must exist before anything can fail
        self.plans: dict[str, _PlanState] = {}          # plan_id -> state
        self.plan_of: dict[str, str] = {}               # finding_id -> current plan_id
        self._healthy_streak: dict[str, int] = {}       # finding_id -> consecutive ticks with a healthy root
        self._answered: dict[str, int] = {}             # finding_id -> receipts answered by a replan/escalation
        self.log: deque = deque(maxlen=cfg.log_lines)
        self._tick: tuple[Optional[dict], Optional[datetime]] = (None, None)
        self._threads: list[threading.Thread] = []
        self._lock = threading.RLock()
        self.stations: dict[str, float] = {}            # station -> last active ts (the strip)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = store or FindingStore(str(cfg.data_dir / "findings.json"))
        self.chain = chain or ReceiptChain(cfg.data_dir / "receipts.jsonl")
        self.topology = topology or self._load_topology()
        self.emitter = FindingEmitter(sable, self.store, cfg.site_id, topology=self.topology) if sable else None
        self.context = {"lab_dir": str(cfg.lab_dir), "lab_compose": str(cfg.lab_dir / "docker-compose.yml")}
        self.golden = self._load_golden()
        self.executor = Executor(overlord, self.catalog, self.context, emit=self.emit) if overlord else None
        self.verifier = Verifier(self.latest_tick, self.store, self.executor, self.chain, self.topology,
                                 emit=self.emit, stale_after_s=cfg.verify_stale_after_s, sleep=sleep,
                                 max_settle_s=cfg.verify_max_settle_s) if overlord else None
        self.store.subscribe(self._on_finding_change)

    # ---------------------------------------------------------------- plumbing

    def emit(self, ev: dict) -> None:
        ev = {"ts": self._clock().isoformat(), **ev}
        if ev.get("type") == "log":
            self.log.append(ev["line"])
        station = {"finding": "DETECT", "plan": "PROPOSE", "approval": "GATE", "countdown": "GATE",
                   "step": "EXECUTE", "rollback": "EXECUTE", "verification": "VERIFY",
                   "receipt": "RECEIPT"}.get(ev.get("type"))
        if station:
            self.stations[station] = time.time()
        self._emit(ev)

    def note(self, text: str, source: str = "console", level: str = "info") -> None:
        self.emit({"type": "log", "line": {"ts": self._clock().isoformat(), "level": level,
                                            "source": source, "text": text}})

    def _load_topology(self) -> Topology:
        if self.sable is not None:
            try:
                return Topology.from_api(self.sable.topology())
            except Exception as e:                    # noqa: BLE001 — fall through to the lab file
                self.note(f"SABLE topology unavailable ({e}); using the lab topology", level="warn")
        lab = self.cfg.lab_dir / "topology.yaml"
        if lab.is_file():
            return Topology.from_yaml(lab)
        return Topology([], [], "empty")

    def _load_golden(self) -> dict[str, dict[str, str]]:
        """lab/golden.yaml: node_id -> {file, key, value[, service]} — the
        known-good config the template planner restores for a degraded node."""
        p = self.cfg.lab_dir / "golden.yaml"
        if not p.is_file():
            return {}
        import yaml
        data = yaml.safe_load(p.read_text()) or {}
        return {str(k): {str(a): str(b) for a, b in (v or {}).items()} for k, v in data.items()}

    # ---------------------------------------------------------------- DETECT

    def on_tick(self, tick: dict[str, Any]) -> Optional[Finding]:
        """A SABLE tick: remember it (verification reads the freshest), let
        the emitter turn it into a finding, then resolve what has healed."""
        self._tick = (tick, self._clock())
        self.stations["DETECT"] = time.time()
        finding = self.emitter.on_tick(tick) if self.emitter else None
        self._sweep_healed(tick)
        return finding

    def latest_tick(self) -> tuple[Optional[dict], Optional[datetime]]:
        """The freshest tick, with the scorer's ground truth beside the model's
        states. SABLE broadcasts only the aggregate accuracy, so the per-node
        reading is fetched from /api/node/{idx} once per tick, the first time
        a verifier asks (the stub carries it on the tick already)."""
        tick, ts = self._tick
        if tick is None or "ground_truth" in tick or self.sable is None:
            return tick, ts
        enriched = attach_ground_truth(tick, self.sable)
        with self._lock:
            if self._tick[0] is tick:              # no newer tick landed meanwhile
                self._tick = (enriched, ts)
        return enriched, ts

    def _on_finding_change(self, finding: Finding) -> None:
        self.emit({"type": "finding", "finding": finding.model_dump(mode="json")})
        if finding.status == "open" and finding.id not in self.plan_of:
            self._propose(finding)
        elif finding.status == "reopened" and self._unanswered_reopen(finding):
            self._replan_or_escalate(finding)

    def _propose(self, finding: Finding, exclude_actions: Iterable[str] = ()) -> None:
        """Prefer an action the finding's failed attempts have not tried; when
        the template chain has nothing else, the same action goes again (a fix
        that was verified too early is the common case), up to the attempt cap."""
        exclude = set(exclude_actions)
        try:
            self.plan(finding.id, "template", exclude_actions=exclude)
            return
        except NoTemplate as e:
            if not exclude:
                self.note(f"no plan for {finding.id}: {e}", level="warn")
                return
            self.note(f"no untried template for {finding.id} ({e}); retrying the same action", source="gate")
        except PlanValidationError as e:
            self.note(f"no plan for {finding.id}: {e}", level="warn")
            return
        try:
            self.plan(finding.id, "template")
        except (NoTemplate, PlanValidationError) as e:
            self.note(f"no plan for {finding.id}: {e}", level="warn")

    # ---------------------------------------------------------------- lifecycle

    def _sweep_healed(self, tick: dict[str, Any]) -> None:
        """Resolve every open finding whose root cause has read healthy for
        `resolve_after_ticks` ticks in a row and has nothing executing. A
        transient blip must not stay open for ever and swallow the next real
        fault into its dedup key; a plan in flight is the verifier's to judge."""
        states = node_states(tick, self.topology)
        for f in self.store.list_open():
            healthy = states.get(f.root_cause.node_id) == "healthy"
            streak = self._healthy_streak.get(f.id, 0) + 1 if healthy else 0
            self._healthy_streak[f.id] = streak
            if streak >= self.cfg.resolve_after_ticks and not self._executing(f.id):
                self._resolve(f, streak)

    def _executing(self, finding_id: str) -> bool:
        pid = self.plan_of.get(finding_id)
        return pid is not None and self.plans[pid].status == "executing"

    def _resolve(self, finding: Finding, streak: int) -> None:
        """No action: withdraw the pending plan (a countdown stops, an approval
        no longer starts anything), drop the mapping, mark the finding resolved."""
        with self._lock:
            pid = self.plan_of.pop(finding.id, None)
            st = self.plans.get(pid) if pid else None
            if st is not None and st.status in ("proposed", "countdown", "held"):
                st.status = "withdrawn"
        self._healthy_streak.pop(finding.id, None)
        self._answered.pop(finding.id, None)
        self.note(f"finding {finding.id} resolved without action: {finding.root_cause.node_id} "
                  f"healthy for {streak} consecutive tick(s)"
                  + (f"; plan {pid} withdrawn" if st is not None and st.status == "withdrawn" else ""),
                  source="detect")
        self.store.resolve(finding.id)

    def _unanswered_reopen(self, finding: Finding) -> bool:
        """A reopen is answered once per failed receipt. A later occurrence of
        the same condition (the store's dedup refresh) carries no new receipt
        and is not a second reopen — even while the next attempt is executing."""
        return len(finding.receipt_ids) > self._answered.get(finding.id, 0)

    def _replan_or_escalate(self, finding: Finding) -> None:
        self._answered[finding.id] = len(finding.receipt_ids)
        attempts = self._failed_attempts(finding)
        if attempts >= self.cfg.max_attempts:
            self._escalate(finding, attempts)
            return
        tried = self._tried_actions(finding)
        self.note(f"finding {finding.id} reopened after failed attempt {attempts}/{self.cfg.max_attempts}; "
                  f"proposing a fresh plan (already tried: {sorted(tried) or 'nothing'})", source="gate")
        self._propose(finding, exclude_actions=tried)

    def _tried_actions(self, finding: Finding) -> set[str]:
        """Catalog actions the finding's failed attempts executed: from sealed
        receipts, plus the plan of the attempt that just reopened it (its
        receipt may not be in the chain yet)."""
        tried: set[str] = set()
        for rid in finding.receipt_ids:
            r = self.chain.get(rid)
            if r is not None:
                tried.update(s.action_id for s in r.steps if s.status in ("ok", "failed"))
        pid = self.plan_of.get(finding.id)
        if pid is not None and self.plans[pid].status in ("executing", "done", "failed"):
            tried.update(s.action_id for s in self.plans[pid].plan.steps)
        return tried

    def _failed_attempts(self, finding: Finding) -> int:
        """Receipts attached to a still-open finding are failed attempts (a
        pass closes it); the one that just reopened it may not be sealed yet."""
        n = 0
        for rid in finding.receipt_ids:
            r = self.chain.get(rid)
            if r is None or (r.verification is not None and r.verification.status == "fail"):
                n += 1
        return n

    def _escalate(self, finding: Finding, attempts: int) -> None:
        reason = (f"{attempts} failed attempt(s) for {finding.id} (max {self.cfg.max_attempts}); "
                  f"not re-planning — a human decides")
        self.note(f"finding {finding.id} escalated: {reason}", source="gate", level="warn")
        self.emit({"type": "escalation", "finding_id": finding.id, "plan_id": self.plan_of.get(finding.id),
                   "decision": "human_plus", "reason": reason})
        self._audit("approval.escalated", finding_id=finding.id, attempts=attempts,
                    max_attempts=self.cfg.max_attempts)
        self.store.escalate(finding.id)

    # ---------------------------------------------------------------- PROPOSE + GATE

    def _complete_fn(self):
        def complete(prompt: str, system: str, purpose: str) -> dict:
            return self.overlord.complete(prompt, system=system, provider=self.cfg.llm_provider,
                                         model=self.cfg.llm_model, purpose=purpose)
        return complete

    def plan(self, finding_id: str, planner: str = "template", force_action: Optional[str] = None,
             exclude_actions: Iterable[str] = ()) -> Plan:
        """(Re)plan a finding; the gate annotates the plan; a gated plan whose
        decision is auto/delay starts on its own. Raises PlanValidationError
        for an LLM plan that failed validation (never repaired), NoTemplate
        when no template applies."""
        finding = self.store.get(finding_id)
        if finding is None:
            raise KeyError(finding_id)
        if planner == "llm":
            if self.overlord is None:
                raise PlanValidationError(["no OVERLORD connection for the LLM planner"])
            plan = LLMPlanner(self.catalog, self.topology, self._complete_fn(), model=self.cfg.llm_model).plan(finding)
            disabled = [s.action_id for s in plan.steps if s.action_id in self.cfg.disabled_actions]
            if disabled:
                raise PlanValidationError([f"action disabled by CONSOLE_DISABLE_ACTIONS: {', '.join(disabled)}"])
        else:
            # a disabled action makes the template fall back (restart instead of
            # a config restore, say) — the lab's negative scenario is exactly that
            plan = TemplatePlanner(self.catalog, self.topology, golden=self.golden,
                                   disabled=self.cfg.disabled_actions).plan(finding, exclude_actions=exclude_actions)
        if force_action:
            plan = self._force_action(plan, finding, force_action)
        plan = self.policy.apply(finding, plan)
        with self._lock:
            old = self.plan_of.get(finding_id)
            if old and self.plans[old].status in ("proposed", "countdown", "held"):
                self.plans[old].status = "superseded"
            st = _PlanState(plan=plan, finding_id=finding_id)
            self.plans[plan.id] = st
            self.plan_of[finding_id] = plan.id
        self.emit({"type": "plan", "plan": plan.model_dump(mode="json"), "status": st.status})
        self.note(f"plan {plan.id} for {finding_id}: {[s.action_id for s in plan.steps]} → "
                  f"{plan.gate.decision} ({plan.gate.rule_id})", source="gate")
        self._act_on_gate(st)
        return plan

    def _force_action(self, plan: Plan, finding: Finding, action_id: str) -> Plan:
        """Lab negative test: swap the plan's action for another catalog
        action applicable to the target — deliberately the wrong fix."""
        from console.planner import expected_blast_radius, validate_plan
        step = plan.steps[0]
        spec = self.catalog.get(action_id)
        params = {k: step.params.get(k, step.target_node) for k in (spec.params.get("required") or [])}
        d = plan.model_dump(mode="json")
        d["steps"] = [{**step.model_dump(mode="json"), "action_id": action_id, "params": params,
                       "reversibility": spec.reversibility,
                       "compensation": (self.catalog.compensation_for_action(action_id, params).model_dump()
                                        if self.catalog.compensation_for_action(action_id, params) else None),
                       "precondition": spec.preconditions[0] if spec.preconditions else "none",
                       "timeout_s": spec.executor.grants.timeout_s}]
        d["verification"] = {"predicate": spec.verification, "window_s": plan.verification.window_s}
        nodes = expected_blast_radius([step.target_node], self.topology)
        d["blast_radius"] = {"nodes": nodes, "count": len(nodes)}
        for k in ("id", "created_at", "gate", "planner"):
            d.pop(k, None)
        return validate_plan(d, self.catalog, finding, self.topology)

    def _act_on_gate(self, st: _PlanState) -> None:
        decision = st.plan.gate.decision
        if decision == "auto":
            self._start(st, Approval(actor="policy", decision="approve"))
        elif decision == "delay":
            st.status, st.remaining_s = "countdown", int(st.plan.gate.delay_s or 0)
            t = threading.Thread(target=self._countdown, args=(st,), daemon=True)
            self._threads.append(t)
            t.start()
        elif decision == "report_only":
            self.note(f"plan {st.plan.id} is report-only; it will not execute", source="gate")

    def _countdown(self, st: _PlanState) -> None:
        while st.status == "countdown" and (st.remaining_s or 0) > 0:
            self.emit({"type": "countdown", "plan_id": st.plan.id, "remaining_s": st.remaining_s})
            self._sleep(1)
            if st.status == "countdown":
                st.remaining_s -= 1
        if st.status == "countdown":
            self._start(st, Approval(actor="policy:delay", decision="approve"))

    # ---------------------------------------------------------------- approvals

    def approve(self, plan_id: str, actor: str, decision: str) -> dict:
        st = self.plans.get(plan_id)
        if st is None:
            raise KeyError(plan_id)
        approval = Approval(actor=actor, decision=decision)
        with self._lock:
            st.approvals.append(approval)
        self.emit({"type": "approval", "plan_id": plan_id, "approval": approval.model_dump(mode="json")})
        self._audit("approval.recorded", plan_id=plan_id, finding_id=st.finding_id, actor=actor, decision=decision,
                    gate=st.plan.gate.decision, rule_id=st.plan.gate.rule_id)
        started = False
        gate = st.plan.gate.decision
        if decision == "reject":
            st.status = "rejected"
        elif decision == "hold":
            if st.status == "countdown":
                st.status = "held"
        elif decision in ("approve", "execute_now"):
            if gate == "report_only":
                self.note(f"plan {plan_id} is report-only; an approval does not execute it", source="gate", level="warn")
            elif st.status in ("proposed", "countdown", "held"):
                started = self._start(st, approval)
        return {"approval": approval.model_dump(mode="json"), "plan": st.plan.model_dump(mode="json"),
                "started": started, "status": st.status}

    # ---------------------------------------------------------------- EXECUTE + VERIFY

    def _start(self, st: _PlanState, approval: Approval) -> bool:
        if self.executor is None:
            self.note("no OVERLORD connection; cannot execute", level="error")
            return False
        with self._lock:
            if st.status == "executing":
                return False
            st.status = "executing"
            if approval not in st.approvals:
                st.approvals.append(approval)
        t = threading.Thread(target=self._run_plan, args=(st,), daemon=True)
        self._threads.append(t)
        t.start()
        return True

    def _run_plan(self, st: _PlanState) -> None:
        finding = self.store.get(st.finding_id)
        try:
            receipt = self.executor.execute(st.plan, finding, list(st.approvals))
            receipt = self.verifier.verify(st.plan, finding, receipt)
            st.receipt_id = receipt.id
            fresh = self.store.get(st.finding_id)
            if fresh is not None and receipt.id not in fresh.receipt_ids:
                self.store.upsert(fresh.model_copy(update={"receipt_ids": fresh.receipt_ids + [receipt.id]}))
        except Exception as e:                        # noqa: BLE001 — the loop never dies on one plan
            self.note(f"plan {st.plan.id} crashed: {e}", level="error")
        finally:
            st.status = "done"

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Join execution/countdown threads (tests and the e2e runner)."""
        deadline = time.monotonic() + timeout
        for t in list(self._threads):
            t.join(max(0.0, deadline - time.monotonic()))
        self._threads = [t for t in self._threads if t.is_alive()]
        return not self._threads

    # ---------------------------------------------------------------- read side

    def state(self) -> dict:
        sable = {"url": self.cfg.sable_url, "ok": False}
        if self.sable is not None:
            try:
                s = self.sable.status()
                nm = {}
                try:
                    nm = self.sable.get("/api/nemotron/status")
                except Exception:                     # noqa: BLE001
                    pass
                sable.update(ok=True, device=s.get("device"), cycle=s.get("cycle"),
                             provider=nm.get("provider", "local"), model=nm.get("model"))
            except Exception as e:                    # noqa: BLE001
                sable["error"] = str(e)
        ov = {"socket": self.cfg.overlord_socket, "ok": False}
        if self.overlord is not None:
            try:
                p = self.overlord.ping()
                ov.update(ok=True, version=p.get("version"))
            except Exception as e:                    # noqa: BLE001
                ov["error"] = str(e)
        now = time.time()
        return {
            "sable": sable, "overlord": ov,
            "counts": {"findings_open": len(self.store.list_open()),
                       "plans_pending": sum(1 for s in self.plans.values() if s.status in ("proposed", "countdown", "held")),
                       "receipts": self.chain.verify()["count"]},
            "policy": {"matrix": self.policy.matrix.model_dump(), "bands": self.policy.bands.model_dump(),
                       "default_deny": self.policy.is_default_deny, "delay_s": self.policy.delay_s},
            "stations": {k: (now - v) < 10 for k, v in self.stations.items()},
            "lab": self.cfg.lab_enabled,
        }

    def plan_for(self, finding_id: str) -> Optional[dict]:
        pid = self.plan_of.get(finding_id)
        if pid is None:
            return None
        st = self.plans[pid]
        return {**st.plan.model_dump(mode="json"), "_status": st.status, "_remaining_s": st.remaining_s,
                "_approvals": [a.model_dump(mode="json") for a in st.approvals], "_receipt_id": st.receipt_id}

    def set_policy(self, data: dict, actor: str = "operator") -> Policy:
        self.policy = Policy.model_validate(data)
        self._audit("approval.policy_changed", actor=actor, default_deny=self.policy.is_default_deny)
        self.emit({"type": "state", "state": self.state()})
        return self.policy

    # ---------------------------------------------------------------- analyst + lab

    def analyst_chat(self, finding_id: Optional[str], message: str, history: list[dict]) -> dict:
        if self.sable is None:
            return {"reply": "[SABLE unavailable]", "tool_calls": [], "provider": "none", "generated": True}
        scope = ""
        f = self.store.get(finding_id) if finding_id else None
        if f is not None:
            scope = (f"Regarding finding {f.id}: root cause {f.root_cause.node_id} ({f.root_cause.component_type}) "
                     f"is {f.root_cause.state}; affected: "
                     f"{', '.join(a.node_id for a in f.affected_nodes) or 'none'}; confidence {f.confidence.score:.2f} "
                     f"({f.confidence.method}); mode {f.detection_mode}.\n\n")
        r = self.sable.post("/api/nemotron/chat", {"message": scope + message, "history": history})
        nm = {}
        try:
            nm = self.sable.get("/api/nemotron/status")
        except Exception:                             # noqa: BLE001
            pass
        # SABLE's own facts ride in a ```sable fence so the UI can render them
        # apart from the model's inference — data first, generated text after
        reply = r.get("reply", "")
        if f is not None:
            facts = {"finding": f.id, "root_cause": f.root_cause.model_dump(), "affected": [a.node_id for a in f.affected_nodes],
                     "confidence": f.confidence.model_dump(), "detection_mode": f.detection_mode, "severity": f.severity}
            reply = "```sable\n" + json.dumps(facts, indent=2) + "\n```\n" + reply
        return {"reply": reply, "tool_calls": r.get("tool_calls", []),
                "provider": nm.get("provider", "local"), "generated": True}

    _ARG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")

    def lab_fault(self, name: str, args: Optional[list[str]] = None) -> dict:
        """Run console/lab/faults/<name>.sh [args] — a whitelist of the scripts
        present (helpers named _*.sh excluded); args are plain tokens only."""
        if not self.cfg.lab_enabled:
            raise PermissionError("lab controls are disabled (CONSOLE_LAB=1 enables them)")
        faults = self.cfg.lab_dir / "faults"
        allowed = {p.stem for p in faults.glob("*.sh") if not p.name.startswith("_")} if faults.is_dir() else set()
        if name not in allowed:
            raise KeyError(f"unknown fault {name!r}; known: {', '.join(sorted(allowed))}")
        args = [str(a) for a in (args or [])]
        bad = [a for a in args if not self._ARG.match(a)]
        if bad:
            raise ValueError(f"invalid fault argument(s): {bad}")
        self.note(f"injecting lab fault {name} {' '.join(args)}".rstrip(), source="lab")
        p = subprocess.run(["bash", str(faults / f"{name}.sh"), *args], capture_output=True, text=True, timeout=240)
        for line in (p.stdout + p.stderr).splitlines()[-30:]:
            self.note(line, source=f"lab/{name}")
        return {"fault": name, "exit_code": p.returncode, "output": (p.stdout + p.stderr)[-4000:]}

    def _audit(self, action: str, **fields) -> None:
        if self.overlord is None or not hasattr(self.overlord, "audit"):
            return
        try:
            self.overlord.audit(action, **{k: v for k, v in fields.items() if v is not None})
        except Exception as e:                        # noqa: BLE001
            self.note(f"audit mirror failed: {e}", level="warn")
