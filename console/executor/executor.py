"""Execute a Plan through OVERLORD and emit a Receipt.

Per step, in order:
  1. approval gate for irreversible steps — refused without a recorded
     approve/execute_now decision (none in the MVP catalog, enforced anyway)
  2. open an OVERLORD session with the MINIMUM scope the catalog binding
     declares (jail / net / timeout / net_allow / limits), over the lab dir
  3. precondition — the named check's argv, inside the session; a miss
     rolls the session back and stops the plan (nothing was changed)
  4. the action's argv (then its optional follow-up), with
     cause={plan_id, step_id, action_id, finding_id} so every path the step
     touches names this step on OVERLORD's provenance; per-step timeout
  5. close → commit. The session IS the snapshot: its retained
     before-content is what `overlord revert` restores.

On a failed step: that session is rolled back (its own changes never
landed), then the COMPLETED steps are compensated in reverse order —
`overlord revert` for a reversible step (its committed file changes are
undone as a new committed session), the catalog's compensation action for a
compensable one. The receipt records all of it; `rollback.performed` is
true. Verification (Phase 5) is written onto the receipt afterwards; the
receipt is sealed onto the chain once, in its final form.

The OVERLORD client is duck-typed (sdk/overlord_client.py's OverlordClient):
  open(target, jail, net, timeout, net_allow, limits) -> live with
      exec(cmd, timeout, label, cause) -> (rc, output, changes); close() -> session (.sid, commit())
  revert(sid, commit) ; audit(action, **fields) ; audit_head()
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Callable, Optional

from console.contracts import (
    Approval, Finding, Plan, Receipt, Rollback, RollbackStep, Snapshot, Step, StepResult, now_utc,
)

_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class ExecutionError(RuntimeError):
    pass


def render_argv(argv: list[str], values: dict[str, Any]) -> list[str]:
    """Fill `{name}` placeholders from values; an unknown one is an error,
    never silently left in a command line."""
    out = []
    for token in argv:
        def sub(m):
            k = m.group(1)
            if k not in values:
                raise ExecutionError(f"unbound placeholder {{{k}}} in argv")
            return str(values[k])
        out.append(_PLACEHOLDER.sub(sub, token))
    return out


def _digest(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode(errors="replace")
    return hashlib.sha256(data).hexdigest()


class Executor:
    def __init__(self, client, catalog, context: dict[str, Any], emit: Optional[Callable] = None,
                 audit: bool = True):
        self.client = client
        self.catalog = catalog
        self.context = dict(context)          # lab_dir, lab_compose, ...
        self.emit = emit or (lambda ev: None)
        self.audit = audit

    # ---------------------------------------------------------------- catalog access (duck-typed)

    def _binding(self, action_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """{argv, then, grants, target_dir} for an action, rendered."""
        b = self.catalog.render(action_id, params, self.context)
        if hasattr(b, "model_dump"):
            b = b.model_dump()
        return b

    def _check(self, check_id: str, action_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """{argv, expect} for a precondition check, rendered."""
        if hasattr(self.catalog, "render_check"):
            c = self.catalog.render_check(check_id, action_id, params, self.context)
            return c.model_dump() if hasattr(c, "model_dump") else c
        checks = getattr(self.catalog, "checks", {}) or {}
        spec = checks.get(check_id) if isinstance(checks, dict) else None
        if spec is None:
            raise ExecutionError(f"unknown precondition check: {check_id}")
        spec = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
        values = {**self.context, **params}
        return {"argv": render_argv(list(spec.get("argv") or ["true"]), values),
                "expect": {k: (render_argv([v], values)[0] if isinstance(v, str) else v)
                           for k, v in (spec.get("expect") or {"exit_code": 0}).items()}}

    def _reversibility(self, step: Step) -> str:
        return str(step.reversibility)

    # ---------------------------------------------------------------- one session

    def _open(self, grants: dict[str, Any], target_dir: str):
        g = dict(grants or {})
        timeout = g.pop("timeout_s", None) or g.pop("timeout", None)
        return self.client.open(target_dir, jail=bool(g.get("jail", True)),
                                net=g.get("net", "none"), timeout=timeout,
                                net_allow=g.get("net_allow"), limits=g.get("limits"))

    def _run(self, live, argv: list[str], timeout: Optional[int], label: str, cause: dict) -> tuple[int, str]:
        self._log(f"$ {' '.join(argv)}", label)
        rc, out, _changes = live.exec(argv, timeout=timeout, label=label, cause=cause)
        if out:
            for line in str(out).splitlines()[-20:]:
                self._log(line, label)
        return rc, str(out)

    def _log(self, text: str, source: str = "executor"):
        self.emit({"type": "log", "line": {"ts": now_utc().isoformat(), "level": "info",
                                            "source": source, "text": text}})

    @staticmethod
    def _expect_ok(rc: int, out: str, expect: dict[str, Any]) -> bool:
        if "exit_code" in expect and rc != int(expect["exit_code"]):
            return False
        if "stdout_contains" in expect and str(expect["stdout_contains"]) not in out:
            return False
        return True

    # ---------------------------------------------------------------- the plan

    def execute(self, plan: Plan, finding: Finding, approvals: list[Approval]) -> Receipt:
        approved = any(a.decision in ("approve", "execute_now") for a in approvals)
        receipt = Receipt(plan_id=plan.id, finding_id=finding.id, approvals=list(approvals))
        completed: list[tuple[Step, StepResult]] = []
        failed_at: Optional[str] = None

        for step in plan.steps:
            res = StepResult(step_id=step.step_id, status="pending")
            cause = {"plan_id": plan.id, "step_id": step.step_id, "action_id": step.action_id,
                     "finding_id": finding.id}
            # 1. an irreversible step never runs without a recorded approval
            if self._reversibility(step) == "irreversible" and not approved:
                res.status, res.error = "skipped", "irreversible step needs a recorded approval"
                receipt.steps.append(res)
                self._step_event(plan, res)
                failed_at = step.step_id
                break
            res.status, res.started_at = "running", now_utc()
            self._step_event(plan, res)
            try:
                binding = self._binding(step.action_id, step.params)
            except Exception as e:                    # noqa: BLE001 — a bad binding is a failed step
                res.status, res.error, res.ended_at = "failed", f"binding: {e}", now_utc()
                receipt.steps.append(res)
                self._step_event(plan, res)
                failed_at = step.step_id
                break
            values = {**self.context, **step.params, "target_node": step.target_node}
            # 2. minimum scope
            live = self._open(binding.get("grants") or {}, binding.get("target_dir") or self.context.get("lab_dir", "."))
            sid = getattr(live, "sid", None)
            res.session_id = sid
            try:
                # 3. precondition
                check = self._check(step.precondition, step.action_id, step.params)
                rc, out = self._run(live, check["argv"], 30, f"{step.step_id}/pre", {**cause, "phase": "precondition"})
                if not self._expect_ok(rc, out, check.get("expect") or {"exit_code": 0}):
                    live.close().rollback()          # nothing was changed; discard the session
                    res.status, res.error = "precondition_failed", f"{step.precondition}: exit {rc}"
                    res.ended_at, res.output_digest = now_utc(), _digest(out)
                    receipt.steps.append(res)
                    self._step_event(plan, res)
                    failed_at = step.step_id
                    break
                # 4. the action, then its follow-up
                argv = list(binding["argv"])
                rc, out = self._run(live, argv, step.timeout_s, step.step_id, cause)
                if rc == 0 and binding.get("then"):
                    rc2, out2 = self._run(live, list(binding["then"]),
                                          step.timeout_s, f"{step.step_id}/then", {**cause, "phase": "then"})
                    rc, out = rc2, out + out2
                session = live.close()
                res.output_digest, res.ended_at = _digest(out), now_utc()
                timed_out = rc == 124 or getattr(live, "expired", False)
                if rc != 0:
                    session.rollback()
                    res.status = "timed_out" if timed_out else "failed"
                    res.error = f"exit {rc}"
                    receipt.steps.append(res)
                    self._step_event(plan, res)
                    failed_at = step.step_id
                    break
                # 5. commit — the session is the snapshot
                session.commit()
                res.status = "ok"
                receipt.steps.append(res)
                receipt.snapshots.append(Snapshot(node_id=step.target_node, snapshot_ref=str(sid)))
                if receipt.session_id is None:
                    receipt.session_id = str(sid)
                completed.append((step, res))
                self._step_event(plan, res)
                self._audit("receipt.step", plan_id=plan.id, step_id=step.step_id,
                            action_id=step.action_id, sid=sid, status="ok")
            except Exception as e:                    # noqa: BLE001 — any failure is a failed step, never a crash
                try:
                    live.close().rollback()
                except Exception:                     # noqa: BLE001
                    pass
                res.status, res.error, res.ended_at = "failed", str(e), now_utc()
                if res not in receipt.steps:
                    receipt.steps.append(res)
                self._step_event(plan, res)
                failed_at = step.step_id
                break

        if failed_at is not None:
            self._audit("receipt.step", plan_id=plan.id, step_id=failed_at, status="failed")
            if completed:
                receipt.rollback = self.compensate(plan, completed)
        return receipt

    # ---------------------------------------------------------------- compensation

    def compensate(self, plan: Plan, completed: list[tuple[Step, StepResult]]) -> Rollback:
        """Undo completed steps in REVERSE order: revert a reversible step's
        committed session; run a compensable step's compensation action."""
        rb = Rollback(performed=True)
        for step, res in reversed(completed):
            kind = self._reversibility(step)
            entry = RollbackStep(step_id=step.step_id, action_id=step.action_id, status="running")
            try:
                if kind == "reversible" or step.compensation is None:
                    r = self.client.revert(res.session_id, commit=True, force=True)
                    entry.session_id = r.get("sid") if isinstance(r, dict) else None
                    entry.status = "ok" if (not isinstance(r, dict) or r.get("committed", True)) else "failed"
                    entry.action_id = "overlord.revert"
                else:
                    comp = step.compensation
                    entry.action_id = comp.action_id
                    binding = self._binding(comp.action_id, comp.params)
                    values = {**self.context, **comp.params, "target_node": step.target_node}
                    live = self._open(binding.get("grants") or {}, binding.get("target_dir") or self.context.get("lab_dir", "."))
                    entry.session_id = getattr(live, "sid", None)
                    cause = {"plan_id": plan.id, "step_id": step.step_id, "action_id": comp.action_id,
                             "compensates": step.action_id}
                    rc, out = self._run(live, list(binding["argv"]), step.timeout_s,
                                        f"{step.step_id}/compensate", cause)
                    session = live.close()
                    if rc == 0:
                        session.commit()
                        entry.status = "ok"
                    else:
                        session.rollback()
                        entry.status, entry.error = "failed", f"exit {rc}"
            except Exception as e:                    # noqa: BLE001 — record, continue compensating the rest
                entry.status, entry.error = "failed", str(e)
            rb.steps.append(entry)
            self._audit("receipt.rollback", plan_id=plan.id, step_id=step.step_id,
                        action_id=entry.action_id, status=entry.status)
            self.emit({"type": "rollback", "plan_id": plan.id, "step": entry.model_dump(mode="json")})
        return rb

    # ---------------------------------------------------------------- receipts

    def finalize(self, receipt: Receipt, chain) -> Receipt:
        """Seal against the chain head, mirror the closing entry onto
        OVERLORD's audit chain (which commits to the final receipt_hash),
        record {seq, hash} as audit_ref — outside the hash — then persist."""
        receipt.seal(prev_receipt_hash=chain.head)
        ref = self._audit("receipt.close", receipt_id=receipt.id, plan_id=receipt.plan_id,
                          finding_id=receipt.finding_id, receipt_hash=receipt.receipt_hash,
                          verification=(receipt.verification.status if receipt.verification else None),
                          rollback=receipt.rollback.performed)
        if isinstance(ref, dict) and ref.get("seq"):
            receipt.audit_ref = {"seq": ref.get("seq"), "hash": ref.get("hash")}
        chain.append(receipt)
        self.emit({"type": "receipt", "receipt": receipt.model_dump(mode="json")})
        return receipt

    # ---------------------------------------------------------------- helpers

    def _step_event(self, plan: Plan, res: StepResult):
        self.emit({"type": "step", "plan_id": plan.id, "step": res.model_dump(mode="json")})

    def _audit(self, action: str, **fields) -> Optional[dict]:
        if not self.audit or not hasattr(self.client, "audit"):
            return None
        try:
            return self.client.audit(action, **{k: v for k, v in fields.items() if v is not None})
        except Exception as e:                        # noqa: BLE001 — the audit mirror never fails a run
            self._log(f"audit mirror failed: {e}")
            return None
