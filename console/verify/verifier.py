"""The verification loop.

`evaluate()` is the pure decision — given the plan, the receipt and the
freshest SABLE tick, what is the verdict — so it is table-testable.
`Verifier` wraps it with the waiting, the consequences (close / compensate
+ reopen / escalate) and the receipt's finalisation.

Predicates (from the catalog):
  node_healthy      every node in the plan's blast radius reports `healthy`
  service_running   the target node(s) of the plan's steps report `healthy`
                    (the lab's `up == 1` collapses into SABLE's `healthy`)

Inconclusive, never a pass:
  - no tick newer than the last step's end (telemetry did not catch up)
  - the tick is older than `stale_after_s`
  - a node in scope is `unreachable` (absence of telemetry is not health)
  - a node in scope is missing from the tick
  - an affected (non-target) node is `oscillating`: the dependents are still
    settling after the fix. The verifier re-checks ONCE after `settle_s`,
    against a tick newer than the first look; still oscillating then is an
    escalation, not a fail — the fix itself landed, a human looks at the
    flapping dependents. An oscillating TARGET node is a fail like any other
    unhealthy state.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from console.contracts import (
    Finding, FindingStatus, Plan, Receipt, StepResult, VerificationResult, VerificationStatus, now_utc,
)


class Verdict:
    """What evaluate() decided and why — becomes the receipt's VerificationResult.
    `recheck` asks the Verifier for one more look after the settle; `tick_time`
    is the tick the verdict was read from (the re-check must see a newer one)."""

    def __init__(self, status: str, observed: dict[str, Any], reason: str, recheck: bool = False,
                 tick_time: Optional[datetime] = None):
        self.status, self.observed, self.reason = status, observed, reason
        self.recheck, self.tick_time = recheck, tick_time

    def to_result(self, checked_at: Optional[datetime] = None) -> VerificationResult:
        return VerificationResult(status=self.status, observed={**self.observed, "reason": self.reason},
                                  checked_at=checked_at or now_utc())


def _targets(plan: Plan) -> list[str]:
    return sorted({s.target_node for s in plan.steps})


def _scope(plan: Plan) -> list[str]:
    pred = plan.verification.predicate
    if pred == "service_running":
        return _targets(plan)
    return list(plan.blast_radius.nodes) or _targets(plan)


def node_states(tick: dict[str, Any], topology) -> dict[str, str]:
    """{node_id: state} from a SABLE tick, mapping the engine's index to the
    topology's id positionally (docker/server.py:86-106)."""
    out = {}
    for n in tick.get("nodes") or []:
        nid = None
        if topology is not None and "id" in n:
            try:
                nid = topology.id_at(int(n["id"]))
            except (TypeError, ValueError):
                nid = None
        nid = nid or n.get("node_id") or n.get("topo_id") or (f"node-{n.get('id')}" if "id" in n else None)
        if nid is not None:
            out[str(nid)] = str(n.get("state", "unknown"))
    return out


_node_states = node_states     # the older private name, kept for callers


def _freshness_bound(receipt: Receipt, after: Optional[datetime]) -> tuple[Optional[datetime], str]:
    """The instant a usable tick must be newer than, and what that instant is."""
    ended = max((s.ended_at for s in receipt.steps if s.ended_at), default=None)
    if after is not None and (ended is None or after >= ended):
        return after, "the re-check"
    return ended, "the last step"


def evaluate(plan: Plan, receipt: Receipt, tick: Optional[dict[str, Any]], tick_time: Optional[datetime],
             topology=None, stale_after_s: int = 120, now: Optional[datetime] = None,
             after: Optional[datetime] = None, allow_recheck: bool = True) -> Verdict:
    """The verdict for one plan, from the freshest tick SABLE has. `after`
    raises the freshness bound (the re-check wants a tick newer than the
    first look); `allow_recheck=False` is that second look."""
    now = now or now_utc()
    if tick is None or tick_time is None:
        return Verdict("inconclusive", {}, "no tick since execution")
    bound, what = _freshness_bound(receipt, after)
    if bound is not None and tick_time <= bound:
        return Verdict("inconclusive", {"tick_time": tick_time.isoformat()},
                       f"the newest tick predates {what}; telemetry has not caught up", tick_time=tick_time)
    if now - tick_time > timedelta(seconds=stale_after_s):
        return Verdict("inconclusive", {"tick_time": tick_time.isoformat()},
                       f"the newest tick is older than {stale_after_s}s", tick_time=tick_time)
    states = node_states(tick, topology)
    scope = _scope(plan)
    observed = {n: states.get(n, "missing") for n in scope}
    missing = [n for n, s in observed.items() if s == "missing"]
    unreachable = [n for n, s in observed.items() if s == "unreachable"]
    if missing:
        return Verdict("inconclusive", observed, f"no telemetry for {', '.join(missing)}", tick_time=tick_time)
    if unreachable:
        return Verdict("inconclusive", observed,
                       f"{', '.join(unreachable)} unreachable — absence of telemetry is not health", tick_time=tick_time)
    targets = set(_targets(plan))
    settling = [n for n, s in observed.items() if s == "oscillating" and n not in targets]
    bad = [n for n, s in observed.items() if s != "healthy" and n not in settling]
    if bad:
        detail = ", ".join(f"{n}={observed[n]}" for n in bad + settling)
        return Verdict("fail", observed, f"still not healthy: {detail}", tick_time=tick_time)
    if settling:
        detail = ", ".join(f"{n}=oscillating" for n in settling)
        if allow_recheck:
            return Verdict("inconclusive", observed, f"{detail} — dependents still settling; re-check pending",
                           recheck=True, tick_time=tick_time)
        return Verdict("inconclusive", observed, f"{detail} — still oscillating after the settle re-check",
                       tick_time=tick_time)
    return Verdict("pass", observed, f"{plan.verification.predicate}: all {len(scope)} node(s) healthy",
                   tick_time=tick_time)


class Verifier:
    """Waits the window, takes the freshest tick, decides, and applies the
    consequence. `latest_tick()` -> (tick_dict, tick_time) | (None, None)."""

    def __init__(self, latest_tick: Callable[[], tuple[Optional[dict], Optional[datetime]]],
                 store, executor, chain, topology=None, emit: Optional[Callable] = None,
                 stale_after_s: int = 120, poll_s: float = 2.0, max_wait_s: float = 90.0,
                 sleep: Callable[[float], Any] = None, settle_s: float = 15.0):
        self.latest_tick, self.store, self.executor, self.chain = latest_tick, store, executor, chain
        self.topology, self.emit = topology, (emit or (lambda ev: None))
        self.stale_after_s, self.poll_s, self.max_wait_s = stale_after_s, poll_s, max_wait_s
        self.settle_s = settle_s
        self._sleep = sleep or time.sleep

    # ---------------------------------------------------------------- sync

    def verify(self, plan: Plan, finding: Finding, receipt: Receipt, wait: bool = True) -> Receipt:
        if wait and plan.verification.window_s:
            self._sleep(plan.verification.window_s)
        verdict = self._decide(plan, receipt)
        if verdict.recheck:
            self._announce_settle(plan, verdict)
            self._sleep(self.settle_s)
            verdict = self._recheck(plan, receipt, verdict)
        return self.apply(plan, finding, receipt, verdict)

    def _decide(self, plan: Plan, receipt: Receipt, after: Optional[datetime] = None,
                allow_recheck: bool = True) -> Verdict:
        """Poll for a tick newer than the last step (or `after`), up to max_wait_s."""
        deadline = time.monotonic() + self.max_wait_s
        while True:
            tick, ts = self.latest_tick()
            verdict = evaluate(plan, receipt, tick, ts, self.topology, self.stale_after_s,
                               after=after, allow_recheck=allow_recheck)
            fresh_needed = verdict.status == "inconclusive" and (
                "no tick" in verdict.reason or "predates" in verdict.reason)
            if not fresh_needed or time.monotonic() >= deadline:
                return verdict
            self._sleep(self.poll_s)

    def _recheck(self, plan: Plan, receipt: Receipt, first: Verdict) -> Verdict:
        """The one re-check after the settle: a NEWER tick than the first
        look, and no further re-check. The receipt records both looks."""
        verdict = self._decide(plan, receipt, after=first.tick_time, allow_recheck=False)
        verdict.observed = {**verdict.observed, "recheck": {"after_s": self.settle_s, "first": first.reason}}
        return verdict

    def _announce_settle(self, plan: Plan, verdict: Verdict) -> None:
        self.emit({"type": "log", "line": {"ts": now_utc().isoformat(), "level": "info", "source": "verify",
                                            "text": f"plan {plan.id}: {verdict.reason}; re-checking in {self.settle_s:g}s"}})

    def apply(self, plan: Plan, finding: Finding, receipt: Receipt, verdict: Verdict) -> Receipt:
        """Write the verdict onto the receipt and carry out its consequence."""
        receipt.verification = verdict.to_result()
        self.emit({"type": "verification", "plan_id": plan.id, "receipt_id": receipt.id,
                   "result": receipt.verification.model_dump(mode="json")})
        if verdict.status == "pass":
            self.store.close(finding.id, receipt.id)
        elif verdict.status == "fail":
            completed = [(s, r) for r in receipt.steps if r.status == "ok"
                         for s in plan.steps if s.step_id == r.step_id]
            if completed and not receipt.rollback.performed:
                receipt.rollback = self.executor.compensate(plan, completed)
            self.store.reopen(finding.id, receipt.id)
        else:                                   # inconclusive: a human decides; never a pass
            self.store.attach(finding.id, receipt.id) if hasattr(self.store, "attach") else None
            self.emit({"type": "escalation", "plan_id": plan.id, "finding_id": finding.id,
                       "receipt_id": receipt.id, "decision": "human_plus", "reason": verdict.reason})
        return self.executor.finalize(receipt, self.chain)

    # ---------------------------------------------------------------- async

    async def verify_async(self, plan: Plan, finding: Finding, receipt: Receipt) -> Receipt:
        if plan.verification.window_s:
            await asyncio.sleep(plan.verification.window_s)
        loop = asyncio.get_running_loop()
        verdict = await loop.run_in_executor(None, self._decide, plan, receipt)
        if verdict.recheck:
            self._announce_settle(plan, verdict)
            await asyncio.sleep(self.settle_s)
            verdict = await loop.run_in_executor(None, self._recheck, plan, receipt, verdict)
        return self.apply(plan, finding, receipt, verdict)
