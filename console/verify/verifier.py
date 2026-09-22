"""The verification loop.

`evaluate()` is the pure decision — given the plan, the receipt and the
freshest SABLE tick, what is the verdict — so it is table-testable.
`Verifier` wraps it with the waiting, the consequences (close / compensate
+ reopen / escalate) and the receipt's finalisation.

Predicates (from the catalog):
  node_healthy      every node in the plan's blast radius reports `healthy`
  service_running   the target node(s) of the plan's steps report `healthy`
                    (the lab's `up == 1` collapses into SABLE's `healthy`)

Two readings per node: the MODEL's state (SABLE, the source of truth — a
pass needs it healthy across the scope) and the health scorer's ground
truth from raw telemetry when the tick carries it (ground_truth.py).

  pass          model healthy everywhere in scope
  fail          model non-healthy AND telemetry non-healthy (or unknown) —
                the fix did not land; also the model reading `failed` on a
                TARGET whatever telemetry says, and `unreachable` anywhere
                when telemetry agrees
  inconclusive, terminal (escalate now, never a pass):
                no tick newer than the last step, a stale tick, a node
                missing from the tick, a node `unreachable` without
                telemetry saying otherwise (absence of telemetry is not
                health)
  inconclusive, settling (re-check):
                model non-healthy while telemetry is healthy — a
                DISAGREEMENT, not a failure: the trained model lags real
                recovery by minutes (live evidence: `app` oscillating/
                degraded for ~150s after a fix while Prometheus read
                healthy throughout); also `oscillating` on a non-target
                node without telemetry. The Verifier re-checks every
                `settle_s` against a NEWER tick up to `max_settle_s`; the
                first look with the model healthy is a pass, a fail in
                between is a fail, the deadline is inconclusive. Nothing
                is compensated on a disagreement.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from console.contracts import (
    Finding, FindingStatus, Plan, Receipt, StepResult, VerificationResult, VerificationStatus, now_utc,
)
from console.sable_bridge.ground_truth import ground_truth_states


class Verdict:
    """What evaluate() decided and why — becomes the receipt's VerificationResult.
    `recheck` asks the Verifier for another look after the settle; `tick_time`
    is the tick the verdict was read from (the next look must see a newer one)."""

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
    """{node_id: model state} from a SABLE tick, mapping the engine's index to
    the topology's id positionally (docker/server.py:86-106)."""
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


def _sort_nodes(observed: dict[str, str], truth: dict[str, str], targets: set[str]) -> dict[str, list[str]]:
    """Each non-healthy node in scope into one bucket:
    fails (the fix did not land), terminal (escalate now), disagree (model
    unhealthy, telemetry healthy or itself still oscillating), oscillating
    (non-target, telemetry unknown)."""
    buckets: dict[str, list[str]] = {"fail": [], "terminal": [], "disagree": [], "oscillating": []}
    for node, state in observed.items():
        if state == "healthy":
            continue
        telemetry = truth.get(node)                       # None: unknown
        # The scorer calls a node oscillating after 3+ state changes in 5 min —
        # which is exactly what a fault and its fix look like from telemetry.
        # It is not evidence against the fix; it is telemetry still settling.
        settling = telemetry in ("healthy", "oscillating")
        if state == "unreachable":
            buckets["fail" if not (telemetry is None or settling) else "terminal"].append(node)
        elif node in targets and state == "failed":
            buckets["fail"].append(node)
        elif settling:
            buckets["disagree"].append(node)
        elif state == "oscillating" and telemetry is None and node not in targets:
            buckets["oscillating"].append(node)
        else:
            buckets["fail"].append(node)
    return buckets


def evaluate(plan: Plan, receipt: Receipt, tick: Optional[dict[str, Any]], tick_time: Optional[datetime],
             topology=None, stale_after_s: int = 120, now: Optional[datetime] = None,
             after: Optional[datetime] = None) -> Verdict:
    """The verdict for one plan, from the freshest tick SABLE has. `after`
    raises the freshness bound (a re-check wants a tick newer than the last look)."""
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
    states, truth = node_states(tick, topology), ground_truth_states(tick, topology)
    scope = _scope(plan)
    observed: dict[str, Any] = {n: states.get(n, "missing") for n in scope}
    telemetry = {n: truth[n] for n in scope if n in truth}
    if telemetry:
        observed["telemetry"] = telemetry
    missing = [n for n in scope if observed[n] == "missing"]
    if missing:
        return Verdict("inconclusive", observed, f"no telemetry for {', '.join(missing)}", tick_time=tick_time)
    b = _sort_nodes({n: observed[n] for n in scope}, truth, set(_targets(plan)))
    if b["fail"]:
        detail = ", ".join(f"{n}={observed[n]}" + (f"/telemetry={truth[n]}" if n in truth else "")
                           for n in b["fail"] + b["disagree"] + b["oscillating"])
        return Verdict("fail", observed, f"still not healthy: {detail}", tick_time=tick_time)
    if b["terminal"]:
        return Verdict("inconclusive", observed,
                       f"{', '.join(b['terminal'])} unreachable — absence of telemetry is not health", tick_time=tick_time)
    if b["disagree"] or b["oscillating"]:
        parts = [f"{n} model={observed[n]} telemetry={truth.get(n, 'healthy')}" for n in b["disagree"]]
        parts += [f"{n}=oscillating" for n in b["oscillating"]]
        why = ("model/telemetry disagreement, the model may still be settling" if b["disagree"]
               else "dependents still settling")
        return Verdict("inconclusive", observed, f"{', '.join(parts)} — {why}; re-check pending",
                       recheck=True, tick_time=tick_time)
    return Verdict("pass", observed, f"{plan.verification.predicate}: all {len(scope)} node(s) healthy",
                   tick_time=tick_time)


class Verifier:
    """Waits the window, takes the freshest tick, decides, and applies the
    consequence. `latest_tick()` -> (tick_dict, tick_time) | (None, None)."""

    def __init__(self, latest_tick: Callable[[], tuple[Optional[dict], Optional[datetime]]],
                 store, executor, chain, topology=None, emit: Optional[Callable] = None,
                 stale_after_s: int = 120, poll_s: float = 2.0, max_wait_s: float = 90.0,
                 sleep: Callable[[float], Any] = None, settle_s: float = 15.0, max_settle_s: float = 180.0):
        self.latest_tick, self.store, self.executor, self.chain = latest_tick, store, executor, chain
        self.topology, self.emit = topology, (emit or (lambda ev: None))
        self.stale_after_s, self.poll_s, self.max_wait_s = stale_after_s, poll_s, max_wait_s
        self.settle_s, self.max_settle_s = settle_s, max_settle_s
        self._sleep = sleep or time.sleep

    # ---------------------------------------------------------------- sync

    def verify(self, plan: Plan, finding: Finding, receipt: Receipt, wait: bool = True) -> Receipt:
        if wait and plan.verification.window_s:
            self._sleep(plan.verification.window_s)
        verdict = self._decide(plan, receipt)
        if verdict.recheck:
            first, looks, waited, last = verdict, 0, 0.0, verdict
            while self._settle_more(last, waited):
                self._announce_settle(plan, last, waited)
                self._sleep(self.settle_s)
                waited, looks = waited + self.settle_s, looks + 1
                last = self._decide(plan, receipt, after=last.tick_time)
            verdict = self._with_recheck(first, last, looks, waited)
        return self.apply(plan, finding, receipt, verdict)

    def _decide(self, plan: Plan, receipt: Receipt, after: Optional[datetime] = None) -> Verdict:
        """Poll for a tick newer than the last step (or `after`), up to max_wait_s."""
        deadline = time.monotonic() + self.max_wait_s
        while True:
            tick, ts = self.latest_tick()
            verdict = evaluate(plan, receipt, tick, ts, self.topology, self.stale_after_s, after=after)
            fresh_needed = verdict.status == "inconclusive" and (
                "no tick" in verdict.reason or "predates" in verdict.reason)
            if not fresh_needed or time.monotonic() >= deadline:
                return verdict
            self._sleep(self.poll_s)

    def _settle_more(self, last: Verdict, waited: float) -> bool:
        return last.recheck and waited + self.settle_s <= self.max_settle_s

    def _with_recheck(self, first: Verdict, last: Verdict, looks: int, waited: float) -> Verdict:
        """The final verdict of a settle: the last look, or the deadline
        (inconclusive, never a fail); `observed.recheck` records the looks."""
        if last.recheck:
            base = last.reason.replace("; re-check pending", "")
            last = Verdict("inconclusive", last.observed,
                           f"{base}; still so after {waited:g}s of settle ({looks} re-check(s))",
                           tick_time=last.tick_time)
        readings = {k: v for k, v in last.observed.items() if k != "recheck"}
        last.observed = {**last.observed, "recheck": {"count": looks, "seconds": waited, "first": first.reason,
                                                      "last": readings}}
        return last

    def _announce_settle(self, plan: Plan, verdict: Verdict, waited: float) -> None:
        text = (f"plan {plan.id}: {verdict.reason}; re-checking in {self.settle_s:g}s "
                f"({waited:g}/{self.max_settle_s:g}s of settle used)")
        self.emit({"type": "log", "line": {"ts": now_utc().isoformat(), "level": "info", "source": "verify",
                                            "text": text}})

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
            first, looks, waited, last = verdict, 0, 0.0, verdict
            while self._settle_more(last, waited):
                self._announce_settle(plan, last, waited)
                await asyncio.sleep(self.settle_s)
                waited, looks = waited + self.settle_s, looks + 1
                last = await loop.run_in_executor(None, self._decide, plan, receipt, last.tick_time)
            verdict = self._with_recheck(first, last, looks, waited)
        return self.apply(plan, finding, receipt, verdict)
