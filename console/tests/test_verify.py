"""Phase 5: pass closes the finding with the receipt attached; fail
compensates through OVERLORD and reopens; inconclusive escalates and never
counts as a pass; the receipt is finalised into the chain either way."""

from datetime import datetime, timedelta, timezone

import pytest

from console.contracts import (
    BlastRadius, Compensation, Confidence, Finding, NodeRef, Plan, Receipt, Step, StepResult,
    Verification,
)
from console.executor import ReceiptChain
from console.topology import Topology
from console.verify import Verifier, evaluate

T0 = datetime(2026, 9, 22, 14, 0, 0, tzinfo=timezone.utc)
NODES = [{"id": "dns", "type": "DNS_SERVER"}, {"id": "app", "type": "APPLICATION_SERVICE"},
         {"id": "db", "type": "SERVER_VIRTUAL"}]
TOPO = Topology(NODES, [{"source": "app", "target": "dns", "type": "DNS_DEPENDENCY", "criticality": "HARD"}])


def tick(states, source="live"):
    """A SABLE tick with engine node indices 0..n positionally = NODES."""
    return {"cycle": 7, "source": source,
            "nodes": [{"id": i, "state": states.get(n["id"], "healthy"), "confidence": 0.9}
                      for i, n in enumerate(NODES) if n["id"] in states or True]}


def plan(predicate="node_healthy"):
    return Plan(id="pln-1", finding_id="fnd-1", created_at=T0, steps=[
        Step(step_id="s1", action_id="restart_service", target_node="dns", params={"service": "dns"},
             reversibility="compensable", compensation=Compensation(action_id="restart_service", params={"service": "dns"}),
             precondition="none", timeout_s=30)],
        blast_radius=BlastRadius(nodes=["dns", "app"], count=2),
        verification=Verification(predicate=predicate, window_s=0))


def finding():
    return Finding(id="fnd-1", site_id="lab", detection_mode="live_feed",
                   root_cause=NodeRef(node_id="dns", component_type="DNS_SERVER", state="failed"),
                   confidence=Confidence(score=0.9, method="engine_native"), severity="high",
                   engine_version="sable@test")


def receipt(ended=T0):
    return Receipt(plan_id="pln-1", finding_id="fnd-1", session_id="sid-1",
                   steps=[StepResult(step_id="s1", status="ok", started_at=T0, ended_at=ended, session_id="sid-1")])


LATER = T0 + timedelta(seconds=10)
NOW = T0 + timedelta(seconds=20)


# ---------------------------------------------------------------- evaluate (pure)

def test_pass_when_every_node_in_blast_radius_is_healthy():
    v = evaluate(plan(), receipt(), tick({}), LATER, TOPO, now=NOW)
    assert v.status == "pass" and v.observed == {"dns": "healthy", "app": "healthy"}


def test_fail_when_a_node_in_scope_is_not_healthy():
    v = evaluate(plan(), receipt(), tick({"app": "degraded"}), LATER, TOPO, now=NOW)
    assert v.status == "fail" and "app=degraded" in v.reason


def test_service_running_predicate_scopes_to_step_targets():
    v = evaluate(plan("service_running"), receipt(), tick({"app": "degraded"}), LATER, TOPO, now=NOW)
    assert v.status == "pass" and v.observed == {"dns": "healthy"}


@pytest.mark.parametrize("case, tk, ts, needle, now", [
    ("no tick", None, None, "no tick", NOW),
    ("tick predates last step", tick({}), T0 - timedelta(seconds=1), "predates", NOW),
    ("stale tick", tick({}), LATER, "older than", LATER + timedelta(seconds=999)),   # after the step, but long ago
    ("unreachable node", tick({"dns": "unreachable"}), LATER, "unreachable", NOW),
    ("node missing from tick", {"cycle": 1, "nodes": [{"id": 0, "state": "healthy"}]}, LATER, "no telemetry for app", NOW),
])
def test_inconclusive_never_counts_as_pass(case, tk, ts, needle, now):
    v = evaluate(plan(), receipt(), tk, ts, TOPO, now=now)
    assert v.status == "inconclusive", case
    assert needle in v.reason


# ---------------------------------------------------------------- consequences

class Store:
    def __init__(self):
        self.closed, self.reopened, self.attached = [], [], []

    def close(self, fid, rid):
        self.closed.append((fid, rid))

    def reopen(self, fid, rid):
        self.reopened.append((fid, rid))

    def attach(self, fid, rid):
        self.attached.append((fid, rid))


class Exec:
    def __init__(self):
        self.compensated, self.finalized = [], []

    def compensate(self, plan, completed):
        self.compensated.append([s.step_id for s, _ in completed])
        from console.contracts import Rollback, RollbackStep
        return Rollback(performed=True, steps=[RollbackStep(step_id=s.step_id, action_id="restart_service", status="ok")
                                                for s, _ in reversed(completed)])

    def finalize(self, r, chain):
        chain.append(r)
        self.finalized.append(r.id)
        return r


def make(tick_fn, tmp_path):
    store, ex, chain, events = Store(), Exec(), ReceiptChain(tmp_path / "r.jsonl"), []
    v = Verifier(tick_fn, store, ex, chain, TOPO, emit=events.append, sleep=lambda s: None, max_wait_s=0)
    return v, store, ex, chain, events


def test_pass_closes_the_finding_with_the_receipt_attached(tmp_path):
    v, store, ex, chain, events = make(lambda: (tick({}), datetime.now(timezone.utc)), tmp_path)
    r = v.verify(plan(), finding(), receipt())
    assert r.verification.status == "pass"
    assert store.closed == [("fnd-1", r.id)] and not store.reopened
    assert ex.compensated == [] and ex.finalized == [r.id]
    assert chain.verify()["ok"] and chain.head == r.receipt_hash
    assert {e["type"] for e in events} == {"verification"}


def test_fail_compensates_completed_steps_and_reopens(tmp_path):
    v, store, ex, chain, _ = make(lambda: (tick({"app": "failed"}), datetime.now(timezone.utc)), tmp_path)
    r = v.verify(plan(), finding(), receipt())
    assert r.verification.status == "fail"
    assert ex.compensated == [["s1"]] and r.rollback.performed and r.rollback.steps[0].step_id == "s1"
    assert store.reopened == [("fnd-1", r.id)] and not store.closed
    assert ex.finalized == [r.id]


def test_fail_does_not_compensate_twice_when_execution_already_rolled_back(tmp_path):
    v, store, ex, chain, _ = make(lambda: (tick({"app": "failed"}), datetime.now(timezone.utc)), tmp_path)
    from console.contracts import Rollback
    rc = receipt(); rc.rollback = Rollback(performed=True)
    v.verify(plan(), finding(), rc)
    assert ex.compensated == [] and store.reopened


def test_inconclusive_escalates_and_is_never_a_pass(tmp_path):
    v, store, ex, chain, events = make(lambda: (None, None), tmp_path)
    r = v.verify(plan(), finding(), receipt())
    assert r.verification.status == "inconclusive"
    assert not store.closed and not store.reopened and store.attached == [("fnd-1", r.id)]
    esc = [e for e in events if e["type"] == "escalation"]
    assert esc and esc[0]["decision"] == "human_plus"
    assert ex.finalized == [r.id]


def test_waits_for_a_tick_newer_than_the_last_step(tmp_path):
    ended = datetime.now(timezone.utc)
    ticks = iter([(tick({}), ended - timedelta(seconds=5)),          # stale: predates the step
                  (tick({}), ended + timedelta(seconds=5))])         # fresh
    store, ex, chain = Store(), Exec(), ReceiptChain(tmp_path / "r.jsonl")
    slept = []
    v = Verifier(lambda: next(ticks), store, ex, chain, TOPO, sleep=slept.append, poll_s=1, max_wait_s=30)
    r = v.verify(plan(), finding(), receipt(ended=ended))
    assert r.verification.status == "pass" and slept == [1]


def test_window_is_honoured(tmp_path):
    store, ex, chain, slept = Store(), Exec(), ReceiptChain(tmp_path / "r.jsonl"), []
    v = Verifier(lambda: (tick({}), datetime.now(timezone.utc)), store, ex, chain, TOPO, sleep=slept.append)
    p = plan().model_copy(update={"verification": Verification(predicate="node_healthy", window_s=30)})
    v.verify(p, finding(), receipt())
    assert slept[0] == 30
