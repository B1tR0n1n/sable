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


STATES = ["healthy", "degraded", "failed", "unreachable", "oscillating"]


def tick(states, source="live", truth=None, cycle=7):
    """A SABLE tick with engine node indices 0..n positionally = NODES. `truth`
    (a {node: state} dict, healthy by default) adds the scorer's `ground_truth`
    state indices the way the stub / the console's fetch attach them; None
    leaves the tick without telemetry, like a replay tick."""
    t = {"cycle": cycle, "source": source,
         "nodes": [{"id": i, "state": states.get(n["id"], "healthy"), "confidence": 0.9}
                   for i, n in enumerate(NODES)]}
    if truth is not None:
        t["ground_truth"] = [STATES.index(truth.get(n["id"], "healthy")) for n in NODES]
    return t


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


def test_oscillating_dependent_is_inconclusive_and_asks_for_one_recheck():
    v = evaluate(plan(), receipt(), tick({"app": "oscillating"}), LATER, TOPO, now=NOW)
    assert v.status == "inconclusive" and v.recheck is True
    assert "app=oscillating" in v.reason and "settling" in v.reason
    assert v.to_result().status == "inconclusive"                    # the contract shape is unchanged


# ---------------------------------------------------------------- model vs telemetry

def test_pass_carries_the_telemetry_reading_when_known():
    v = evaluate(plan(), receipt(), tick({}, truth={}), LATER, TOPO, now=NOW)
    assert v.status == "pass" and v.observed["telemetry"] == {"dns": "healthy", "app": "healthy"}


def test_model_unhealthy_but_telemetry_healthy_is_a_disagreement_not_a_fail():
    v = evaluate(plan(), receipt(), tick({"app": "degraded"}, truth={}), LATER, TOPO, now=NOW)
    assert v.status == "inconclusive" and v.recheck is True
    assert "app" in v.reason and "model=degraded" in v.reason and "telemetry=healthy" in v.reason
    assert "disagree" in v.reason
    assert v.observed["app"] == "degraded" and v.observed["telemetry"]["app"] == "healthy"


def test_model_and_telemetry_both_unhealthy_is_a_fail():
    v = evaluate(plan(), receipt(), tick({"app": "degraded"}, truth={"app": "degraded"}), LATER, TOPO, now=NOW)
    assert v.status == "fail" and "app=degraded" in v.reason and v.observed["telemetry"]["app"] == "degraded"
    v = evaluate(plan(), receipt(), tick({"app": "failed"}, truth={"app": "unreachable"}), LATER, TOPO, now=NOW)
    assert v.status == "fail"


def test_unknown_telemetry_is_no_evidence_against_the_model():
    """A tick without ground truth (a replay, an engine without the scorer): the
    model's non-healthy reading stands, as before."""
    v = evaluate(plan(), receipt(), tick({"app": "degraded"}), LATER, TOPO, now=NOW)
    assert v.status == "fail" and "telemetry" not in v.observed


def test_target_degraded_while_telemetry_healthy_is_a_disagreement_too():
    v = evaluate(plan(), receipt(), tick({"dns": "degraded"}, truth={}), LATER, TOPO, now=NOW)
    assert v.status == "inconclusive" and v.recheck is True and "dns" in v.reason


def test_target_failed_by_the_model_is_a_fail_whatever_telemetry_says():
    v = evaluate(plan(), receipt(), tick({"dns": "failed"}, truth={}), LATER, TOPO, now=NOW)
    assert v.status == "fail" and v.recheck is False and "dns=failed" in v.reason


def test_target_unreachable_is_a_fail_when_telemetry_agrees_and_inconclusive_without_it():
    v = evaluate(plan(), receipt(), tick({"dns": "unreachable"}, truth={"dns": "unreachable"}), LATER, TOPO, now=NOW)
    assert v.status == "fail"
    v = evaluate(plan(), receipt(), tick({"dns": "unreachable"}), LATER, TOPO, now=NOW)
    assert v.status == "inconclusive" and v.recheck is False and "unreachable" in v.reason


def test_a_terminal_inconclusive_beats_a_recheck():
    """dns (target) unreachable without telemetry escalates at once even though
    app is merely disagreeing: absent telemetry is not something to wait out."""
    t = tick({"dns": "unreachable", "app": "degraded"}, truth={"app": "healthy"})
    t["ground_truth"][0] = None                       # dns: no telemetry at all
    v = evaluate(plan(), receipt(), t, LATER, TOPO, now=NOW)
    assert v.status == "inconclusive" and v.recheck is False


def test_oscillating_target_node_is_a_fail():
    v = evaluate(plan(), receipt(), tick({"dns": "oscillating"}), LATER, TOPO, now=NOW)
    assert v.status == "fail" and v.recheck is False and "dns=oscillating" in v.reason


def test_oscillating_dependent_beside_an_unhealthy_node_is_a_fail():
    v = evaluate(plan(), receipt(), tick({"app": "oscillating", "dns": "degraded"}), LATER, TOPO, now=NOW)
    assert v.status == "fail" and v.recheck is False


def test_after_bound_demands_a_tick_newer_than_the_first_look():
    v = evaluate(plan(), receipt(), tick({}), LATER, TOPO, now=NOW, after=LATER)
    assert v.status == "inconclusive" and "predates" in v.reason
    v = evaluate(plan(), receipt(), tick({}), LATER + timedelta(seconds=1), TOPO, now=NOW, after=LATER)
    assert v.status == "pass"


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


# ---------------------------------------------------------------- the oscillating re-check

def _settle_ticks(first_state, second_state, poll_stale_once=False):
    t1 = datetime.now(timezone.utc)
    t2 = t1 + timedelta(seconds=20)
    seq = [(tick({"app": first_state}), t1)]
    if poll_stale_once:
        seq.append((tick({"app": first_state}), t1))                 # the same tick again: not newer
    seq.append((tick({"app": second_state}), t2))
    return iter(seq)


def test_oscillating_dependent_is_rechecked_once_after_the_settle_and_passes(tmp_path):
    ticks = _settle_ticks("oscillating", "healthy", poll_stale_once=True)
    store, ex, chain, slept, events = Store(), Exec(), ReceiptChain(tmp_path / "r.jsonl"), [], []
    v = Verifier(lambda: next(ticks), store, ex, chain, TOPO, emit=events.append, sleep=slept.append,
                 poll_s=1, max_wait_s=30, settle_s=15)
    r = v.verify(plan(), finding(), receipt())
    assert r.verification.status == "pass"
    assert slept == [15, 1]                                           # the settle, then one poll for a NEWER tick
    assert store.closed == [("fnd-1", r.id)] and not store.reopened and ex.compensated == []
    rc = r.verification.observed["recheck"]
    assert rc["count"] == 1 and rc["seconds"] == 15 and "oscillating" in rc["first"]
    assert rc["last"] == {"dns": "healthy", "app": "healthy"}
    assert [e["type"] for e in events if e["type"] == "log"]          # the settle is announced in the session log


def test_still_oscillating_after_the_recheck_escalates_and_does_not_compensate(tmp_path):
    ticks = _settle_ticks("oscillating", "oscillating")
    store, ex, chain, slept, events = Store(), Exec(), ReceiptChain(tmp_path / "r.jsonl"), [], []
    v = Verifier(lambda: next(ticks), store, ex, chain, TOPO, emit=events.append, sleep=slept.append,
                 poll_s=1, max_wait_s=30, settle_s=15, max_settle_s=15)
    r = v.verify(plan(), finding(), receipt())
    assert r.verification.status == "inconclusive"
    assert "oscillating" in r.verification.observed["reason"] and "after 15s" in r.verification.observed["reason"]
    assert slept == [15] and r.verification.observed["recheck"]["count"] == 1
    assert not store.closed and not store.reopened and ex.compensated == []
    assert [e for e in events if e["type"] == "escalation"][0]["decision"] == "human_plus"


def test_a_failed_recheck_compensates_and_reopens(tmp_path):
    ticks = _settle_ticks("oscillating", "failed")
    store, ex, chain, slept, _ = Store(), Exec(), ReceiptChain(tmp_path / "r.jsonl"), [], []
    v = Verifier(lambda: next(ticks), store, ex, chain, TOPO, sleep=slept.append, poll_s=1, max_wait_s=30)
    r = v.verify(plan(), finding(), receipt())
    assert r.verification.status == "fail" and slept == [15.0]
    assert ex.compensated == [["s1"]] and store.reopened == [("fnd-1", r.id)]


async def test_async_path_rechecks_too(tmp_path, monkeypatch):
    import asyncio
    ticks = _settle_ticks("oscillating", "healthy")
    naps = []

    async def nap(s):
        naps.append(s)
    monkeypatch.setattr(asyncio, "sleep", nap)
    store, ex, chain = Store(), Exec(), ReceiptChain(tmp_path / "r.jsonl")
    v = Verifier(lambda: next(ticks), store, ex, chain, TOPO, sleep=lambda s: None, max_wait_s=0, settle_s=15)
    r = await v.verify_async(plan(), finding(), receipt())
    assert r.verification.status == "pass" and naps == [15]


# ---------------------------------------------------------------- the settle loop (model lag)

def _looks(readings, truth=None):
    """One tick per look, each 20s newer than the last; `readings` are app's model states."""
    t0 = datetime.now(timezone.utc)
    return iter([(tick({"app": st}, truth=truth, cycle=10 + i), t0 + timedelta(seconds=20 * i))
                 for i, st in enumerate(readings)])


def test_the_model_settling_on_the_third_look_is_a_pass_with_two_rechecks(tmp_path):
    ticks = _looks(["degraded", "degraded", "healthy"], truth={})     # telemetry healthy throughout
    store, ex, chain, slept = Store(), Exec(), ReceiptChain(tmp_path / "r.jsonl"), []
    v = Verifier(lambda: next(ticks), store, ex, chain, TOPO, sleep=slept.append, poll_s=1, max_wait_s=30,
                 settle_s=15, max_settle_s=180)
    r = v.verify(plan(), finding(), receipt())
    assert r.verification.status == "pass" and slept == [15, 15]
    rc = r.verification.observed["recheck"]
    assert rc["count"] == 2 and rc["seconds"] == 30 and "disagree" in rc["first"]
    assert rc["last"] == {"dns": "healthy", "app": "healthy", "telemetry": {"dns": "healthy", "app": "healthy"}}
    assert store.closed == [("fnd-1", r.id)] and ex.compensated == []


def test_the_settle_deadline_ends_in_inconclusive_never_a_fail(tmp_path):
    ticks = _looks(["degraded"] * 6, truth={})
    store, ex, chain, slept, events = Store(), Exec(), ReceiptChain(tmp_path / "r.jsonl"), [], []
    v = Verifier(lambda: next(ticks), store, ex, chain, TOPO, emit=events.append, sleep=slept.append,
                 poll_s=1, max_wait_s=30, settle_s=15, max_settle_s=30)
    r = v.verify(plan(), finding(), receipt())
    assert r.verification.status == "inconclusive" and slept == [15, 15]
    rc = r.verification.observed["recheck"]
    assert rc["count"] == 2 and rc["seconds"] == 30 and rc["last"]["app"] == "degraded"
    assert "after 30s" in r.verification.observed["reason"] and "app" in r.verification.observed["reason"]
    assert ex.compensated == [] and not store.reopened and not store.closed
    assert [e for e in events if e["type"] == "escalation"][0]["decision"] == "human_plus"


def test_telemetry_turning_unhealthy_during_the_settle_is_a_fail(tmp_path):
    t0 = datetime.now(timezone.utc)
    ticks = iter([(tick({"app": "degraded"}, truth={}), t0),
                  (tick({"app": "degraded"}, truth={"app": "degraded"}, cycle=8), t0 + timedelta(seconds=20))])
    store, ex, chain, slept = Store(), Exec(), ReceiptChain(tmp_path / "r.jsonl"), []
    v = Verifier(lambda: next(ticks), store, ex, chain, TOPO, sleep=slept.append, poll_s=1, max_wait_s=30)
    r = v.verify(plan(), finding(), receipt())
    assert r.verification.status == "fail" and ex.compensated == [["s1"]] and store.reopened
    assert r.verification.observed["recheck"]["count"] == 1


def test_telemetry_oscillating_is_settling_not_evidence_against_the_fix():
    """The scorer flags 3+ state changes in 5 min as oscillating — a fault plus
    its fix produces exactly that. Model degraded + telemetry oscillating on a
    dependent is a settle, not a fail (live: kill_primary, 2026-09-22)."""
    v = evaluate(plan(), receipt(), tick({"app": "degraded"}, truth={"app": "oscillating", "dns": "healthy"}),
                 LATER, TOPO, now=NOW)
    assert v.status == "inconclusive" and v.recheck is True and "app model=degraded telemetry=oscillating" in v.reason
    # model unreachable with telemetry present and settling: a disagreement to
    # re-check, not terminal — the model held a restarted db unreachable
    v = evaluate(plan(), receipt(), tick({"dns": "unreachable"}, truth={"dns": "oscillating"}), LATER, TOPO, now=NOW)
    assert v.status == "inconclusive" and v.recheck is True
    v = evaluate(plan(), receipt(), tick({"dns": "unreachable"}, truth={"dns": "healthy"}), LATER, TOPO, now=NOW)
    assert v.status == "inconclusive" and v.recheck is True and "telemetry=healthy" in v.reason
    # no telemetry at all: terminal, as before
    v = evaluate(plan(), receipt(), tick({"dns": "unreachable"}), LATER, TOPO, now=NOW)
    assert v.status == "inconclusive" and v.recheck is False
    # telemetry degraded still confirms the model: fail
    v = evaluate(plan(), receipt(), tick({"app": "degraded"}, truth={"app": "degraded", "dns": "healthy"}),
                 LATER, TOPO, now=NOW)
    assert v.status == "fail"
