"""Phase 4 acceptance: fault injection at every step position proves correct
reverse-order compensation; every run — pass or fail — produces a Receipt;
irreversible steps refuse to run without a recorded approval; the receipt
chain seals and verifies. A fake OVERLORD client makes it deterministic."""

from datetime import datetime, timezone

import pytest

from console.contracts import (
    Approval, BlastRadius, Compensation, Confidence, Finding, NodeRef, Plan, Receipt, Step,
    Verification,
)
from console.executor import ExecutionError, Executor, ReceiptChain
from console.executor.executor import render_argv

T0 = datetime(2026, 9, 22, 14, 0, 0, tzinfo=timezone.utc)
CTX = {"lab_dir": "/lab", "lab_compose": "/lab/docker-compose.yml"}


# ---------------------------------------------------------------- fakes

class FakeCatalog:
    """The catalog surface the executor uses: render(), checks, compensation."""
    checks = {
        "service_exists": {"argv": ["docker", "ps", "--services"], "expect": {"exit_code": 0, "stdout_contains": "{service}"}},
        "none": {"argv": ["true"], "expect": {"exit_code": 0}},
    }
    ACTIONS = {
        "restart_service": {"argv": ["docker", "compose", "-f", "{lab_compose}", "restart", "{service}"],
                            "grants": {"jail": False, "net": "none", "timeout_s": 60}, "target_dir": "{lab_dir}"},
        "set_config_value": {"argv": ["python3", "setcfg.py", "{file}", "{key}", "{value}"],
                             "then": ["docker", "compose", "-f", "{lab_compose}", "restart", "{service}"],
                             "grants": {"jail": True, "net": "none", "timeout_s": 30}, "target_dir": "{lab_dir}"},
        "failover_to_replica": {"argv": ["bash", "failover.sh", "{primary}", "{replica}"],
                                "grants": {"jail": False, "net": "none", "timeout_s": 60}, "target_dir": "{lab_dir}"},
        "failback_to_primary": {"argv": ["bash", "failback.sh", "{primary}", "{replica}"],
                                "grants": {"jail": False, "net": "none", "timeout_s": 60}, "target_dir": "{lab_dir}"},
        "nuke": {"argv": ["rm", "-rf", "/"], "grants": {"jail": True, "net": "none", "timeout_s": 5}, "target_dir": "{lab_dir}"},
    }

    def render(self, action_id, params, context):
        spec = self.ACTIONS[action_id]
        values = {**context, **params}
        out = {"argv": render_argv(spec["argv"], values),
               "grants": dict(spec["grants"]),
               "target_dir": render_argv([spec["target_dir"]], values)[0]}
        if spec.get("then"):
            out["then"] = render_argv(spec["then"], values)
        return out


class FakeSession:
    def __init__(self, client, sid):
        self.c, self.sid, self.state = client, sid, "pending"

    def commit(self, **kw):
        self.state = "committed"
        self.c.committed.append(self.sid)
        return {"committed": True}

    def rollback(self):
        self.state = "rolled_back"
        self.c.rolled_back.append(self.sid)
        return "/lab"


class FakeLive:
    def __init__(self, client, sid, grants, target):
        self.c, self.sid, self.grants, self.target = client, sid, grants, target
        self.expired = False

    def exec(self, cmd, timeout=None, cwd=None, label=None, on_output=None, cause=None):
        self.c.execs.append({"sid": self.sid, "argv": list(cmd), "timeout": timeout, "label": label, "cause": cause})
        rc, out = self.c.outcome(cmd)
        if rc == 124:
            self.expired = True
        return rc, out, []

    def close(self):
        return FakeSession(self.c, self.sid)


class FakeClient:
    """Scripted outcomes: `fail` is a set of argv substrings that fail;
    everything else exits 0 with a canned stdout."""

    def __init__(self, fail=(), timeout=(), stdout="dns app db db-replica proxy\n"):
        self.fail, self.timeout, self.stdout = set(fail), set(timeout), stdout
        self.n, self.execs, self.opened, self.committed, self.rolled_back = 0, [], [], [], []
        self.reverts, self.audits = [], []

    def outcome(self, cmd):
        line = " ".join(cmd)
        if any(t in line for t in self.timeout):
            return 124, "timed out"
        if any(f in line for f in self.fail):
            return 1, "boom"
        return 0, self.stdout

    def open(self, target, jail=False, net="host", timeout=None, net_allow=None, limits=None, **kw):
        self.n += 1
        sid = f"sid-{self.n}"
        g = {"jail": jail, "net": net, "timeout": timeout, "net_allow": net_allow, "limits": limits}
        self.opened.append({"sid": sid, "target": target, **g})
        return FakeLive(self, sid, g, target)

    def revert(self, sid, commit=False, force=False, stack=False):
        self.n += 1
        self.reverts.append((sid, commit, force))
        return {"sid": f"sid-{self.n}", "reverted": sid, "changes": [("modified", "x")], "skipped": [], "failed": [], "committed": commit}

    def audit(self, action, **fields):
        self.audits.append({"action": action, **fields})
        return {"seq": len(self.audits), "hash": "h" * 8, "action": action}

    def audit_head(self):
        return {"seq": len(self.audits), "hash": "h" * 8, "keyed": True}


# ---------------------------------------------------------------- fixtures

def finding():
    return Finding(id="fnd-1", site_id="lab", detection_mode="live_feed",
                   root_cause=NodeRef(node_id="dns", component_type="DNS_SERVER", state="failed"),
                   confidence=Confidence(score=0.9, method="engine_native"), severity="high",
                   engine_version="sable@test", created_at=T0)


def three_step_plan():
    """s1 reversible (config), s2 compensable (restart), s3 compensable (failover)."""
    return Plan(id="pln-1", finding_id="fnd-1", created_at=T0, steps=[
        Step(step_id="s1", action_id="set_config_value", target_node="app",
             params={"file": "app.conf", "key": "db_host", "value": "db.lab", "service": "app"},
             reversibility="reversible", precondition="none", timeout_s=30),
        Step(step_id="s2", action_id="restart_service", target_node="dns", params={"service": "dns"},
             reversibility="compensable",
             compensation=Compensation(action_id="restart_service", params={"service": "dns"}),
             precondition="service_exists", timeout_s=60),
        Step(step_id="s3", action_id="failover_to_replica", target_node="db", params={"primary": "db", "replica": "db-replica"},
             reversibility="compensable",
             compensation=Compensation(action_id="failback_to_primary", params={"primary": "db", "replica": "db-replica"}),
             precondition="none", timeout_s=60),
    ], blast_radius=BlastRadius(nodes=["app", "dns", "db"], count=3),
        verification=Verification(predicate="node_healthy", window_s=30))


APPROVED = [Approval(actor="operator", decision="approve", timestamp=T0)]


def run(client, plan=None, approvals=APPROVED):
    events = []
    ex = Executor(client, FakeCatalog(), CTX, emit=events.append)
    r = ex.execute(plan or three_step_plan(), finding(), approvals)
    return r, ex, events


# ---------------------------------------------------------------- tests

def test_happy_path_commits_each_step_with_cause_and_minimum_scope():
    c = FakeClient()
    r, ex, events = run(c)
    assert [s.status for s in r.steps] == ["ok", "ok", "ok"]
    assert not r.rollback.performed and r.rollback.steps == []
    assert c.committed == ["sid-1", "sid-2", "sid-3"] and c.rolled_back == []
    # s1: jailed, net none, per-step timeout; s2: not jailed (docker socket)
    assert c.opened[0]["jail"] is True and c.opened[0]["net"] == "none" and c.opened[0]["timeout"] == 30
    assert c.opened[1]["jail"] is False and c.opened[1]["target"] == "/lab"
    # the action's exec carries the cause naming the plan step
    act = [e for e in c.execs if e["label"] == "s2"][0]
    assert act["cause"] == {"plan_id": "pln-1", "step_id": "s2", "action_id": "restart_service", "finding_id": "fnd-1"}
    assert act["argv"] == ["docker", "compose", "-f", "/lab/docker-compose.yml", "restart", "dns"]
    assert act["timeout"] == 60
    # precondition ran first inside the same session, then the action, then s1's follow-up
    labels = [e["label"] for e in c.execs]
    assert labels == ["s1/pre", "s1", "s1/then", "s2/pre", "s2", "s3/pre", "s3"]
    # snapshots reference the sessions; the receipt's session is the first step's
    assert [s.snapshot_ref for s in r.snapshots] == ["sid-1", "sid-2", "sid-3"] and r.session_id == "sid-1"
    assert {e["type"] for e in events} >= {"step", "log"}
    assert [a["action"] for a in c.audits].count("receipt.step") == 3


@pytest.mark.parametrize("fail_step, expect_statuses, expect_comp", [
    ("setcfg.py", ["failed"], []),                                      # s1 fails: nothing to compensate
    ("restart dns", ["ok", "failed"], ["overlord.revert"]),             # s2 fails: revert s1
    ("failover.sh", ["ok", "ok", "failed"], ["restart_service", "overlord.revert"]),  # s3 fails: comp s2, then revert s1
])
def test_failure_at_each_position_compensates_completed_steps_in_reverse(fail_step, expect_statuses, expect_comp):
    c = FakeClient(fail=[fail_step])
    r, ex, _ = run(c)
    assert [s.status for s in r.steps] == expect_statuses
    failed_sid = r.steps[-1].session_id
    assert failed_sid in c.rolled_back, "the failed step's own session is rolled back"
    assert r.rollback.performed == bool(expect_comp)
    assert [s.action_id for s in r.rollback.steps] == expect_comp
    assert all(s.status == "ok" for s in r.rollback.steps)
    if "overlord.revert" in expect_comp:
        assert c.reverts == [("sid-1", True, True)]           # the reversible step's committed session is reverted
    if "restart_service" in expect_comp:
        comp = [e for e in c.execs if e["label"] == "s2/compensate"][0]
        assert comp["cause"]["compensates"] == "restart_service"
        assert comp["argv"][-1] == "dns"
    # order on the receipt is reverse of execution
    assert [s.step_id for s in r.rollback.steps] == [s.step_id for s in reversed(r.steps[:-1])]


def test_precondition_failure_stops_before_anything_runs():
    c = FakeClient(stdout="app db\n")                         # 'dns' absent → service_exists fails for s2
    r, ex, _ = run(c)
    assert [s.status for s in r.steps] == ["ok", "precondition_failed"]
    assert r.steps[1].error.startswith("service_exists")
    assert not any(e["label"] == "s2" for e in c.execs), "the action never ran"
    assert "sid-2" in c.rolled_back and "sid-2" not in c.committed
    assert [s.action_id for s in r.rollback.steps] == ["overlord.revert"]   # s1 undone


def test_timeout_is_recorded_as_timed_out():
    c = FakeClient(timeout=["failover.sh"])
    r, _, _ = run(c)
    assert r.steps[-1].status == "timed_out"
    assert r.rollback.performed


def test_irreversible_step_refuses_without_recorded_approval():
    plan = three_step_plan().model_copy(update={"steps": [Step(
        step_id="x", action_id="nuke", target_node="db", params={}, reversibility="irreversible",
        precondition="none", timeout_s=5)]})
    c = FakeClient()
    r, _, _ = run(c, plan, approvals=[Approval(actor="op", decision="hold", timestamp=T0)])
    assert r.steps[0].status == "skipped" and "approval" in r.steps[0].error
    assert c.opened == [] and c.execs == [], "nothing was opened or run"
    r2, _, _ = run(FakeClient(), plan, approvals=APPROVED)
    assert r2.steps[0].status == "ok"


def test_every_run_produces_a_sealed_receipt_and_the_chain_verifies(tmp_path):
    chain = ReceiptChain(tmp_path / "receipts.jsonl")
    c = FakeClient()
    r1, ex, _ = run(c)
    ex.finalize(r1, chain)
    r2, ex2, _ = run(FakeClient(fail=["restart dns"]))
    ex2.finalize(r2, chain)
    assert r1.verify_hash() and r2.verify_hash() and r2.prev_receipt_hash == r1.receipt_hash
    assert chain.verify() == {"ok": True, "broken_at": None, "count": 2}
    assert r1.audit_ref and r1.audit_ref["seq"] >= 1               # mirrored onto OVERLORD's audit chain
    assert [a["action"] for a in c.audits][-1] == "receipt.close"
    reloaded = ReceiptChain(tmp_path / "receipts.jsonl")
    assert reloaded.verify()["ok"] and reloaded.head == r2.receipt_hash
    assert [x.id for x in reloaded.list()] == [r2.id, r1.id]
    assert reloaded.list(finding_id="nope") == []


def test_compensation_failure_is_recorded_and_the_rest_still_runs():
    c = FakeClient(fail=["failover.sh", "restart dns"])        # s3 fails; then compensating s2 also fails
    # make s2 itself succeed on first run but fail as a compensation: script by call count
    calls = {"n": 0}
    orig = c.outcome

    def outcome(cmd):
        line = " ".join(cmd)
        if "restart dns" in line:
            calls["n"] += 1
            return (0, "ok") if calls["n"] == 1 else (1, "boom")
        return orig(cmd)
    c.outcome = outcome
    r, _, _ = run(c)
    assert [s.status for s in r.steps] == ["ok", "ok", "failed"]
    assert [(s.action_id, s.status) for s in r.rollback.steps] == [("restart_service", "failed"), ("overlord.revert", "ok")]
    assert r.rollback.steps[0].error == "exit 1"


def test_render_argv_refuses_unbound_placeholders():
    assert render_argv(["a", "{x}"], {"x": 1}) == ["a", "1"]
    with pytest.raises(ExecutionError):
        render_argv(["{missing}"], {})
