"""Post-first-run finding lifecycle.

  - a finding born from a transient blip (no execution in flight) resolves
    itself once SABLE reports its root cause healthy N ticks in a row;
  - a finding reopened by a failed verification gets a fresh plan;
  - re-planning is capped: after `max_attempts` failed receipts the finding
    escalates instead of looping fix / fail / fix.
"""

import time

from fastapi.testclient import TestClient

from console.tests.test_server import open_finding, recs, tick, world  # noqa: F401  (world is a fixture)


def healthy_ticks(loop, sable, n=1):
    sable.recs = recs(None)
    for _ in range(n):
        loop.on_tick(tick({}))


def approve(c, pid):
    return c.post(f"/api/plans/{pid}/approve", json={"actor": "k", "decision": "approve"}).json()


def plans_for(loop, fid):
    return [s for s in loop.plans.values() if s.finding_id == fid]


# ---------------------------------------------------------------- auto-resolve

def test_transient_blip_resolves_after_n_healthy_ticks(world):
    _, sable, ov, loop, app = world(resolve_after_ticks=3)
    with TestClient(app) as c:
        f = open_finding(loop, sable)
        pid = loop.plan_of[f.id]
        healthy_ticks(loop, sable, 2)
        assert c.get(f"/api/findings/{f.id}").json()["status"] == "open"        # not yet
        healthy_ticks(loop, sable, 1)
        got = c.get(f"/api/findings/{f.id}").json()
        assert got["status"] == "resolved" and got["receipt_ids"] == []
        assert c.get("/api/findings?status=resolved").json()[0]["id"] == f.id
        # the pending plan is withdrawn, unmapped, and can no longer be approved into execution
        assert loop.plans[pid].status == "withdrawn" and f.id not in loop.plan_of
        assert c.get(f"/api/findings/{f.id}/plan").status_code == 404
        assert approve(c, pid)["started"] is False and ov.execs == []
        assert c.get("/api/state").json()["counts"] == {"findings_open": 0, "plans_pending": 0, "receipts": 0}
        assert any("resolved" in ln["text"] and f.id in ln["text"] for ln in c.get("/api/log").json())
        # the same fault later is a NEW finding, not an occurrence of the resolved one
        g = open_finding(loop, sable)
        assert g.id != f.id and g.occurrences == 1 and loop.plan_of[g.id] != pid


def test_the_healthy_streak_resets_on_a_relapse(world):
    _, sable, _, loop, _ = world(resolve_after_ticks=3)
    f = open_finding(loop, sable)
    healthy_ticks(loop, sable, 2)
    sable.recs = recs("dns", "failed")
    assert loop.on_tick(tick({"dns": "failed"})).id == f.id                    # same finding, occurrence 2
    healthy_ticks(loop, sable, 2)
    assert loop.store.get(f.id).status == "open"
    healthy_ticks(loop, sable, 1)
    assert loop.store.get(f.id).status == "resolved"


def test_the_default_is_three_ticks(world):
    _, sable, _, loop, _ = world()
    f = open_finding(loop, sable)
    healthy_ticks(loop, sable, 2)
    assert loop.store.get(f.id).status == "open"
    healthy_ticks(loop, sable, 1)
    assert loop.store.get(f.id).status == "resolved"


def test_a_countdown_is_stopped_by_an_auto_resolve(world):
    from console.policy import Policy
    _, sable, ov, loop, app = world(policy=Policy.target(), sleep_scale=0.01, resolve_after_ticks=1)
    with TestClient(app) as c:
        f = open_finding(loop, sable)                      # high + compensable → delay countdown
        pid = loop.plan_of[f.id]
        assert loop.plans[pid].status == "countdown"
        healthy_ticks(loop, sable, 1)
        assert loop.plans[pid].status == "withdrawn"
        assert loop.wait_idle(10) and ov.execs == []
        assert c.get(f"/api/findings/{f.id}").json()["status"] == "resolved"


def test_a_finding_under_execution_is_closed_by_the_verifier_not_resolved(world):
    _, sable, ov, loop, app = world(resolve_after_ticks=1)
    with TestClient(app) as c:
        f = open_finding(loop, sable)
        assert approve(c, loop.plan_of[f.id])["started"] is True
        time.sleep(0.2)
        healthy_ticks(loop, sable, 1)                      # in flight: the sweep leaves it to the verifier
        assert loop.wait_idle(10)
        got = c.get(f"/api/findings/{f.id}").json()
        assert got["status"] == "closed" and len(got["receipt_ids"]) == 1
        assert c.get("/api/receipts").json()[0]["verification"]["status"] == "pass"


# ---------------------------------------------------------------- replan on reopen

def test_a_reopened_finding_gets_a_fresh_plan(world):
    _, sable, ov, loop, app = world()
    with TestClient(app) as c:
        f = open_finding(loop, sable)
        first = loop.plan_of[f.id]
        approve(c, first)
        time.sleep(0.2)
        loop.on_tick(tick({"dns": "failed"}))              # still broken after the fix
        assert loop.wait_idle(10)
        assert c.get(f"/api/findings/{f.id}").json()["status"] == "reopened"
        second = loop.plan_of[f.id]
        assert second != first
        assert loop.plans[first].status == "done" and loop.plans[second].status == "proposed"
        p = c.get(f"/api/findings/{f.id}/plan").json()
        assert p["id"] == second and p["_status"] == "proposed" and p["gate"]["decision"] == "human"
        # a later occurrence of the same condition does not spawn a third plan
        loop.on_tick(tick({"dns": "failed"}))
        assert loop.plan_of[f.id] == second and len(plans_for(loop, f.id)) == 2
        assert c.get("/api/state").json()["counts"]["plans_pending"] == 1


def test_replans_are_capped_and_the_finding_escalates(world):
    _, sable, ov, loop, app = world(max_attempts=2)
    events = []
    inner = loop._emit
    loop._emit = lambda ev: (inner(ev), events.append(ev))
    with TestClient(app) as c:
        f = open_finding(loop, sable)
        for _attempt in range(2):
            pid = loop.plan_of[f.id]
            assert approve(c, pid)["started"] is True
            time.sleep(0.2)
            loop.on_tick(tick({"dns": "failed"}))
            assert loop.wait_idle(10)
        got = c.get(f"/api/findings/{f.id}").json()
        assert got["status"] == "escalated" and len(got["receipt_ids"]) == 2
        assert len(plans_for(loop, f.id)) == 2 and loop.plans[loop.plan_of[f.id]].status == "done"
        assert c.get("/api/state").json()["counts"] == {"findings_open": 1, "plans_pending": 0, "receipts": 2}
        esc = [e for e in events if e["type"] == "escalation" and e.get("finding_id") == f.id]
        assert esc and esc[-1]["decision"] == "human_plus" and "2" in esc[-1]["reason"]
        assert any("escalat" in ln["text"] and f.id in ln["text"] for ln in c.get("/api/log").json())
        # the condition persisting folds into the escalated finding; still no new plan
        loop.on_tick(tick({"dns": "failed"}))
        g = loop.store.get(f.id)
        assert g.status == "escalated" and g.occurrences >= 2 and len(plans_for(loop, f.id)) == 2
        # an operator may still plan by hand …
        r = c.post(f"/api/findings/{f.id}/plan", json={"planner": "template"})
        assert r.status_code == 200 and r.json()["id"] != pid and loop.plans[r.json()["id"]].status == "proposed"
        # … and an escalated finding still resolves itself when the condition clears
        healthy_ticks(loop, sable, 3)
        assert loop.store.get(f.id).status == "resolved"


def test_max_attempts_one_escalates_on_the_first_failure(world):
    _, sable, ov, loop, app = world(max_attempts=1)
    with TestClient(app) as c:
        f = open_finding(loop, sable)
        approve(c, loop.plan_of[f.id])
        time.sleep(0.2)
        loop.on_tick(tick({"dns": "failed"}))
        assert loop.wait_idle(10)
        assert c.get(f"/api/findings/{f.id}").json()["status"] == "escalated"
        assert len(plans_for(loop, f.id)) == 1


# ---------------------------------------------------------------- store transitions

def test_store_resolve_and_escalate_transitions(tmp_path):
    from console.sable_bridge import FindingStore
    from console.tests.test_verify import finding
    store = FindingStore(tmp_path / "f.json")
    seen = []
    store.subscribe(lambda f: seen.append(f.status))
    f = store.upsert(finding())
    assert store.escalate(f.id).status == "escalated"
    assert store.list_open() == [store.get(f.id)]                  # escalated is still open
    assert store.find_open(f.dedup_key).id == f.id                 # … and still absorbs occurrences
    assert store.attach(f.id, "rcp-x").status == "escalated" and store.get(f.id).receipt_ids == ["rcp-x"]
    assert store.resolve(f.id).status == "resolved" and store.get(f.id).receipt_ids == ["rcp-x"]
    assert store.list_open() == [] and store.find_open(f.dedup_key) is None
    assert store.list("resolved")[0].id == f.id
    assert seen == ["open", "escalated", "escalated", "resolved"]
    assert FindingStore(tmp_path / "f.json").get(f.id).status == "resolved"
