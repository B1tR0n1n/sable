"""Ground truth beside the model: where the scorer's reading comes from, how
the console attaches it to a tick, and what the loop does when the model
lags telemetry after a fix (live evidence 2026-09-22: `app` read
oscillating/degraded for ~150s while Prometheus said healthy throughout)."""

import time

from fastapi.testclient import TestClient

from console.sable_bridge import attach_ground_truth, ground_truth_states
from console.tests.test_lifecycle import approve, healthy_ticks
from console.tests.test_server import LAB_NODES, open_finding, recs, tick, world  # noqa: F401  (world is a fixture)
from console.topology import Topology

TOPO = Topology(LAB_NODES, [])


# ---------------------------------------------------------------- reading it

def test_ground_truth_list_maps_positionally_to_topology_ids():
    t = tick({"app": "degraded"}, truth={"app": "healthy", "db": "failed"})
    assert ground_truth_states(t, TOPO) == {"dns": "healthy", "app": "healthy", "db": "failed", "db-replica": "healthy",
                                            "proxy": "healthy", "prometheus": "healthy"}


def test_per_node_truth_and_none_entries_are_honoured():
    t = tick({})
    t["nodes"][0]["truth"] = "degraded"
    assert ground_truth_states(t, TOPO) == {"dns": "degraded"}
    t = tick({}, truth={})
    t["ground_truth"][1] = None                                   # unknown for app, known elsewhere
    assert "app" not in ground_truth_states(t, TOPO) and ground_truth_states(t, TOPO)["dns"] == "healthy"
    assert ground_truth_states(tick({}), TOPO) == {}
    assert ground_truth_states({**tick({}), "ground_truth": None}, TOPO) == {}


# ---------------------------------------------------------------- fetching it

class NodeClient:
    """SABLE's /api/node/{idx}: trajectory entries carry `truth` only when the
    engine was given ground truth for that cycle."""

    def __init__(self, truth_by_idx, cycle=5, fail_on=()):
        self.truth, self.cycle, self.fail_on, self.calls = truth_by_idx, cycle, set(fail_on), []

    def node(self, idx):
        self.calls.append(idx)
        if idx in self.fail_on:
            raise RuntimeError("boom")
        entry = {"cycle": self.cycle, "prediction": "healthy"}
        if idx in self.truth:
            entry["truth"] = self.truth[idx]
        return {"node_id": idx, "trajectory": [{"cycle": self.cycle - 1, "prediction": "failed", "truth": "failed"}, entry]}


def test_attach_fetches_the_matching_cycle_and_leaves_the_tick_alone():
    t = tick({"app": "degraded"})
    out = attach_ground_truth(t, NodeClient({0: "healthy", 1: "healthy", 2: "degraded"}))
    assert "ground_truth" not in t and out is not t                  # a new dict; the input is untouched
    assert out["ground_truth"] == [0, 0, 1, None, None, None]
    assert ground_truth_states(out, TOPO) == {"dns": "healthy", "app": "healthy", "db": "degraded"}


def test_attach_ignores_a_trajectory_for_another_cycle_and_survives_errors():
    out = attach_ground_truth(tick({}), NodeClient({0: "healthy"}, cycle=9))
    assert out["ground_truth"] is None                               # tried, nothing usable: never refetched
    out = attach_ground_truth(tick({}), NodeClient({0: "healthy", 1: "healthy"}, fail_on={1}))
    assert out["ground_truth"] == [0, None, None, None, None, None]
    assert attach_ground_truth({**tick({}), "ground_truth": [0] * 6}, NodeClient({}))["ground_truth"] == [0] * 6


def test_the_stub_emits_ground_truth_and_truth_in_its_node_report():
    from console.lab.sable_stub import StubEngine
    e = StubEngine(["dns", "app"], {"dns": "DNS_SERVER", "app": "APPLICATION_SERVICE"})
    t = e.tick_from_states({"app": "degraded"}, {})
    assert t["ground_truth"] == [0, 1]                               # the stub has no model: state == truth
    assert e.get_node_report(1)["trajectory"][-1] == {"cycle": 1, "prediction": "degraded", "truth": "degraded"}


# ---------------------------------------------------------------- the loop

def _wait_for_log(loop, needle, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(needle in ln["text"] for ln in loop.log):
            return True
        time.sleep(0.02)
    return False


def test_model_lag_after_a_fix_escalates_instead_of_rolling_back(world):
    _, sable, ov, loop, app = world(verify_max_settle_s=15)          # one settle look, then the deadline
    events = []
    inner = loop._emit
    loop._emit = lambda ev: (inner(ev), events.append(ev))
    with TestClient(app) as c:
        f = open_finding(loop, sable)
        assert approve(c, loop.plan_of[f.id])["started"] is True
        time.sleep(0.2)
        sable.recs = recs(None)
        loop.on_tick(tick({"app": "degraded"}, truth={}))           # the model lags; Prometheus: all healthy
        assert _wait_for_log(loop, "re-checking in")
        loop.on_tick(tick({"app": "oscillating"}, truth={}))        # a newer look: still lagging
        assert loop.wait_idle(15)
        rc = c.get("/api/receipts").json()[0]
        assert rc["verification"]["status"] == "inconclusive" and rc["rollback"]["performed"] is False
        assert "disagree" in rc["verification"]["observed"]["reason"] and rc["verification"]["observed"]["recheck"]["count"] == 1
        assert [e["argv"][-2:] for e in ov.execs if e["argv"][-2:] == ["restart", "dns"]] == [["restart", "dns"]]   # no compensation
        got = c.get(f"/api/findings/{f.id}").json()
        assert got["status"] == "open" and got["receipt_ids"] == [rc["id"]]                                 # not reopened; receipt linked
        esc = [e for e in events if e["type"] == "escalation" and e.get("finding_id") == f.id]
        assert esc and esc[0]["decision"] == "human_plus"
        assert len([s for s in loop.plans.values() if s.finding_id == f.id]) == 1                            # no re-plan
        healthy_ticks(loop, sable, 3)                                                                       # the model catches up
        assert c.get(f"/api/findings/{f.id}").json()["status"] == "resolved"


def test_the_loop_fetches_ground_truth_once_per_tick_when_the_engine_omits_it(world):
    _, sable, ov, loop, app = world()
    calls = []
    sable.node = lambda idx: (calls.append(idx), {"node_id": idx, "trajectory": [{"cycle": 5, "prediction": "healthy", "truth": "healthy"}]})[1]
    loop.on_tick(tick({}))
    t1, _ = loop.latest_tick()
    t2, _ = loop.latest_tick()
    assert t1["ground_truth"] == [0] * len(LAB_NODES) and t2 is t1 and calls == list(range(len(LAB_NODES)))
    loop.on_tick(tick({}, truth={}))                                 # carried by the tick: no fetch
    loop.latest_tick()
    assert calls == list(range(len(LAB_NODES)))
