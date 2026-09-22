"""Phase 7 server: the loop end to end over the API with fakes for SABLE
and OVERLORD — finding → auto plan → gate → approval / countdown / auto →
execute → verify → receipt — plus the analyst, policy, lab and WS routes."""

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from console.policy import Policy
from console.server import Config, Loop, create_app
from console.tests.test_executor import FakeClient

LAB_NODES = [{"id": "dns", "type": "DNS_SERVER", "label": "DNS"},
             {"id": "app", "type": "APPLICATION_SERVICE", "label": "App"},
             {"id": "db", "type": "SERVER_VIRTUAL", "label": "DB"},
             {"id": "db-replica", "type": "SERVER_VIRTUAL", "label": "DB replica"},
             {"id": "proxy", "type": "LOAD_BALANCER", "label": "Proxy"},
             {"id": "prometheus", "type": "MONITORING_SERVER", "label": "Prometheus"}]
LAB_EDGES = [{"source": "app", "target": "dns", "type": "DNS_DEPENDENCY", "criticality": "HARD"},
             {"source": "app", "target": "db", "type": "SERVICE_DEPENDENCY", "criticality": "HARD"},
             {"source": "db", "target": "db-replica", "type": "REPLICATION_DEPENDENCY", "criticality": "SOFT"},
             {"source": "proxy", "target": "app", "type": "SERVICE_DEPENDENCY", "criticality": "HARD"},
             {"source": "proxy", "target": "db", "type": "SERVICE_DEPENDENCY", "criticality": "HARD"}]
IDX = {n["id"]: i for i, n in enumerate(LAB_NODES)}


def tick(states, conf=0.9):
    return {"cycle": 5, "source": "live", "avg_confidence": conf,
            "nodes": [{"id": i, "state": states.get(n["id"], "healthy"), "state_idx": 0, "confidence": conf,
                       "probs": {"healthy": 0.05, "degraded": 0.05, "failed": 0.85, "unreachable": 0.03, "oscillating": 0.02}
                       if states.get(n["id"]) else {"healthy": conf, "degraded": 0.05, "failed": 0.02, "unreachable": 0.02, "oscillating": 0.01},
                       "trend": "stable"} for i, n in enumerate(LAB_NODES)]}


def recs(root, state="failed"):
    if root is None:
        return {"summary": "All healthy.", "total_affected": 0, "actions": [], "root_cause": None}
    return {"summary": f"{root} {state}.", "total_affected": 1, "root_cause": IDX[root], "root_cause_id": root,
            "root_cause_label": root, "root_cause_type": next(n["type"] for n in LAB_NODES if n["id"] == root),
            "actions": [{"priority": 1, "action": "INVESTIGATE ROOT CAUSE", "target": root, "target_id": root,
                         "target_type": "X", "reason": f"{root} failed at tick 2", "recommendation": "restart"}]}


class FakeSable:
    def __init__(self):
        self.recs, self.chats = recs(None), []

    def status(self):
        return {"device": "cpu", "cycle": 5, "n_nodes": len(LAB_NODES)}

    def version_info(self):
        return {"version": "1.0"}

    def topology(self):
        return {"name": "lab", "nodes": LAB_NODES, "edges": LAB_EDGES}

    def recommendations(self):
        return self.recs

    def get(self, path):
        if path == "/api/nemotron/status":
            return {"available": True, "url": "anthropic:claude-sonnet-5", "provider": "claude", "model": "claude-sonnet-5"}
        raise KeyError(path)

    def post(self, path, body=None):
        if path == "/api/nemotron/chat":
            self.chats.append(body)
            return {"reply": "SABLE says dns is failed; I infer a restart will do."}
        raise KeyError(path)

    async def subscribe_ticks(self, on_tick, stop=None, on_message=None):
        while stop is None or not stop.is_set():
            await asyncio.sleep(0.05)


class Overlord(FakeClient):
    def __init__(self, **kw):
        super().__init__(stdout="dns\napp\ndb\ndb-replica\nproxy\nprometheus\n", **kw)
        self.llm_text = '{"steps": [{"step_id": "s1", "action_id": "rm_rf_everything", "target_node": "dns", "params": {}, "reversibility": "irreversible", "precondition": "none", "timeout_s": 5}], "verification": {"predicate": "node_healthy", "window_s": 30}, "finding_id": "x"}'

    def ping(self):
        return {"version": "0.30.0", "pid": 1}

    def complete(self, prompt, system="", provider="anthropic", model=None, purpose=None, **kw):
        import hashlib
        return {"text": self.llm_text, "provider": provider, "model": model or "fake", "stop": "end_turn",
                "usage": {"in": 1, "out": 1}, "refusal": None,
                "prompt_sha256": hashlib.sha256((system + "\n" + prompt).encode()).hexdigest(),
                "output_sha256": hashlib.sha256(self.llm_text.encode()).hexdigest()}


@pytest.fixture
def world(tmp_path):
    def make(policy=None, lab=False, sleep_scale=0.0):
        cfg = Config(data_dir=tmp_path / "data", lab_dir=tmp_path / "lab", lab_enabled=lab,
                     verify_stale_after_s=120)
        sable, ov = FakeSable(), Overlord()
        loop = Loop(cfg, sable, ov, policy=policy or Policy.load(), sleep=lambda s: time.sleep(s * sleep_scale))
        loop.verifier.max_wait_s = 5
        loop.verifier.poll_s = 0.05
        app = create_app(loop, serve_ui=False)
        return cfg, sable, ov, loop, app
    return make


def open_finding(loop, sable, root="dns", state="failed"):
    sable.recs = recs(root, state)
    f = loop.on_tick(tick({root: state}))
    assert f is not None and f.status == "open"
    return f


# ---------------------------------------------------------------- tests

def test_state_reports_both_engines_and_the_default_deny_policy(world):
    _, _, _, loop, app = world()
    with TestClient(app) as c:
        s = c.get("/api/state").json()
    assert s["sable"]["ok"] and s["sable"]["provider"] == "claude" and s["overlord"]["ok"]
    assert s["overlord"]["version"] == "0.30.0" and s["policy"]["default_deny"] is True
    assert s["counts"] == {"findings_open": 0, "plans_pending": 0, "receipts": 0}


def test_a_finding_is_planned_at_once_and_gated_by_the_shipped_default(world):
    _, sable, _, loop, app = world()
    with TestClient(app) as c:
        f = open_finding(loop, sable)
        rows = c.get("/api/findings?status=open").json()
        assert [r["id"] for r in rows] == [f.id] and rows[0]["root_cause"]["node_id"] == "dns"
        p = c.get(f"/api/findings/{f.id}/plan").json()
        assert p["steps"][0]["action_id"] == "restart_service" and p["gate"]["decision"] == "human"
        assert p["gate"]["rule_id"].startswith("default-deny:") and p["_status"] == "proposed"
        assert c.get("/api/findings/nope").status_code == 404
        assert c.get("/api/state").json()["counts"]["plans_pending"] == 1


def test_approve_executes_verifies_closes_and_receipts(world):
    _, sable, ov, loop, app = world()
    with TestClient(app) as c:
        f = open_finding(loop, sable)
        pid = c.get(f"/api/findings/{f.id}/plan").json()["id"]
        r = c.post(f"/api/plans/{pid}/approve", json={"actor": "keith", "decision": "approve"}).json()
        assert r["started"] is True and r["approval"]["actor"] == "keith"
        time.sleep(0.2)
        sable.recs = recs(None)
        loop.on_tick(tick({}))                                  # the fix landed: a fresh healthy tick
        assert loop.wait_idle(10)
        rc = c.get("/api/receipts").json()
        assert len(rc) == 1 and rc[0]["verification"]["status"] == "pass" and rc[0]["rollback"]["performed"] is False
        assert rc[0]["approvals"][0]["actor"] == "keith" and rc[0]["steps"][0]["status"] == "ok"
        assert c.get("/api/receipts/verify").json()["ok"] is True
        assert c.get(f"/api/findings/{f.id}").json()["status"] == "closed"
        assert c.get(f"/api/findings/{f.id}").json()["receipt_ids"] == [rc[0]["id"]]
        md = c.get(f"/api/receipts/{rc[0]['id']}/export?format=md").text
        assert "**pass**" in md and rc[0]["receipt_hash"] in md
        assert c.get(f"/api/receipts/{rc[0]['id']}/export").headers["content-disposition"].endswith('.json"')
        # the real command line reached OVERLORD with the plan step as its cause
        act = [e for e in ov.execs if e["cause"] and e["cause"].get("step_id") == rc[0]["steps"][0]["step_id"]
               and "phase" not in e["cause"]][0]
        assert act["argv"][-2:] == ["restart", "dns"]
        assert [a["action"] for a in ov.audits] == ["approval.recorded", "receipt.step", "receipt.close"]
        assert c.get("/api/log").json()[-1]["text"]


def test_failed_verification_compensates_and_reopens(world):
    _, sable, ov, loop, app = world()
    with TestClient(app) as c:
        f = open_finding(loop, sable)
        pid = loop.plan_of[f.id]
        c.post(f"/api/plans/{pid}/approve", json={"actor": "k", "decision": "approve"})
        time.sleep(0.2)
        loop.on_tick(tick({"dns": "failed"}))                    # still broken after the fix
        assert loop.wait_idle(10)
        rc = c.get("/api/receipts").json()[0]
        assert rc["verification"]["status"] == "fail" and rc["rollback"]["performed"] is True
        assert rc["rollback"]["steps"][0]["action_id"] == "restart_service"   # the catalog compensation
        assert c.get(f"/api/findings/{f.id}").json()["status"] == "reopened"


def test_reject_and_hold_never_execute(world):
    _, sable, ov, loop, app = world()
    with TestClient(app) as c:
        f = open_finding(loop, sable)
        pid = loop.plan_of[f.id]
        r = c.post(f"/api/plans/{pid}/approve", json={"actor": "k", "decision": "reject"}).json()
        assert r["started"] is False and r["status"] == "rejected"
        assert c.post(f"/api/plans/{pid}/approve", json={"actor": "k", "decision": "bogus"}).status_code == 400
        assert ov.execs == []


def test_target_policy_auto_executes_a_high_confidence_reversible_plan(world):
    _, sable, ov, loop, app = world(policy=Policy.target())
    with TestClient(app) as c:
        f = open_finding(loop, sable, root="dns", state="degraded")    # → clear_dns_cache, reversible
        p = c.get(f"/api/findings/{f.id}/plan").json()
        assert p["steps"][0]["action_id"] == "clear_dns_cache" and p["gate"]["decision"] == "auto"
        time.sleep(0.2)
        loop.on_tick(tick({}))
        assert loop.wait_idle(10)
        rc = c.get("/api/receipts").json()
        assert rc and rc[0]["approvals"][0]["actor"] == "policy" and rc[0]["verification"]["status"] == "pass"


def test_delay_countdown_can_be_cut_short_with_execute_now(world):
    _, sable, ov, loop, app = world(policy=Policy.target(), sleep_scale=0.01)   # 120 × 10ms countdown
    with TestClient(app) as c:
        f = open_finding(loop, sable, root="dns", state="failed")     # high + compensable → delay
        p = c.get(f"/api/findings/{f.id}/plan").json()
        assert p["gate"]["decision"] == "delay" and p["gate"]["delay_s"] == 120 and p["_status"] == "countdown"
        r = c.post(f"/api/plans/{p['id']}/approve", json={"actor": "k", "decision": "execute_now"}).json()
        assert r["started"] is True
        time.sleep(0.2)
        loop.on_tick(tick({}))
        assert loop.wait_idle(10)
        assert c.get("/api/receipts").json()[0]["approvals"][-1]["decision"] == "execute_now"


def test_llm_plan_that_invents_an_action_is_rejected_with_reasons(world):
    _, sable, ov, loop, app = world()
    with TestClient(app) as c:
        f = open_finding(loop, sable)
        r = c.post(f"/api/findings/{f.id}/plan", json={"planner": "llm"})
        assert r.status_code == 422 and any("rm_rf_everything" in x for x in r.json()["reasons"])
        # the template plan is untouched and still the finding's current plan
        assert c.get(f"/api/findings/{f.id}/plan").json()["steps"][0]["action_id"] == "restart_service"


def test_policy_can_be_replaced_and_is_validated(world):
    _, sable, ov, loop, app = world()
    with TestClient(app) as c:
        assert c.get("/api/policy").json()["matrix"]["high"]["reversible"] == "human"
        target = Policy.target().model_dump()
        assert c.put("/api/policy", json=target).json()["matrix"]["high"]["reversible"] == "auto"
        assert c.get("/api/state").json()["policy"]["default_deny"] is False
        bad = {**target, "matrix": {**target["matrix"], "high": {"reversible": "yolo", "compensable": "human", "irreversible": "human"}}}
        assert c.put("/api/policy", json=bad).status_code == 422
        assert [a["action"] for a in ov.audits] == ["approval.policy_changed"]


def test_analyst_chat_is_scoped_to_the_finding_and_flagged_generated(world):
    _, sable, ov, loop, app = world()
    with TestClient(app) as c:
        f = open_finding(loop, sable)
        r = c.post("/api/analyst/chat", json={"finding_id": f.id, "message": "what is upstream?", "history": []}).json()
        assert r["generated"] is True and r["provider"] == "claude" and "dns" in r["reply"]
        assert sable.chats[0]["message"].startswith(f"Regarding finding {f.id}")


def test_lab_faults_are_gated_and_whitelisted(world, tmp_path):
    _, _, _, loop, app = world(lab=False)
    with TestClient(app) as c:
        assert c.post("/api/lab/fault", json={"name": "stop_service"}).status_code == 403
    cfg, sable, ov, loop2, app2 = world(lab=True)
    faults = cfg.lab_dir / "faults"
    faults.mkdir(parents=True)
    (faults / "noop.sh").write_text("#!/usr/bin/env bash\necho injected\n")
    with TestClient(app2) as c:
        assert c.post("/api/lab/fault", json={"name": "rm_rf"}).status_code == 404
        r = c.post("/api/lab/fault", json={"name": "noop"}).json()
        assert r["exit_code"] == 0 and "injected" in r["output"]


def test_websocket_says_hello_and_streams_findings(world):
    _, sable, _, loop, app = world()
    with TestClient(app) as c:
        with c.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello" and hello["state"]["sable"]["ok"]
            open_finding(loop, sable)
            types = set()
            for _ in range(3):
                types.add(ws.receive_json()["type"])
            assert {"finding", "plan"} <= types or "log" in types


def test_spa_fallback_serves_index_for_client_routes_and_files_as_themselves(world, tmp_path):
    _, _, _, loop, _ = world()
    dist = tmp_path / "dist"; (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>console</html>")
    (dist / "assets" / "a.js").write_text("js;")
    app = create_app(loop, serve_ui=True, ui_dist=dist)
    with TestClient(app) as c:
        assert c.get("/").text == "<html>console</html>"
        assert c.get("/receipts/rcp-123").text == "<html>console</html>"
        assert c.get("/assets/a.js").text == "js;"
        assert c.get("/api/state").status_code == 200                # the API is not shadowed
        assert c.get("/../etc/passwd").text == "<html>console</html>"   # no escape from dist


def test_lab_fault_args_are_validated_and_helpers_are_not_faults(world):
    cfg, sable, ov, loop, app = world(lab=True)
    faults = cfg.lab_dir / "faults"; faults.mkdir(parents=True)
    (faults / "_lib.sh").write_text("echo helper\n")
    (faults / "stop_service.sh").write_text("#!/usr/bin/env bash\necho stopped ${1:-dns}\n")
    with TestClient(app) as c:
        assert c.post("/api/lab/fault", json={"name": "_lib"}).status_code == 404
        assert c.post("/api/lab/fault", json={"name": "stop_service", "args": ["dns; rm -rf /"]}).status_code == 400
        assert "stopped app" in c.post("/api/lab/fault", json={"name": "stop_service", "args": ["app"]}).json()["output"]
        assert "stopped dns" in c.post("/api/lab/fault", json={"name": "stop_service"}).json()["output"]


def test_analyst_reply_leads_with_a_fenced_sable_block(world):
    _, sable, ov, loop, app = world()
    with TestClient(app) as c:
        f = open_finding(loop, sable)
        r = c.post("/api/analyst/chat", json={"finding_id": f.id, "message": "why?", "history": []}).json()
        head, _, rest = r["reply"].partition("```\n")
        assert head.startswith("```sable\n") and json.loads(head[len("```sable\n"):])["finding"] == f.id
        assert rest.startswith("SABLE says")
