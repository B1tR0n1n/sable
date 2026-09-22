"""CONSOLE_TOKEN: when set, every mutating route wants `Authorization:
Bearer <token>`; reads and /ws stay open. Unset → everything open, one
warning line at startup (the lab default)."""

from fastapi.testclient import TestClient

from console.policy import Policy
from console.tests.test_server import open_finding, world  # noqa: F401  (world is a fixture)

WARNING = "CONSOLE_TOKEN unset"


def bearer(tok):
    return {"Authorization": f"Bearer {tok}"}


def test_mutating_routes_require_the_token_when_set(world, monkeypatch):
    monkeypatch.setenv("CONSOLE_TOKEN", "s3cret-token")
    _, sable, ov, loop, app = world()
    with TestClient(app) as c:
        f = open_finding(loop, sable)
        pid = loop.plan_of[f.id]
        # reads and the stream stay open
        assert c.get("/api/state").status_code == 200
        assert c.get(f"/api/findings/{f.id}/plan").status_code == 200
        assert c.get("/api/log").status_code == 200
        with c.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "hello"
        # writes are refused without / with a wrong / with a malformed header
        body = {"actor": "k", "decision": "approve"}
        for headers in ({}, bearer("nope"), bearer("s3cret-toke"), bearer("s3cret-token "),
                        {"Authorization": "Basic s3cret-token"}, {"Authorization": "s3cret-token"}):
            r = c.post(f"/api/plans/{pid}/approve", json=body, headers=headers)
            assert r.status_code == 401 and set(r.json()) == {"error"} and r.json()["error"], headers
        assert c.put("/api/policy", json=Policy.target().model_dump()).status_code == 401
        assert c.post(f"/api/findings/{f.id}/plan", json={"planner": "template"}).status_code == 401
        assert c.post("/api/analyst/chat", json={"finding_id": f.id, "message": "?", "history": []}).status_code == 401
        assert c.post("/api/lab/fault", json={"name": "stop_service"}).status_code == 401
        assert ov.execs == [] and sable.chats == [] and loop.policy.is_default_deny
        # the right token passes through to the route
        r = c.post(f"/api/plans/{pid}/approve", json=body, headers=bearer("s3cret-token"))
        assert r.status_code == 200 and r.json()["started"] is True
        assert c.put("/api/policy", json=Policy.target().model_dump(), headers=bearer("s3cret-token")).status_code == 200
        assert loop.wait_idle(10)
    assert not any(WARNING in ln["text"] for ln in loop.log)


def test_unset_token_allows_everything_and_warns_once(world, monkeypatch):
    monkeypatch.delenv("CONSOLE_TOKEN", raising=False)
    _, sable, ov, loop, app = world()
    with TestClient(app) as c:
        f = open_finding(loop, sable)
        r = c.post(f"/api/plans/{f.id and loop.plan_of[f.id]}/approve", json={"actor": "k", "decision": "reject"})
        assert r.status_code == 200 and r.json()["status"] == "rejected"
        assert c.put("/api/policy", json=Policy.target().model_dump()).status_code == 200
    warnings = [ln for ln in loop.log if WARNING in ln["text"]]
    assert len(warnings) == 1 and warnings[0]["level"] == "warn"


def test_empty_token_counts_as_unset(world, monkeypatch):
    monkeypatch.setenv("CONSOLE_TOKEN", "")
    _, sable, ov, loop, app = world()
    with TestClient(app) as c:
        assert c.put("/api/policy", json=Policy.target().model_dump()).status_code == 200
    assert sum(1 for ln in loop.log if WARNING in ln["text"]) == 1
