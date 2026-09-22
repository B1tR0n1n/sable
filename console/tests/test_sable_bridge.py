"""Phase 2 — SABLE finding emitter. No network: SABLE's output is replayed
from fixtures shaped exactly like docker/sable_engine.py:179-225 (tick),
:448-453 + docker/server.py:120-160 (enriched recommendations) and
docker/server.py:319-333 (topology), with the real msp_demo.yaml graph."""

from __future__ import annotations

import asyncio
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from console.contracts import Finding
from console.sable_bridge import (
    FindingEmitter,
    FindingStore,
    MappingError,
    SableClient,
    SableError,
    create_app,
    dedup_key_for,
    node_info_from_recs,
    to_finding,
)
from console.topology import Topology

FIXTURES = Path(__file__).parent / "fixtures"
SITE = "lab-1"
ENGINE = "sable@deadbeef engine/1.0 device/cpu"


def load(name: str):
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.fixture(scope="module")
def topology() -> Topology:
    return Topology.from_api(load("topology_msp_demo"))


@pytest.fixture
def scenario():
    return load("scenario_dns_cascade")["ticks"]


def roundtrip(f: Finding) -> Finding:
    back = Finding.model_validate(f.model_dump(mode="json"))
    assert back == f
    return back


# ---------------------------------------------------------------- fixtures are SABLE-shaped


def test_fixtures_match_sable_shapes(topology):
    tick = load("tick_dns_failed")
    assert set(tick) >= {"cycle", "nodes", "class_counts", "routing", "avg_confidence",
                         "min_confidence", "max_confidence", "inference_ms"}
    assert len(tick["nodes"]) == len(topology.nodes) == 32
    for n in tick["nodes"]:
        assert set(n) >= {"id", "state", "state_idx", "confidence", "probs", "transition", "trend", "routing",
                          "label", "topo_id", "component_type"}
        assert abs(sum(n["probs"].values()) - 1.0) < 1e-3
        assert n["confidence"] == max(n["probs"].values())
        assert n["topo_id"] == topology.id_at(n["id"])
    recs = load("recs_dns_failed")
    assert set(recs) == {"summary", "total_affected", "actions", "root_cause",
                         "root_cause_label", "root_cause_id", "root_cause_type"}
    assert recs["actions"][0]["action"] == "INVESTIGATE ROOT CAUSE"
    assert recs["root_cause_id"] == topology.id_at(recs["root_cause"]) == "dns-1"
    assert recs["root_cause_type"] == "DNS_SERVER"
    live = load("tick_san_unreachable")
    assert live["source"] == "live"
    assert "component_type" not in live["nodes"][0]      # the live path enriches label/topo_id only
    mc = load("tick_dns_failed_mc")
    assert mc["mc_samples"] == 10 and "mc_agreement" in mc["nodes"][0]


# ---------------------------------------------------------------- mapping


def test_dns_failed_root_is_critical_live_feed(topology):
    f = to_finding(load("tick_dns_failed"), load("recs_dns_failed"), topology, SITE, ENGINE)
    assert f is not None
    assert f.root_cause.model_dump() == {"node_id": "dns-1", "component_type": "DNS_SERVER", "state": "failed"}
    assert f.detection_mode == "live_feed"
    assert f.confidence.method == "engine_native" and f.confidence.samples is None
    assert f.confidence.score == load("tick_dns_failed")["nodes"][7]["confidence"]
    assert f.severity == "critical"                      # 4 affected dependents of dns-1
    assert {a.node_id for a in f.affected_nodes} == {"vm-web-1", "vm-web-2", "vm-app-1", "app-erp", "mon-1"}
    assert {a.component_type for a in f.affected_nodes} == {"SERVER_VIRTUAL", "APPLICATION_SERVICE",
                                                            "MONITORING_SERVER"}
    assert f.summary == load("recs_dns_failed")["summary"] and f.summary_generated is False
    assert f.site_id == SITE and f.engine_version == ENGINE
    assert f.status == "open" and f.occurrences == 1 and f.updated_at is None
    metrics = {e.metric for e in f.evidence}
    assert {"dns-1:state_confidence", "dns-1:p_failed", "dns-1:first_affected_tick",
            "vm-web-1:state_confidence", "vm-web-1:first_affected_tick"} <= metrics
    assert all(e.unit == "p" for e in f.evidence if e.metric.endswith(("state_confidence", "p_failed")))
    assert next(e for e in f.evidence if e.metric == "dns-1:first_affected_tick").value == 2.0
    roundtrip(f)


def test_failed_root_with_few_dependents_is_high(topology, scenario):
    first = scenario[0]                                   # dns-1 failed, nothing else yet
    f = to_finding(first["tick"], first["recs"], topology, SITE, ENGINE)
    assert f.severity == "high" and f.affected_nodes == []
    roundtrip(f)


def test_application_service_degraded_is_medium(topology):
    f = to_finding(load("tick_app_degraded"), load("recs_app_degraded"), topology, SITE, ENGINE)
    assert f.root_cause.node_id == "app-erp" and f.root_cause.component_type == "APPLICATION_SERVICE"
    assert f.root_cause.state == "degraded"
    assert f.severity == "medium" and f.detection_mode == "live_feed"
    assert f.confidence.method == "engine_native"
    assert [a.node_id for a in f.affected_nodes] == ["vm-web-1"]
    assert f.summary_generated is False
    roundtrip(f)


def test_unreachable_storage_array_is_unmonitored_gap(topology):
    f = to_finding(load("tick_san_unreachable"), load("recs_san_unreachable"), topology, SITE, ENGINE)
    assert f.root_cause.model_dump() == {"node_id": "san-1", "component_type": "STORAGE_ARRAY",
                                         "state": "unreachable"}
    assert f.detection_mode == "unmonitored_gap"
    assert f.severity == "critical"                      # hyp-1, hyp-2, vm-db-1, san-tgt-1 all depend on san-1
    assert {a.node_id for a in f.affected_nodes} == {"hyp-1", "hyp-2", "vm-db-1", "san-tgt-1", "mon-1"}
    # component types come from the topology (the live tick carries none)
    assert {a.component_type for a in f.affected_nodes} == {"HYPERVISOR", "SERVER_VIRTUAL", "STORAGE_TARGET",
                                                            "MONITORING_SERVER"}
    assert f.summary_generated is False
    roundtrip(f)


def test_mc_dropout_tick_reports_agreement_and_samples(topology):
    tick = load("tick_dns_failed_mc")
    f = to_finding(tick, load("recs_dns_failed"), topology, SITE, ENGINE)
    assert f.confidence.method == "mc_dropout" and f.confidence.samples == 10
    assert f.confidence.score == tick["nodes"][7]["mc_agreement"]
    assert f.severity == "critical" and f.root_cause.node_id == "dns-1"
    assert "dns-1:mc_agreement" in {e.metric for e in f.evidence}
    roundtrip(f)


def test_healthy_tick_yields_no_finding(topology):
    assert to_finding(load("tick_healthy"), load("recs_healthy"), topology, SITE, ENGINE) is None
    assert dedup_key_for(load("tick_healthy"), load("recs_healthy"), topology, SITE) is None
    # a root index whose tick state is healthy (stale recs) is also nothing to report
    tick = copy.deepcopy(load("tick_dns_failed"))
    tick["nodes"][7]["state"] = "healthy"
    assert to_finding(tick, load("recs_dns_failed"), topology, SITE, ENGINE) is None


def test_missing_root_in_tick_is_a_mapping_error(topology):
    tick = copy.deepcopy(load("tick_dns_failed"))
    tick["nodes"] = [n for n in tick["nodes"] if n["id"] != 7]
    with pytest.raises(MappingError):
        to_finding(tick, load("recs_dns_failed"), topology, SITE, ENGINE)


def test_ids_fall_back_when_topology_is_short(topology):
    """The live path can present more nodes than the YAML knows (server.py:471-473
    rebuilds _topo_nodes as UNKNOWN); ids fall back to topo_id / node-<i> and
    types to the tick's component_type."""
    small = Topology(list(topology.nodes.values())[:8], [], "short")   # keeps dns-1 (index 7)
    tick = copy.deepcopy(load("tick_dns_failed"))
    f = to_finding(tick, load("recs_dns_failed"), small, SITE, ENGINE)
    ids = {a.node_id for a in f.affected_nodes}
    assert ids == {"vm-web-1", "vm-web-2", "vm-app-1", "app-erp", "mon-1"}   # via topo_id
    assert {a.component_type for a in f.affected_nodes} >= {"SERVER_VIRTUAL"}  # via tick enrichment
    for n in tick["nodes"]:
        n.pop("topo_id", None); n.pop("component_type", None)
    f = to_finding(tick, load("recs_dns_failed"), small, SITE, ENGINE)
    assert "node-31" in {a.node_id for a in f.affected_nodes}
    assert {a.component_type for a in f.affected_nodes} == {"UNKNOWN"}


def test_node_facts_are_parsed_from_action_reasons():
    facts = node_info_from_recs(load("recs_dns_failed"))
    assert facts["dns-1"] == {"first_affected_tick": 2}
    assert facts["vm-app-1"] == {"first_affected_tick": 3}
    # a future server publishing recs["nodes"] wins over the parsed reasons
    facts = node_info_from_recs({"nodes": [{"node_id": "dns-1", "first_affected_tick": 9,
                                            "ticks_in_current_state": 4}], "actions": []})
    assert facts["dns-1"] == {"first_affected_tick": 9, "ticks_in_current_state": 4}


def test_timestamp_is_tick_time_or_now(topology):
    tick = copy.deepcopy(load("tick_dns_failed"))
    tick["timestamp"] = "2026-09-22T10:00:00Z"
    f = to_finding(tick, load("recs_dns_failed"), topology, SITE, ENGINE)
    assert {e.timestamp for e in f.evidence} == {datetime(2026, 9, 22, 10, tzinfo=timezone.utc)}
    before = datetime.now(timezone.utc)
    f = to_finding(load("tick_dns_failed"), load("recs_dns_failed"), topology, SITE, ENGINE)
    assert all(e.timestamp >= before for e in f.evidence)


# ---------------------------------------------------------------- dedup


def test_persisting_condition_updates_one_finding(topology, scenario):
    prev = None
    for step in scenario[:3]:
        f = to_finding(step["tick"], step["recs"], topology, SITE, ENGINE, previous=prev)
        if prev is not None:
            assert f.id == prev.id and f.created_at == prev.created_at
        prev = f
        time.sleep(0.001)
    assert prev.occurrences == 3
    assert prev.updated_at is not None and prev.updated_at > prev.created_at
    assert prev.status == "open"
    assert prev.severity == "critical"                    # refreshed as the cascade grew (was high)
    assert {a.node_id for a in prev.affected_nodes} == {"vm-web-1", "vm-web-2", "vm-app-1"}
    roundtrip(prev)


def test_different_root_state_is_a_different_finding(topology, scenario):
    step = scenario[2]
    a = to_finding(step["tick"], step["recs"], topology, SITE, ENGINE)
    tick = copy.deepcopy(step["tick"])
    tick["nodes"][7]["state"] = "degraded"
    assert dedup_key_for(tick, step["recs"], topology, SITE) != a.dedup_key
    with pytest.raises(MappingError):                     # the mapper refuses a mismatched `previous`
        to_finding(tick, step["recs"], topology, SITE, ENGINE, previous=a)
    b = to_finding(tick, step["recs"], topology, SITE, ENGINE)
    assert b.id != a.id and b.severity == "medium"


def test_scenario_replay_is_stable_and_deduplicated(topology, scenario):
    """PLAN.md Phase 2 acceptance, on fixture data: five consecutive ticks of
    one condition produce one Finding with five occurrences, whose root and
    dedup key never change, and whose component types are all real."""
    store = FindingStore(None)
    keys, ids = set(), set()
    for step in scenario:
        prev = store.find_open(dedup_key_for(step["tick"], step["recs"], topology, SITE))
        f = store.upsert(to_finding(step["tick"], step["recs"], topology, SITE, ENGINE, previous=prev))
        keys.add(f.dedup_key); ids.add(f.id)
    assert len(store.list()) == 1 and len(keys) == 1 and len(ids) == 1
    only = store.list()[0]
    assert only.occurrences == 5 and only.root_cause.node_id == "dns-1"
    assert {a.component_type for a in only.affected_nodes} == {"SERVER_VIRTUAL", "APPLICATION_SERVICE"}
    assert {a.node_id for a in only.affected_nodes} == {"vm-web-1", "vm-web-2", "vm-app-1", "app-erp"}
    roundtrip(only)


def test_summarize_hook_flags_generated_summary(topology):
    f = to_finding(load("tick_dns_failed"), load("recs_dns_failed"), topology, SITE, ENGINE,
                   summarize=lambda fnd: f"{fnd.root_cause.node_id} is down; {len(fnd.affected_nodes)} impacted")
    assert f.summary == "dns-1 is down; 5 impacted" and f.summary_generated is True
    roundtrip(f)


# ---------------------------------------------------------------- store


def test_store_dedups_new_objects_with_same_key(topology, scenario):
    store = FindingStore(None)
    seen = []
    store.subscribe(seen.append)
    a = store.upsert(to_finding(scenario[0]["tick"], scenario[0]["recs"], topology, SITE, ENGINE))
    b = store.upsert(to_finding(scenario[1]["tick"], scenario[1]["recs"], topology, SITE, ENGINE))
    assert b.id == a.id and b.occurrences == 2 and b.updated_at is not None
    assert len(store) == 1 and [s.id for s in seen] == [a.id, a.id]
    # same id -> replaced as given, no extra occurrence
    c = store.upsert(b.model_copy(update={"summary": "edited"}))
    assert c.occurrences == 2 and store.get(a.id).summary == "edited"


def test_store_close_reopen_and_persistence(tmp_path, topology):
    path = tmp_path / "findings.json"
    store = FindingStore(str(path))
    events = []
    store.subscribe(lambda f: events.append((f.id, f.status)))
    f = store.upsert(to_finding(load("tick_dns_failed"), load("recs_dns_failed"), topology, SITE, ENGINE))
    g = store.upsert(to_finding(load("tick_app_degraded"), load("recs_app_degraded"), topology, SITE, ENGINE))
    assert path.exists()

    closed = store.close(f.id, "rcp-1")
    assert closed.status == "closed" and closed.receipt_ids == ["rcp-1"]
    assert store.list(status="open") == [g]
    assert store.find_open(f.dedup_key) is None
    # a closed condition recurring is a NEW finding
    h = store.upsert(to_finding(load("tick_dns_failed"), load("recs_dns_failed"), topology, SITE, ENGINE))
    assert h.id != f.id and h.occurrences == 1

    reopened = store.reopen(f.id, "rcp-2")
    assert reopened.status == "reopened" and reopened.receipt_ids == ["rcp-1", "rcp-2"]
    assert store.find_open(f.dedup_key).id in (f.id, h.id)
    with pytest.raises(KeyError):
        store.close("fnd-nope")
    assert [e[1] for e in events] == ["open", "open", "closed", "open", "reopened"]

    reloaded = FindingStore(str(path))
    assert {x.id: x for x in reloaded.list()} == {x.id: x for x in store.list()}
    assert reloaded.get(f.id).status == "reopened"
    assert [x.id for x in reloaded.list()][0] == reopened.id          # newest first

    store.unsubscribe(events.clear)                      # unknown callback is a no-op
    bad = lambda f: (_ for _ in ()).throw(RuntimeError("boom"))
    store.subscribe(bad)
    store.close(g.id)                                    # a failing subscriber does not break writes
    assert store.get(g.id).status == "closed"


# ---------------------------------------------------------------- API


@pytest.fixture
def api(topology):
    store = FindingStore(None)
    f = store.upsert(to_finding(load("tick_dns_failed"), load("recs_dns_failed"), topology, SITE, ENGINE))
    return TestClient(create_app(store)), store, f


def test_api_list_get_404(api):
    client, store, f = api
    r = client.get("/findings")
    assert r.status_code == 200 and [x["id"] for x in r.json()] == [f.id]
    assert Finding.model_validate(r.json()[0]) == f
    assert client.get("/findings", params={"status": "closed"}).json() == []
    assert client.get("/findings", params={"status": "bogus"}).status_code == 400
    r = client.get(f"/findings/{f.id}")
    assert r.status_code == 200 and r.json()["root_cause"]["node_id"] == "dns-1"
    r = client.get("/findings/fnd-missing")
    assert r.status_code == 404 and r.json() == {"error": "finding fnd-missing not found"}
    store.close(f.id, "rcp-9")
    assert client.get("/findings", params={"status": "closed"}).json()[0]["receipt_ids"] == ["rcp-9"]


def test_api_stream_sends_finding_on_change(api, topology):
    client, store, f = api
    with client.websocket_connect("/findings/stream") as ws:
        g = store.upsert(to_finding(load("tick_app_degraded"), load("recs_app_degraded"), topology, SITE, ENGINE))
        msg = ws.receive_json()
        assert msg["type"] == "finding" and msg["finding"]["id"] == g.id
        assert Finding.model_validate(msg["finding"]) == g
        store.close(f.id, "rcp-1")
        msg = ws.receive_json()
        assert msg["finding"]["id"] == f.id and msg["finding"]["status"] == "closed"
    assert store._subscribers == []                      # unsubscribed on disconnect


# ---------------------------------------------------------------- emitter


class FakeSable:
    """Stands in for SableClient: returns fixture recs/topology, records calls,
    and replays a list of ws messages to subscribe_ticks."""

    def __init__(self, recs, topo, messages=None, status=None):
        self.recs, self.topo, self.messages = recs, topo, messages or []
        self._status = status or {"engine_loaded": True, "device": "cpu", "mc_samples": 0, "cycle": 4}
        self.calls = []

    def status(self):
        self.calls.append("status"); return dict(self._status)

    def version_info(self):
        return {"title": "SABLE Engine", "version": "1.0"}

    def recommendations(self):
        self.calls.append("recommendations"); return copy.deepcopy(self.recs)

    def topology(self):
        self.calls.append("topology"); return copy.deepcopy(self.topo)

    async def subscribe_ticks(self, on_tick, stop=None, on_message=None):
        for msg in self.messages:
            if msg.get("type") in ("tick", "live_tick"):
                r = on_tick(msg)
                if asyncio.iscoroutine(r):
                    await r
        if stop is not None:
            stop.set()


def test_emitter_on_tick_upserts_and_dedups(scenario):
    topo = load("topology_msp_demo")
    store = FindingStore(None)
    fake = FakeSable(scenario[0]["recs"], topo, status={"device": "cuda"})
    em = FindingEmitter(fake, store, SITE, engine_sha="abc1234")
    assert em.engine_version() == "sable@abc1234 engine/1.0 device/cuda"

    findings = []
    for step in scenario[:3]:
        fake.recs = step["recs"]
        findings.append(em.on_tick(step["tick"]))
    assert len(store) == 1 and len({f.id for f in findings}) == 1
    assert store.list()[0].occurrences == 3
    assert fake.calls.count("topology") == 1             # fetched lazily once
    assert fake.calls.count("recommendations") == 3

    fake.recs = load("recs_healthy")
    assert em.on_tick(load("tick_healthy")) is None and len(store) == 1

    fake.recs = load("recs_app_degraded")
    g = em.poll_once(load("tick_app_degraded"))
    assert g.root_cause.node_id == "app-erp" and len(store) == 2
    assert fake.calls.count("topology") == 2             # poll_once refreshes


def test_emitter_summarize_only_when_condition_changes(scenario):
    calls = []

    def summarize(f):
        calls.append(f.occurrences)
        return f"generated for {f.root_cause.node_id}"

    store = FindingStore(None)
    fake = FakeSable(scenario[0]["recs"], load("topology_msp_demo"))
    em = FindingEmitter(fake, store, SITE, summarize=summarize)
    for step in scenario:                                # cascade grows on ticks 1..4, static on 5
        fake.recs = step["recs"]
        f = em.on_tick(step["tick"])
    assert f.summary == "generated for dns-1" and f.summary_generated is True and f.occurrences == 5
    assert calls == [1, 2, 3, 4]


def test_emitter_run_consumes_websocket_ticks(scenario):
    topo = load("topology_msp_demo")
    messages = [{"type": "status", "cycle": 0}]
    for step in scenario:
        messages.append({"type": "tick", **step["tick"]})
    messages.append({"type": "complete", "cycle": 6})
    fake = FakeSable(scenario[-1]["recs"], topo, messages=messages)
    store = FindingStore(None)
    em = FindingEmitter(fake, store, SITE)

    async def go():
        stop = asyncio.Event()
        await asyncio.wait_for(em.run(stop), timeout=5)

    asyncio.run(go())
    assert em.ticks_seen == 5 and len(store) == 1 and store.list()[0].occurrences == 5


def test_emitter_run_survives_connection_errors():
    class Flaky(FakeSable):
        def __init__(self):
            super().__init__(load("recs_healthy"), load("topology_msp_demo"))
            self.attempts = 0

        async def subscribe_ticks(self, on_tick, stop=None, on_message=None):
            self.attempts += 1
            if self.attempts < 3:
                raise SableError("connection refused")
            stop.set()

    flaky = Flaky()
    em = FindingEmitter(flaky, FindingStore(None), SITE, reconnect_delay=0.01)
    asyncio.run(asyncio.wait_for(em.run(asyncio.Event()), timeout=5))
    assert flaky.attempts == 3 and "connection refused" in em.last_error


# ---------------------------------------------------------------- client (offline)


def test_client_urls_and_errors():
    c = SableClient("http://sable:8080/")
    assert c.base_url == "http://sable:8080" and c.ws_url == "ws://sable:8080/ws"
    assert SableClient("https://sable").ws_url == "wss://sable/ws"
    with pytest.raises(SableError):
        c.live_tick({"gnn": [], "pomdp": {}})           # incomplete live_monitor payload
    with pytest.raises(SableError) as ei:
        SableClient("http://127.0.0.1:9", timeout=0.2).status()   # nothing listens on port 9
    assert ei.value.status_code is None
    c.close()


def test_client_maps_http_errors(monkeypatch):
    import httpx
    c = SableClient("http://sable:8080")
    transport = httpx.MockTransport(lambda req: httpx.Response(400, json={"error": "No scenario loaded"}))
    c._http = httpx.Client(base_url=c.base_url, transport=transport)
    with pytest.raises(SableError) as ei:
        c.tick()
    assert ei.value.status_code == 400 and "No scenario loaded" in str(ei.value)
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"root_cause": None, "actions": []}))
    c._http = httpx.Client(base_url=c.base_url, transport=transport)
    assert c.recommendations() == {"root_cause": None, "actions": []}
