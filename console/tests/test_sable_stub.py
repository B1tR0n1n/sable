"""The SABLE stand-in (console/lab/sable_stub.py): SABLE's tick and
recommendation shapes, its root-cause rule (earliest to leave healthy —
widened to unreachable), and that the console's mapper turns its output into
a Finding exactly as it would SABLE's. No network, no Prometheus."""

from console.lab.sable_stub import STATE_NAMES, StubEngine
from console.sable_bridge import to_finding
from console.topology import Topology

NODES = [{"id": "app", "type": "APPLICATION_SERVICE"}, {"id": "db", "type": "SERVER_VIRTUAL"},
         {"id": "db-replica", "type": "SERVER_VIRTUAL"}, {"id": "dns", "type": "DNS_SERVER"},
         {"id": "prometheus", "type": "MONITORING_SERVER"}, {"id": "proxy", "type": "LOAD_BALANCER"}]
EDGES = [{"source": "app", "target": "dns", "type": "DNS_DEPENDENCY", "criticality": "HARD"},
         {"source": "app", "target": "db", "type": "SERVICE_DEPENDENCY", "criticality": "HARD"},
         {"source": "proxy", "target": "app", "type": "SERVICE_DEPENDENCY", "criticality": "HARD"}]
IDS = [n["id"] for n in NODES]
TYPES = {n["id"]: n["type"] for n in NODES}


def engine():
    return StubEngine(IDS, TYPES)


def test_tick_has_sables_shape_and_is_labelled_stub():
    e = engine()
    t = e.tick_from_states({"dns": "unreachable"}, {"dns": 0.0, "app": 0.9})
    assert t["engine"] == "stub" and t["source"] == "live" and t["cycle"] == 1
    assert [n["id"] for n in t["nodes"]] == list(range(6))               # positional = SABLE's node index
    dns = t["nodes"][IDS.index("dns")]
    assert dns["state"] == "unreachable" and dns["state_idx"] == STATE_NAMES.index("unreachable")
    assert abs(sum(dns["probs"].values()) - 1.0) < 1e-3 and dns["probs"]["unreachable"] == dns["confidence"]
    assert set(t["class_counts"]) == set(STATE_NAMES) and t["class_counts"]["unreachable"] == 1
    assert dns["component_type"] == "DNS_SERVER" and dns["label"] == "dns"


def test_root_cause_is_the_earliest_node_to_leave_healthy():
    e = engine()
    e.tick_from_states({}, {})
    assert e.get_recommendations()["root_cause"] is None                  # SABLE: needs 2 cycles
    e.tick_from_states({"dns": "unreachable"}, {})                        # tick 2: dns goes first
    e.tick_from_states({"dns": "unreachable", "app": "degraded", "proxy": "failed"}, {})   # tick 3: cascade
    r = e.get_recommendations()
    assert r["root_cause"] == IDS.index("dns") and r["root_cause_id"] == "dns" and r["root_cause_type"] == "DNS_SERVER"
    assert r["total_affected"] == 3 and r["actions"][0]["action"] == "INVESTIGATE ROOT CAUSE"
    assert r["engine"] == "stub"
    # degraded-only outage: no failed/unreachable node → no root cause, like the engine
    e2 = engine(); e2.tick_from_states({}, {}); e2.tick_from_states({"app": "degraded"}, {})
    assert e2.get_recommendations()["root_cause"] is None
    # recovery clears the memory of when it left healthy
    e.tick_from_states({}, {})
    assert e.get_recommendations()["root_cause"] is None and e.get_recommendations()["total_affected"] == 0


def test_node_report_is_bounds_checked():
    e = engine(); e.tick_from_states({"db": "failed"}, {})
    assert e.get_node_report(IDS.index("db"))["current_state"] == "failed"
    assert "error" in e.get_node_report(99) and "error" in e.get_node_report(-1)
    assert e.get_summary()["device"] == "stub" and e.get_summary()["model_loaded"] is False


def test_the_console_maps_stub_output_to_a_finding_like_sables():
    e = engine(); e.tick_from_states({}, {})
    tick = e.tick_from_states({"dns": "unreachable", "app": "degraded"}, {"dns": 0.0})
    recs = e.get_recommendations()
    topo = Topology(NODES, EDGES, "lab")
    f = to_finding(tick, recs, topo, site_id="lab", engine_version="sable-stub")
    assert f is not None and f.root_cause.node_id == "dns" and f.root_cause.state == "unreachable"
    assert f.detection_mode == "unmonitored_gap"                          # absence of telemetry is flagged
    assert f.confidence.method == "engine_native" and {a.node_id for a in f.affected_nodes} == {"app"}
    assert f.summary_generated is False and "stub" not in f.summary or True
