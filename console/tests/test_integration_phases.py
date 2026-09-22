"""Cross-phase seams, on the real modules:

  - the mapper flags an UPSTREAM unreachable node as an unmonitored gap
    without overriding SABLE's (failed) root cause
  - the executor drives a plan the real TemplatePlanner produced from the
    real Catalog, and the command lines that reach OVERLORD are the
    catalog's bindings, rendered
"""

import json
from pathlib import Path

from console.catalog import Catalog
from console.contracts import Approval, Finding
from console.executor import Executor
from console.planner import TemplatePlanner
from console.sable_bridge import to_finding
from console.topology import Topology
from console.tests.test_executor import FakeClient, T0

FIX = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIX / name).read_text())


def test_upstream_unreachable_node_marks_the_finding_as_a_gap():
    topo = Topology.from_api(_load("topology_msp_demo.json"))
    tick, recs = _load("tick_san_unreachable.json"), _load("recs_san_unreachable.json")
    states = {topo.id_at(n["id"]): n["state"] for n in tick["nodes"]}
    # SABLE names the first FAILED node as root; pick one that depends on the unreachable SAN
    failed = [n for n, s in states.items() if s == "failed"]
    root = next(n for n in failed if "san-1" in topo.dependencies(n))
    recs = {**recs, "root_cause": topo.index_of(root), "root_cause_id": root}
    f = to_finding(tick, recs, topo, site_id="lab", engine_version="sable@test")
    assert f.root_cause.node_id == root and f.root_cause.state == "failed"   # SABLE's diagnosis kept
    assert f.detection_mode == "unmonitored_gap"                             # ...but the gap is flagged
    # and a failed root with NO unreachable upstream stays live_feed
    tick2, recs2 = _load("tick_dns_failed.json"), _load("recs_dns_failed.json")
    assert to_finding(tick2, recs2, topo, site_id="lab", engine_version="x").detection_mode == "live_feed"


def lab_topology():
    return Topology.from_api({"name": "lab", "nodes": [
        {"id": "dns", "type": "DNS_SERVER", "label": "DNS"},
        {"id": "app", "type": "APPLICATION_SERVICE", "label": "App"},
        {"id": "db", "type": "SERVER_VIRTUAL", "label": "DB"},
        {"id": "db-replica", "type": "SERVER_VIRTUAL", "label": "DB replica"},
        {"id": "proxy", "type": "LOAD_BALANCER", "label": "Proxy"},
        {"id": "prometheus", "type": "MONITORING_SERVER", "label": "Prometheus"},
    ], "edges": [
        {"source": "app", "target": "dns", "type": "DNS_DEPENDENCY", "criticality": "HARD"},
        {"source": "app", "target": "db", "type": "SERVICE_DEPENDENCY", "criticality": "HARD"},
        {"source": "db", "target": "db-replica", "type": "REPLICATION_DEPENDENCY", "criticality": "SOFT"},
        {"source": "proxy", "target": "app", "type": "SERVICE_DEPENDENCY", "criticality": "HARD"},
        {"source": "proxy", "target": "db", "type": "SERVICE_DEPENDENCY", "criticality": "HARD"},
        {"source": "prometheus", "target": "app", "type": "MONITORING_DEPENDENCY", "criticality": "SOFT"},
    ]})


def lab_finding(node, ctype, state):
    from console.contracts import Confidence, NodeRef
    return Finding(id="fnd-lab", site_id="lab", detection_mode="live_feed",
                   root_cause=NodeRef(node_id=node, component_type=ctype, state=state),
                   confidence=Confidence(score=0.9, method="engine_native"), severity="high",
                   engine_version="sable@test")


def test_executor_runs_a_real_template_plan_with_the_real_catalog_bindings():
    cat, topo = Catalog.load(), lab_topology()
    ctx = {"lab_dir": "/lab", "lab_compose": "/lab/docker-compose.yml"}
    plan = TemplatePlanner(cat, topo).plan(lab_finding("dns", "DNS_SERVER", "failed"))
    assert [s.action_id for s in plan.steps] == ["restart_service"]
    c = FakeClient(stdout="dns\napp\ndb\ndb-replica\nproxy\nprometheus\n")
    r = Executor(c, cat, ctx).execute(plan, lab_finding("dns", "DNS_SERVER", "failed"),
                                      [Approval(actor="op", decision="approve", timestamp=T0)])
    assert [s.status for s in r.steps] == ["ok"], r.steps
    argv = [e["argv"] for e in c.execs]
    assert ["docker", "compose", "-f", "/lab/docker-compose.yml", "restart", "dns"] in argv
    assert argv[0][:2] == ["docker", "compose"] and "-f" in argv[0]          # the precondition check, rendered
    assert c.opened[0]["jail"] is False and c.opened[0]["net"] == "none" and c.opened[0]["timeout"] == 60
    assert c.opened[0]["target"] == "/lab"


def test_executor_compensates_a_real_failover_with_the_catalog_failback():
    cat, topo = Catalog.load(), lab_topology()
    ctx = {"lab_dir": "/lab", "lab_compose": "/lab/docker-compose.yml"}
    from console.planner import validate_plan
    from console.planner.templates import _DB_FAILOVER
    f = lab_finding("db", "SERVER_VIRTUAL", "failed")
    planner = TemplatePlanner(cat, topo)
    _t, params = planner._pick(_DB_FAILOVER, "db")             # the failover binding, resolved from the topology
    base = planner.plan(f)
    d = base.model_dump(mode="json")
    d["steps"] = [{**d["steps"][0], "action_id": "failover_to_replica", "params": params, "reversibility": "compensable",
                   "compensation": cat.compensation_for_action("failover_to_replica", params).model_dump(),
                   "precondition": cat.get("failover_to_replica").preconditions[0],
                   "timeout_s": cat.get("failover_to_replica").executor.grants.timeout_s}]
    for k in ("id", "created_at", "gate", "planner"):
        d.pop(k, None)                                           # blast_radius stays: same target, same radius
    plan = validate_plan(d, cat, f, topo)
    assert plan.steps[0].action_id == "failover_to_replica"
    assert plan.steps[0].compensation.action_id == "failback_to_primary"
    # make the plan two steps so a later failure forces the failback: append a restart that fails
    from console.contracts import Compensation, Step
    plan2 = plan.model_copy(update={"steps": plan.steps + [Step(
        step_id="s2", action_id="restart_service", target_node="app", params={"service": "app"},
        reversibility="compensable", compensation=Compensation(action_id="restart_service", params={"service": "app"}),
        precondition="service_exists", timeout_s=60)]})
    c = FakeClient(fail=["restart app"], stdout="dns\napp\ndb\ndb-replica\nproxy\nprometheus\n")
    r = Executor(c, cat, ctx).execute(plan2, lab_finding("db", "SERVER_VIRTUAL", "failed"),
                                      [Approval(actor="op", decision="approve", timestamp=T0)])
    assert [s.status for s in r.steps] == ["ok", "failed"]
    assert r.rollback.performed and [s.action_id for s in r.rollback.steps] == ["failback_to_primary"]
    comp = [e for e in c.execs if e["label"].endswith("/compensate")][0]
    assert "UPSTREAM_HOST" in " ".join(comp["argv"]) and "db" in comp["argv"][-1:] + comp["argv"]
