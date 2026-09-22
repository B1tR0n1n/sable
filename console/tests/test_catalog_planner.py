"""Phase 3: the action catalog and the two planners. Ground rule 4 — the LLM
never invents actions: anything not in the catalog is rejected, not repaired."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from console.catalog import SCRIPTS_DIR, Catalog, CatalogError, check_schema
from console.contracts import Confidence, Evidence, Finding, NodeRef, Plan, Reversibility, Step
from console.planner import LLMPlanner, NoTemplate, PlanValidationError, TemplatePlanner, validate_plan
from console.planner.llm import CONSOLE_OWNED_FIELDS, SYSTEM_PROMPT, extract_json_object
from console.planner.templates import TEMPLATES
from console.topology import Topology

T0 = datetime(2026, 9, 22, 14, 0, 0, tzinfo=timezone.utc)

# The lab (Phase 8) as /api/topology will return it: {name, nodes[], edges[]}.
# app needs dns and the db proxy; the proxy fronts db (and its replica, redundantly);
# db replicates to db-replica; prometheus scrapes everything.
LAB_TOPOLOGY = {
    "name": "lab",
    "nodes": [
        {"id": "dns", "type": "DNS_SERVER", "label": "dnsmasq"},
        {"id": "app", "type": "APPLICATION_SERVICE", "label": "App"},
        {"id": "db", "type": "SERVER_VIRTUAL", "label": "DB primary"},
        {"id": "db-replica", "type": "SERVER_VIRTUAL", "label": "DB replica"},
        {"id": "proxy", "type": "LOAD_BALANCER", "label": "DB proxy"},
        {"id": "prometheus", "type": "MONITORING_SERVER", "label": "Prometheus"},
        {"id": "nas", "type": "STORAGE_ARRAY", "label": "NAS"},
    ],
    "edges": [
        {"source": "app", "target": "dns", "type": "DNS_DEPENDENCY", "criticality": "HARD"},
        {"source": "app", "target": "proxy", "type": "SERVICE_DEPENDENCY", "criticality": "HARD"},
        {"source": "proxy", "target": "db", "type": "SERVICE_DEPENDENCY", "criticality": "HARD"},
        {"source": "proxy", "target": "db-replica", "type": "SERVICE_DEPENDENCY", "criticality": "REDUNDANT"},
        {"source": "db", "target": "db-replica", "type": "REPLICATION_DEPENDENCY", "criticality": "SOFT"},
        {"source": "db", "target": "nas", "type": "STORAGE_DEPENDENCY", "criticality": "HARD"},
        *[{"source": "prometheus", "target": t, "type": "MONITORING_DEPENDENCY", "criticality": "SOFT"}
          for t in ("dns", "app", "db", "db-replica", "proxy")],
    ],
}
SERVICE_MAP = {"dns": "dnsmasq", "app": "app", "db": "postgres", "db-replica": "postgres-replica",
               "proxy": "haproxy", "prometheus": "prometheus"}
CONTEXT = {"lab_dir": "/srv/lab", "lab_compose": "/srv/lab/docker-compose.yml"}


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return Catalog.load()


@pytest.fixture(scope="module")
def topology() -> Topology:
    return Topology.from_api(LAB_TOPOLOGY)


def make_finding(node_id: str, state: str, topology: Topology, affected: list[str] = ()) -> Finding:
    return Finding(
        site_id="lab-1", detection_mode="live_feed" if state != "unreachable" else "unmonitored_gap",
        root_cause=NodeRef(node_id=node_id, component_type=topology.component_type(node_id), state=state),
        affected_nodes=[NodeRef(node_id=a, component_type=topology.component_type(a), state="degraded")
                        for a in affected],
        evidence=[Evidence(metric="up", value=0.0, unit="bool", threshold=1.0, timestamp=T0)],
        confidence=Confidence(score=0.9, method="engine_native"), severity="high",
        engine_version="sable@test",
    )


def completion(text: str, **over) -> dict:
    """A result shaped like OverlordClient.complete(): the hashes OVERLORD audits."""
    system, prompt = over.pop("system", ""), over.pop("prompt", "")
    res = {"text": text, "provider": "scripted", "model": "scripted-1", "stop": "end_turn",
           "usage": {"input_tokens": 1, "output_tokens": 1}, "refusal": None,
           "prompt_sha256": hashlib.sha256((system + "\n" + prompt).encode()).hexdigest(),
           "output_sha256": hashlib.sha256(text.encode()).hexdigest()}
    res.update(over)
    return res


def scripted(text_or_fn, **over):
    """complete_fn that ignores the prompt and answers with a fixed text (or a fn of the prompt)."""
    calls = []

    def complete_fn(prompt: str, system: str, purpose: str) -> dict:
        calls.append({"prompt": prompt, "system": system, "purpose": purpose})
        text = text_or_fn(prompt) if callable(text_or_fn) else text_or_fn
        return completion(text, system=system, prompt=prompt, **over)
    complete_fn.calls = calls
    return complete_fn


def good_llm_plan(finding: Finding, **step_over) -> dict:
    step = {"action_id": "restart_service", "target_node": "dns", "params": {"service": "dnsmasq"},
            "reversibility": "compensable",
            "compensation": {"action_id": "restart_service", "params": {"service": "dnsmasq"}},
            "precondition": "service_exists", "timeout_s": 60}
    step.update(step_over)
    return {"finding_id": finding.id, "steps": [step], "verification": {"predicate": "node_healthy", "window_s": 30}}


# ---------------------------------------------------------------- catalog


def test_catalog_loads_and_every_compensation_names_an_existing_action(catalog):
    assert set(catalog.actions) == {"restart_service", "set_config_value", "failover_to_replica",
                                    "failback_to_primary", "clear_dns_cache"}
    for a in catalog.actions.values():
        assert a.preconditions and all(c in catalog.checks for c in a.preconditions)
        assert a.verification in catalog.predicates
        if a.compensation is not None:
            assert a.compensation.action_id in catalog.actions
            assert Reversibility(a.reversibility) == Reversibility.compensable
        else:
            assert Reversibility(a.reversibility) != Reversibility.compensable
    assert catalog.context_keys == ["lab_dir", "lab_compose"]
    assert (SCRIPTS_DIR / "set_config_value.py").exists()


def test_actions_for_excludes_compensation_only(catalog):
    ids = {a.action_id for a in catalog.actions_for("SERVER_VIRTUAL")}
    assert ids == {"restart_service", "failover_to_replica"}
    assert "failback_to_primary" not in ids and catalog.get("failback_to_primary").compensation_only
    assert {a.action_id for a in catalog.actions_for("DNS_SERVER")} == {"restart_service", "set_config_value", "clear_dns_cache"}
    assert catalog.actions_for("STORAGE_ARRAY") == [catalog.get("failover_to_replica")]
    assert catalog.actions_for("CORE_SWITCH") == []


def test_render_substitutes_params_and_context(catalog):
    r = catalog.render("restart_service", {"service": "dnsmasq"}, CONTEXT)
    assert r["argv"] == ["docker", "compose", "-f", "/srv/lab/docker-compose.yml", "restart", "dnsmasq"]
    assert r["then"] is None and r["target_dir"] == "/srv/lab"
    assert r["grants"] == {"jail": False, "net": "none", "net_allow": [], "timeout_s": 60}
    r = catalog.render("set_config_value", {"file": "app/app.env", "key": "LOG_LEVEL", "value": "info", "service": "app"}, CONTEXT)
    assert r["argv"] == ["python3", str(SCRIPTS_DIR / "set_config_value.py"), "/srv/lab/app/app.env", "LOG_LEVEL", "info"]
    assert r["then"][-2:] == ["restart", "app"]
    chk = catalog.render_check("service_exists", "restart_service", {"service": "dnsmasq"}, CONTEXT)
    assert chk == {"argv": ["docker", "compose", "-f", "/srv/lab/docker-compose.yml", "ps", "--services", "--all"],
                   "expect": {"exit_code": 0, "stdout_contains": "dnsmasq"}}


def test_render_rejects_unknown_placeholder_and_missing_context(catalog, tmp_path):
    with pytest.raises(CatalogError, match="missing \\['lab_compose'\\]"):
        catalog.render("restart_service", {"service": "dnsmasq"}, {"lab_dir": "/srv/lab"})
    # a catalog whose binding names a placeholder no param or context can fill does not even load
    import yaml
    data = yaml.safe_load(Path(catalog.path).read_text())
    data["actions"]["restart_service"]["executor"]["argv"].append("{bogus}")
    bad = tmp_path / "actions.yaml"
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(CatalogError, match="bogus"):
        Catalog.load(bad)
    with pytest.raises(CatalogError, match="unknown action"):
        catalog.render("rm_rf_everything", {}, CONTEXT)


def test_validate_params_rejects_missing_extra_and_malformed(catalog):
    catalog.validate_params("restart_service", {"service": "dnsmasq"})
    with pytest.raises(CatalogError, match="missing required property 'service'"):
        catalog.validate_params("restart_service", {})
    with pytest.raises(CatalogError, match="unexpected property 'force'"):
        catalog.validate_params("restart_service", {"service": "dnsmasq", "force": True})
    with pytest.raises(CatalogError, match="pattern"):
        catalog.validate_params("restart_service", {"service": "dnsmasq; rm -rf /"})
    with pytest.raises(CatalogError, match="expected string"):
        catalog.validate_params("restart_service", {"service": 1})
    with pytest.raises(CatalogError, match="pattern"):        # no escaping the lab dir
        catalog.validate_params("set_config_value", {"file": "../etc/passwd", "key": "k", "value": "v", "service": "app"})
    # the mini schema checker itself
    assert check_schema(True, {"type": "integer"}) and not check_schema(3, {"type": "integer", "minimum": 1})
    assert check_schema("x", {"enum": ["a", "b"]}) and not check_schema("a", {"enum": ["a", "b"]})


def test_compensation_for_builds_from_params_from(catalog):
    s = Step(action_id="failover_to_replica", target_node="db",
             params={"primary": "postgres", "replica": "postgres-replica", "proxy": "haproxy"},
             reversibility="compensable", precondition="replica_running", timeout_s=120,
             compensation={"action_id": "x", "params": {}})
    c = catalog.compensation_for(s)
    assert c.action_id == "failback_to_primary" and c.params == s.params
    s2 = Step(action_id="clear_dns_cache", target_node="dns", params={"service": "dnsmasq"},
              reversibility="reversible", precondition="service_running", timeout_s=30)
    assert catalog.compensation_for(s2) is None


def test_shipped_set_config_value_script_is_idempotent(tmp_path):
    f = tmp_path / "app.env"
    f.write_text("A=1\nLOG_LEVEL=debug\n")
    script = str(SCRIPTS_DIR / "set_config_value.py")
    assert subprocess.run([sys.executable, script, str(f), "LOG_LEVEL", "info"]).returncode == 0
    assert f.read_text() == "A=1\nLOG_LEVEL=info\n"
    assert subprocess.run([sys.executable, script, str(f), "LOG_LEVEL", "info"]).returncode == 0
    assert subprocess.run([sys.executable, script, str(f), "NEW", "x"]).returncode == 0
    assert f.read_text() == "A=1\nLOG_LEVEL=info\nNEW=x\n"
    assert subprocess.run([sys.executable, script, str(f), "bad key", "x"]).returncode == 2


# ---------------------------------------------------------------- template planner


LAB_SCENARIOS = [
    # (node, state, expected action, expected reversibility, expected compensation action)
    ("dns", "failed", "restart_service", Reversibility.compensable, "restart_service"),
    ("dns", "unreachable", "restart_service", Reversibility.compensable, "restart_service"),
    ("dns", "degraded", "clear_dns_cache", Reversibility.reversible, None),
    ("dns", "oscillating", "clear_dns_cache", Reversibility.reversible, None),
    ("app", "failed", "restart_service", Reversibility.compensable, "restart_service"),
    ("app", "degraded", "restart_service", Reversibility.compensable, "restart_service"),
    ("db", "failed", "restart_service", Reversibility.compensable, "restart_service"),
    ("db", "degraded", "restart_service", Reversibility.compensable, "restart_service"),
    ("proxy", "failed", "restart_service", Reversibility.compensable, "restart_service"),
    ("proxy", "degraded", "restart_service", Reversibility.compensable, "restart_service"),
    ("prometheus", "failed", "restart_service", Reversibility.compensable, "restart_service"),
]


@pytest.mark.parametrize("node,state,action,rev,comp", LAB_SCENARIOS)
def test_every_lab_fault_scenario_yields_a_valid_plan(catalog, topology, node, state, action, rev, comp):
    finding = make_finding(node, state, topology, affected=topology.dependents(node))
    plan = TemplatePlanner(catalog, topology, SERVICE_MAP).plan(finding)
    assert isinstance(plan, Plan) and plan.finding_id == finding.id
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.action_id == action and step.target_node == node
    assert plan.reversibility == rev and Reversibility(step.reversibility) == rev
    assert (step.compensation.action_id if step.compensation else None) == comp
    assert plan.blast_radius.nodes == topology.blast_radius(node)
    assert plan.blast_radius.count == len(topology.blast_radius(node))
    assert plan.verification.predicate == catalog.get(action).verification and plan.verification.window_s == 30
    assert plan.planner.kind == "template" and plan.planner.template_id
    assert plan.gate is None                                    # the gate is Phase 6's
    # it passes the same gate a model's plan would, unchanged
    again = validate_plan(plan.model_dump(mode="json"), catalog, finding, topology)
    assert again == plan


def test_db_failover_binding_resolves_from_the_topology(catalog, topology):
    """failover_to_replica is not a template row (a failover leaves the primary
    down, so node_healthy could never pass) but its binding must still resolve
    for the LLM planner and for an operator: primary/replica/proxy from the
    REPLICATION_DEPENDENCY edge and the LOAD_BALANCER that depends on it."""
    from console.planner.templates import _DB_FAILOVER
    planner = TemplatePlanner(catalog, topology, SERVICE_MAP)
    t, params = planner._pick(_DB_FAILOVER, "db")
    assert t.template_id == "db_failed_failover"
    assert params == {"primary": "postgres", "replica": "postgres-replica", "proxy": "haproxy"}
    comp = catalog.compensation_for_action("failover_to_replica", params)
    assert comp.action_id == "failback_to_primary" and comp.params == params
    # and the same row falls back to a restart when the topology has no replica
    bare = Topology.from_api({"name": "t", "nodes": list(topology.nodes.values()),
                              "edges": [e for e in topology.edges if e["type"] != "REPLICATION_DEPENDENCY"]})
    t2, params2 = TemplatePlanner(catalog, bare, SERVICE_MAP)._pick(_DB_FAILOVER, "db")
    assert t2.template_id == "db_failed_restart" and params2 == {"service": "postgres"}


def test_degraded_app_restores_golden_config_and_falls_back_without_it(catalog, topology):
    finding = make_finding("app", "degraded", topology, affected=[])
    golden = {"app": {"file": "config/app.conf", "key": "db_host", "value": "db.lab"}}
    plan = TemplatePlanner(catalog, topology, SERVICE_MAP, golden=golden).plan(finding)
    step = plan.steps[0]
    assert step.action_id == "set_config_value" and plan.planner.template_id == "app_degraded_restore_config"
    assert step.params == {"file": "config/app.conf", "key": "db_host", "value": "db.lab", "service": SERVICE_MAP.get("app", "app")}
    assert plan.reversibility == Reversibility.reversible and step.compensation is None
    # no golden entry → restart; golden present but the action disabled → restart too
    assert TemplatePlanner(catalog, topology, SERVICE_MAP).plan(finding).steps[0].action_id == "restart_service"
    fb = TemplatePlanner(catalog, topology, SERVICE_MAP, golden=golden, disabled=["set_config_value"]).plan(finding)
    assert fb.steps[0].action_id == "restart_service" and fb.planner.template_id == "app_unhealthy_restart"
    # a disabled action with no fallback is NoTemplate, never a silent substitute
    with pytest.raises(NoTemplate):
        TemplatePlanner(catalog, topology, SERVICE_MAP, disabled=["restart_service"]).plan(make_finding("db", "failed", topology, affected=[]))


def test_db_without_replica_falls_back_to_restart(catalog):
    payload = {**LAB_TOPOLOGY, "edges": [e for e in LAB_TOPOLOGY["edges"] if e["type"] != "REPLICATION_DEPENDENCY"]}
    topo = Topology.from_api(payload)
    plan = TemplatePlanner(catalog, topo, SERVICE_MAP).plan(make_finding("db", "failed", topo))
    assert plan.steps[0].action_id == "restart_service" and plan.steps[0].params == {"service": "postgres"}
    assert plan.planner.template_id == "db_failed_restart"


def test_default_service_map_is_identity(catalog, topology):
    plan = TemplatePlanner(catalog, topology).plan(make_finding("dns", "failed", topology))
    assert plan.steps[0].params == {"service": "dns"}


def test_no_template_raises(catalog, topology):
    planner = TemplatePlanner(catalog, topology, SERVICE_MAP)
    with pytest.raises(NoTemplate, match="STORAGE_ARRAY"):
        planner.plan(make_finding("nas", "unreachable", topology))
    with pytest.raises(NoTemplate, match="not in the topology"):
        planner.plan(make_finding("ghost", "failed", Topology.from_api({**LAB_TOPOLOGY, "nodes": LAB_TOPOLOGY["nodes"] + [{"id": "ghost", "type": "DNS_SERVER"}]})))
    with pytest.raises(NoTemplate, match="no template for"):
        planner.plan(make_finding("dns", "healthy", topology))
    assert all(catalog.has_action(t.action_id) and not catalog.get(t.action_id).compensation_only
               for t in TEMPLATES.values())


# ---------------------------------------------------------------- validate_plan


def test_validate_plan_never_modifies_its_input(catalog, topology):
    finding = make_finding("dns", "failed", topology)
    d = good_llm_plan(finding)
    d["blast_radius"] = {"nodes": topology.blast_radius("dns"), "count": len(topology.blast_radius("dns"))}
    before = json.dumps(d, sort_keys=True)
    plan = validate_plan(d, catalog, finding, topology)
    assert json.dumps(d, sort_keys=True) == before and plan.steps[0].action_id == "restart_service"
    d["steps"][0]["action_id"] = "rm_rf_everything"
    with pytest.raises(PlanValidationError):
        validate_plan(d, catalog, finding, topology)
    assert d["steps"][0]["action_id"] == "rm_rf_everything"     # not repaired


# ---------------------------------------------------------------- LLM planner


def llm(catalog, topology, complete_fn):
    return LLMPlanner(catalog, topology, complete_fn, model="scripted-1", service_map=SERVICE_MAP)


BAD_LLM_OUTPUTS = [
    ("unknown action", lambda f: json.dumps(good_llm_plan(f, action_id="rm_rf_everything",
                                                          compensation={"action_id": "rm_rf_everything", "params": {"service": "dnsmasq"}})),
     "unknown action 'rm_rf_everything'"),
    ("unknown precondition", lambda f: json.dumps(good_llm_plan(f, precondition="sudo_ok")), "unknown precondition check 'sudo_ok'"),
    ("precondition not declared for the action", lambda f: json.dumps(good_llm_plan(f, precondition="none")),
     "precondition 'none' is not one the catalog declares for restart_service"),
    ("wrong component type", lambda f: json.dumps(good_llm_plan(f, target_node="nas")), "does not apply to 'nas' (STORAGE_ARRAY)"),
    ("unknown target node", lambda f: json.dumps(good_llm_plan(f, target_node="mainframe")), "'mainframe' is not in the topology"),
    ("extra params", lambda f: json.dumps(good_llm_plan(f, params={"service": "dnsmasq", "force": True})), "unexpected property 'force'"),
    ("missing params", lambda f: json.dumps(good_llm_plan(f, params={})), "missing required property 'service'"),
    ("wrong reversibility", lambda f: json.dumps(good_llm_plan(f, reversibility="reversible", compensation=None)),
     "reversibility 'reversible' does not match the catalog's 'compensable'"),
    ("wrong compensation", lambda f: json.dumps(good_llm_plan(f, compensation={"action_id": "restart_service", "params": {"service": "app"}})),
     "compensation must be exactly the catalog's"),
    ("timeout over ceiling", lambda f: json.dumps(good_llm_plan(f, timeout_s=3600)), "exceeds the catalog ceiling 60"),
    ("compensation-only action", lambda f: json.dumps(good_llm_plan(
        f, action_id="failback_to_primary", target_node="db", params={"primary": "postgres", "replica": "postgres-replica", "proxy": "haproxy"},
        precondition="primary_running", timeout_s=120,
        compensation={"action_id": "failover_to_replica", "params": {"primary": "postgres", "replica": "postgres-replica", "proxy": "haproxy"}})),
     "compensation-only"),
    ("unknown verification predicate", lambda f: json.dumps({**good_llm_plan(f), "verification": {"predicate": "looks_fine", "window_s": 30}}),
     "unknown predicate 'looks_fine'"),
    ("verification not declared by the action", lambda f: json.dumps({**good_llm_plan(f), "verification": {"predicate": "service_running", "window_s": 30}}),
     "not one the plan's actions declare"),
    ("extra top-level field", lambda f: json.dumps({**good_llm_plan(f), "shell": "reboot"}), "shell: Extra inputs are not permitted"),
    ("extra step field", lambda f: json.dumps(good_llm_plan(f, argv=["reboot"])), "argv: Extra inputs are not permitted"),
    ("wrong finding id", lambda f: json.dumps({**good_llm_plan(f), "finding_id": "fnd-other"}), "is not the finding being planned for"),
    ("no steps", lambda f: '{"steps": []}', "no steps"),
    ("non-JSON", lambda f: "I would restart dnsmasq, then check.", "no JSON object"),
    ("empty text", lambda f: "", "no text"),
]


@pytest.mark.parametrize("label,text_fn,reason", BAD_LLM_OUTPUTS, ids=[b[0] for b in BAD_LLM_OUTPUTS])
def test_llm_output_is_rejected_with_the_reason(catalog, topology, label, text_fn, reason):
    finding = make_finding("dns", "failed", topology, affected=["app"])
    with pytest.raises(PlanValidationError) as ei:
        llm(catalog, topology, scripted(text_fn(finding))).plan(finding)
    assert any(reason in r for r in ei.value.reasons), f"{label}: {ei.value.reasons}"
    assert reason in str(ei.value)


def test_llm_refusal_is_rejected(catalog, topology):
    finding = make_finding("dns", "failed", topology)
    fn = scripted("", stop="refusal", refusal={"type": "refusal", "category": "policy"})
    with pytest.raises(PlanValidationError, match="model refused"):
        llm(catalog, topology, fn).plan(finding)
    fn = scripted(json.dumps(good_llm_plan(finding)), output_sha256="0" * 64)
    with pytest.raises(PlanValidationError, match="output_sha256 does not match"):
        llm(catalog, topology, fn).plan(finding)
    assert len(fn.calls) == 1                                   # one call, no silent retry


def test_valid_llm_json_passes_with_provenance_and_console_owned_fields(catalog, topology):
    finding = make_finding("dns", "failed", topology, affected=["app"])
    proposed = good_llm_plan(finding)
    # the model tries to own what the console owns: all of it is ignored
    proposed.update({"id": "pln-fromthemodel", "created_at": "2001-01-01T00:00:00Z",
                     "gate": {"decision": "auto", "rule_id": "made-up", "reason": "trust me"},
                     "planner": {"kind": "template", "template_id": "forged"},
                     "blast_radius": {"nodes": ["dns"], "count": 1}})
    text = "Here is the plan:\n```json\n" + json.dumps(proposed) + "\n```\nLet me know."
    fn = scripted(text)
    plan = llm(catalog, topology, fn).plan(finding)
    assert plan.steps[0].action_id == "restart_service" and plan.steps[0].params == {"service": "dnsmasq"}
    assert plan.id != "pln-fromthemodel" and plan.id.startswith("pln-")
    assert plan.created_at.year == 2026 and plan.gate is None
    assert plan.blast_radius.nodes == topology.blast_radius("dns") == ["dns", "app", "prometheus"]
    assert plan.reversibility == Reversibility.compensable
    call = fn.calls[0]
    assert call["system"] == SYSTEM_PROMPT and call["purpose"] == f"plan:{finding.id}"
    assert plan.planner.kind == "llm" and plan.planner.provider == "scripted" and plan.planner.model == "scripted-1"
    assert plan.planner.prompt_sha256 == hashlib.sha256((SYSTEM_PROMPT + "\n" + call["prompt"]).encode()).hexdigest()
    assert plan.planner.output_sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert plan.planner.template_id is None
    assert set(CONSOLE_OWNED_FIELDS) == {"id", "created_at", "gate", "planner", "blast_radius"}


def test_llm_prompt_offers_only_applicable_catalog_actions(catalog, topology):
    finding = make_finding("db", "failed", topology, affected=["proxy", "app"])
    seen = {}
    fn = scripted(lambda p: seen.setdefault("prompt", p) and json.dumps({
        "finding_id": finding.id,
        "steps": [{"action_id": "failover_to_replica", "target_node": "db",
                   "params": {"primary": "postgres", "replica": "postgres-replica", "proxy": "haproxy"},
                   "reversibility": "compensable",
                   "compensation": {"action_id": "failback_to_primary",
                                    "params": {"primary": "postgres", "replica": "postgres-replica", "proxy": "haproxy"}},
                   "precondition": "replica_running", "timeout_s": 120}],
        "verification": {"predicate": "node_healthy", "window_s": 30}}))
    plan = llm(catalog, topology, fn).plan(finding)
    assert plan.steps[0].compensation.action_id == "failback_to_primary"
    assert plan.blast_radius.nodes == topology.blast_radius("db")
    prompt = seen["prompt"]
    offered = json.loads(prompt.split("ACTION CATALOG — the only actions you may use:\n", 1)[1].split("\n\nREQUIRED OUTPUT")[0])
    ids = {a["action_id"] for a in offered}
    assert "failback_to_primary" not in ids and {"failover_to_replica", "restart_service", "set_config_value"} <= ids
    assert '"service": "postgres-replica"' in prompt and finding.id in prompt
    assert "SIGHUP" not in prompt or "clear_dns_cache" in ids      # no executor bindings leak; descriptions only via listed actions


def test_extract_json_object_takes_the_first_object():
    assert extract_json_object('junk {"a": 1} {"b": 2}') == {"a": 1}
    assert extract_json_object("```json\n{\"a\": {\"b\": [1, 2]}}\n```") == {"a": {"b": [1, 2]}}
    assert extract_json_object("[1, 2, 3]") is None
    assert extract_json_object("{not json} {\"ok\": true}") == {"ok": True}
    assert extract_json_object("") is None
