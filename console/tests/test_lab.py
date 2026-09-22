"""Static validation of console/lab (Phase 8): the compose stack, the SABLE topology, the
Prometheus mapping, the fault scripts and the e2e runner agree with each other, without
Docker. When the compose CLI is present (it renders offline) the compose file must also
pass `docker compose config`."""
from __future__ import annotations

import importlib
import importlib.util
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "console" / "lab"
COMPOSE_FILE = LAB / "docker-compose.yml"
TOPOLOGY_FILE = LAB / "topology.yaml"
SABLE_PROM_FILE = LAB / "sable_prometheus.yaml"
NON_TOPOLOGY_SERVICES = {"blackbox"}          # compose services that are deliberately not nodes


def _load_base_module():
    """adapters/base.py without adapters/__init__.py (which imports numpy)."""
    spec = importlib.util.spec_from_file_location("sable_adapters_base", ROOT / "adapters" / "base.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                       # dataclasses resolve the module by name
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text())


@pytest.fixture(scope="module")
def topology() -> dict:
    return yaml.safe_load(TOPOLOGY_FILE.read_text())


@pytest.fixture(scope="module")
def sable_prom() -> dict:
    return yaml.safe_load(SABLE_PROM_FILE.read_text())


@pytest.fixture(scope="module")
def prom_scrape() -> dict:
    return yaml.safe_load((LAB / "config" / "prometheus.yml").read_text())


# ---------------------------------------------------------------- compose <-> topology


def test_compose_services_match_topology_nodes(compose, topology):
    services = set(compose["services"])
    nodes = {n["id"] for n in topology["nodes"]}
    assert nodes <= services, f"topology nodes without a compose service: {nodes - services}"
    assert services - NON_TOPOLOGY_SERVICES == nodes, \
        f"compose services missing from topology: {services - NON_TOPOLOGY_SERVICES - nodes}"


def test_topology_types_edges_and_positional_order(topology):
    base = _load_base_module()
    ids = [n["id"] for n in topology["nodes"]]
    assert ids == sorted(ids), "SABLE maps node index by position and the live monitor posts sorted ids"
    assert len(ids) == len(set(ids))
    for n in topology["nodes"]:
        assert n["type"] in base.COMPONENT_TYPES, n
    expected_types = {"dns": "DNS_SERVER", "app": "APPLICATION_SERVICE", "db": "SERVER_VIRTUAL",
                      "db-replica": "SERVER_VIRTUAL", "proxy": "LOAD_BALANCER", "prometheus": "MONITORING_SERVER"}
    assert {n["id"]: n["type"] for n in topology["nodes"]} == expected_types
    edges = {(e["source"], e["target"]): (e["type"], e["criticality"]) for e in topology["edges"]}
    for e in topology["edges"]:
        assert e["type"] in base.DEPENDENCY_TYPES, e
        assert e["criticality"] in {c.value for c in base.Criticality}, e
        assert e["source"] in ids and e["target"] in ids, e
    assert edges[("app", "dns")] == ("DNS_DEPENDENCY", "HARD")
    assert edges[("app", "db")] == ("SERVICE_DEPENDENCY", "HARD")
    assert edges[("db", "db-replica")] == ("REPLICATION_DEPENDENCY", "SOFT")
    assert edges[("proxy", "app")] == ("SERVICE_DEPENDENCY", "HARD")
    assert edges[("proxy", "db")] == ("SERVICE_DEPENDENCY", "HARD")       # the catalog's failover shape
    for target in ("dns", "app", "db", "db-replica", "proxy"):
        assert edges[("prometheus", target)] == ("MONITORING_DEPENDENCY", "SOFT")


def test_compose_static_network_and_healthchecks(compose):
    net = compose["networks"]["lab"]
    subnet = net["ipam"]["config"][0]["subnet"]
    prefix = subnet.rsplit(".", 1)[0] + "."
    ips = []
    for name, svc in compose["services"].items():
        assert "healthcheck" in svc, f"{name} has no healthcheck"
        ip = svc["networks"]["lab"]["ipv4_address"]
        assert ip.startswith(prefix), f"{name}: {ip} not in {subnet}"
        ips.append(ip)
        assert svc.get("image"), f"{name}: image must be pinned"
        if "build" not in svc:
            assert ":" in svc["image"] and not svc["image"].endswith(":latest"), f"{name}: pin a tag"
    assert len(ips) == len(set(ips)), "duplicate fixed IPs"
    # the app's resolver is the dns container
    dns_ip = compose["services"]["dns"]["networks"]["lab"]["ipv4_address"]
    assert compose["services"]["app"]["dns"] == [dns_ip]


def test_lab_hosts_records_match_compose_ips(compose):
    records = {}
    for line in (LAB / "config" / "lab.hosts").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ip, name = line.split()
            records[name] = ip
    for svc in ("dns", "app", "db", "db-replica", "proxy", "prometheus"):
        assert records[f"{svc}.lab"] == compose["services"][svc]["networks"]["lab"]["ipv4_address"], svc


def test_blackbox_dns_probe_validates_the_pristine_db_answer(compose):
    bb = yaml.safe_load((LAB / "config" / "blackbox.yml").read_text())
    mod = bb["modules"]["dns_db_lab"]
    assert mod["prober"] == "dns" and mod["dns"]["query_name"] == "db.lab"
    db_ip = compose["services"]["db"]["networks"]["lab"]["ipv4_address"]
    regexes = mod["dns"]["validate_answer_rrs"]["fail_if_not_matches_regexp"]
    assert any(db_ip.replace(".", "\\.") in r for r in regexes)


# ---------------------------------------------------------------- SABLE prometheus config


def test_sable_node_map_names_topology_nodes_and_scrape_targets(sable_prom, topology, prom_scrape, compose):
    block = sable_prom["prometheus"]
    assert block["auto_discover"] is False
    assert block["url"] == sable_prom["prometheus_url"]
    assert sable_prom["topology"] == "topology.yaml"
    types = {n["id"]: n["type"] for n in topology["nodes"]}
    # every scrape instance Prometheus will produce: static targets, or the relabelled probe target
    instances_by_job = {}
    for sc in prom_scrape["scrape_configs"]:
        targets = [t for s in sc["static_configs"] for t in s["targets"]]
        instances_by_job[sc["job_name"]] = set(targets)
    mapped_nodes = []
    for instance, m in block["node_map"].items():
        assert set(m) <= {"node_id", "component_type", "job", "label_filters"}, instance
        assert m["node_id"] in types, f"{instance} maps to unknown node {m['node_id']}"
        assert m["component_type"] == types[m["node_id"]], instance
        assert m["job"] in instances_by_job, f"{instance}: job {m['job']} is not scraped"
        assert instance in instances_by_job[m["job"]], f"{instance} is not a target of job {m['job']}"
        host, port = instance.rsplit(":", 1)
        assert host in compose["services"], instance
        assert port.isdigit()
        mapped_nodes.append(m["node_id"])
    assert sorted(mapped_nodes) == sorted(types), "every topology node needs exactly one instance"
    assert "up" in block["queries"] and "probe_success" in block["queries"]["up"]
    assert "healthy" in block["queries"] and "app_dependency_ok" in block["queries"]["healthy"]


def test_sable_config_builds_a_real_prometheus_config_when_deps_allow(sable_prom):
    pytest.importorskip("numpy")                        # adapters/__init__ needs it
    from console.lab.live_monitor_lab import build_prometheus_config
    cfg = build_prometheus_config(sable_prom)
    assert cfg.auto_discover is False and len(cfg.node_map) == 6
    assert cfg.node_map["dns:53"].node_id == "dns" and cfg.node_map["dns:53"].instance == "dns:53"
    assert set(cfg.queries) == {"up", "healthy"}


def test_prometheus_scrapes_every_service_by_its_name(prom_scrape, compose):
    jobs = {sc["job_name"] for sc in prom_scrape["scrape_configs"]}
    assert set(compose["services"]) <= jobs, set(compose["services"]) - jobs
    for sc in prom_scrape["scrape_configs"]:
        if sc["job_name"] in ("dns", "proxy-http"):
            assert sc["metrics_path"] == "/probe"
            replacements = [r for r in sc["relabel_configs"] if r.get("target_label") == "__address__"]
            assert replacements and replacements[0]["replacement"] == "blackbox:9115"


# ---------------------------------------------------------------- scripts, services, pristine files


FAULTS = ["stop_service", "corrupt_config", "kill_primary", "poison_dns", "heal_all"]


@pytest.mark.parametrize("script", [f"faults/{f}.sh" for f in FAULTS] + ["faults/_lib.sh", "status.sh",
                                                                          "images/dnsmasq/entrypoint.sh"])
def test_shell_scripts_parse(script):
    path = LAB / script
    assert path.exists(), path
    sh = "sh" if script.endswith("entrypoint.sh") else "bash"
    subprocess.run([sh, "-n", str(path)], check=True)
    text = path.read_text()
    if script.startswith("faults/") and not script.endswith("_lib.sh"):
        assert "set -euo pipefail" in text
        assert "_lib.sh" in text, "fault scripts derive LAB_DIR from their own location via _lib.sh"


@pytest.mark.parametrize("source", ["services/app.py", "services/kv.py", "live_monitor_lab.py", "e2e.py"])
def test_python_sources_compile(source):
    py_compile.compile(str(LAB / source), doraise=True)


def test_services_are_stdlib_only():
    stdlib = set(sys.stdlib_module_names)
    for source in ("services/app.py", "services/kv.py", "e2e.py"):
        for line in (LAB / source).read_text().splitlines():
            line = line.strip()
            if line.startswith(("import ", "from ")):
                mod = line.split()[1].split(".")[0]
                assert mod in stdlib, f"{source}: {line}"


def test_every_mounted_config_has_an_identical_pristine_twin(compose):
    mounted = set()
    for svc in compose["services"].values():
        for vol in svc.get("volumes", []):
            src = vol.split(":", 1)[0]
            p = LAB / src
            if not (src.startswith("./config") or src.startswith("./proxy")):
                continue
            if p.is_file():
                mounted.add(p)
            elif p.is_dir():                                   # e.g. ./config:/lab/config
                mounted.update(f for f in p.iterdir() if f.is_file() and not f.name.endswith(".pristine"))
        for env_file in svc.get("env_file", []) or []:
            mounted.add(LAB / env_file)
    assert mounted, "no bind-mounted configs found"
    for p in mounted:
        twin = p.with_name(p.name + ".pristine")
        assert twin.exists(), f"missing pristine twin for {p.relative_to(LAB)}"
        assert twin.read_text() == p.read_text(), f"{p.relative_to(LAB)} differs from its pristine twin"
    for twin in list((LAB / "config").glob("*.pristine")) + list((LAB / "proxy").glob("*.pristine")):
        assert twin.with_name(twin.name[: -len(".pristine")]) in mounted, f"orphan pristine file {twin.name}"


def test_e2e_scenarios_reference_existing_fault_scripts():
    e2e = importlib.import_module("console.lab.e2e")
    scenarios = list(e2e.SCENARIOS) + [e2e.NEGATIVE]
    assert [(s.fault, s.expect_root) for s in e2e.SCENARIOS] == \
        [("stop_service", "dns"), ("corrupt_config", "app"), ("kill_primary", "db"), ("poison_dns", "dns")]
    topo_ids = {n["id"] for n in yaml.safe_load(TOPOLOGY_FILE.read_text())["nodes"]}
    for sc in scenarios:
        assert (LAB / "faults" / f"{sc.fault}.sh").exists(), sc
        assert sc.expect_root in topo_ids, sc
    assert e2e.NEGATIVE.negative and e2e.NEGATIVE_DISABLED_ACTION == "set_config_value"
    assert (LAB / "faults" / "heal_all.sh").exists()          # reset_lab() posts {"name": "heal_all"}


def test_makefile_has_the_lab_targets():
    text = (ROOT / "console" / "Makefile").read_text()
    for target in ("lab-up", "lab-down", "lab-reset", "lab-fault", "lab-status", "lab-e2e", "test"):
        assert f"\n{target}:" in text, target
    assert "console.lab.e2e" in text and "faults/heal_all.sh" in text


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not installed")
def test_docker_compose_config_renders():
    probe = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("docker compose plugin not available")
    result = subprocess.run(["docker", "compose", "-f", str(COMPOSE_FILE), "config"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rendered = yaml.safe_load(result.stdout)
    assert set(rendered["services"]) == set(yaml.safe_load(COMPOSE_FILE.read_text())["services"])
