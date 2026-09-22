"""The console's dependency graph: SABLE's edge convention (source depends on
target), transitive dependents/dependencies, blast radius, and the positional
index ↔ id mapping the server uses."""

from pathlib import Path

from console.topology import Topology

SABLE_ROOT = Path(__file__).resolve().parents[2]

# app depends on dns and db; db replicates from db-replica; vdi depends on app
NODES = [{"id": "dns", "type": "DNS_SERVER", "label": "DNS"},
         {"id": "db", "type": "SERVER_VIRTUAL", "label": "DB"},
         {"id": "db-replica", "type": "SERVER_VIRTUAL", "label": "DB replica"},
         {"id": "app", "type": "APPLICATION_SERVICE", "label": "App"},
         {"id": "vdi", "type": "VDI_BROKER", "label": "VDI"}]
EDGES = [{"source": "app", "target": "dns", "type": "DNS_DEPENDENCY", "criticality": "HARD"},
         {"source": "app", "target": "db", "type": "SERVICE_DEPENDENCY", "criticality": "HARD"},
         {"source": "db", "target": "db-replica", "type": "REPLICATION_DEPENDENCY", "criticality": "SOFT"},
         {"source": "vdi", "target": "app", "type": "SERVICE_DEPENDENCY", "criticality": "HARD"}]


def test_dependents_are_transitive_predecessors():
    t = Topology(NODES, EDGES, "t")
    assert t.dependents("dns") == ["app", "vdi"]           # dns breaks → app, then vdi
    assert t.dependents("db-replica") == ["db", "app", "vdi"]
    assert t.dependents("vdi") == []
    assert t.dependencies("vdi") == ["app", "db", "dns", "db-replica"]
    assert t.direct_dependents("app") == ["vdi"]
    assert t.direct_dependencies("app") == ["db", "dns"]


def test_blast_radius_is_target_plus_dependents():
    t = Topology(NODES, EDGES)
    assert t.blast_radius("db") == ["db", "app", "vdi"]
    assert t.blast_radius("vdi") == ["vdi"]


def test_hard_only_walk_stops_at_soft_edges():
    t = Topology(NODES, EDGES)
    assert t.dependents("db-replica", hard_only=True) == []   # db→replica is SOFT
    assert t.dependents("dns", hard_only=True) == ["app", "vdi"]


def test_positional_index_matches_sable_server_mapping():
    t = Topology(NODES, EDGES)
    assert t.index_of("db-replica") == 2 and t.id_at(2) == "db-replica"
    assert t.index_of("nope") is None and t.id_at(99) is None
    assert t.component_type("dns") == "DNS_SERVER" and t.label("app") == "App"


def test_loads_the_real_msp_demo_yaml():
    t = Topology.from_yaml(SABLE_ROOT / "adapters" / "topologies" / "msp_demo.yaml")
    assert t.name == "msp_demo" and len(t.nodes) >= 30 and t.edges
    # the same payload shape /api/topology returns round-trips
    t2 = Topology.from_api({"name": t.name, "nodes": list(t.nodes.values()), "edges": t.edges})
    assert t2.dependents("core-sw-1") == t.dependents("core-sw-1")
    assert t.dependents("core-sw-1"), "a core switch has downstream dependents"
