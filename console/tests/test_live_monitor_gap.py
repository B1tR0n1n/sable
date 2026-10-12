"""The live monitor never posts a monitoring gap as an outage.

When Prometheus is down (a lab heal recreates it; in the field it restarts),
the adapter still returns every known node, with no metrics. Scored, that is
"every node unreachable". Posted after a long real outage it latched the
trained model's temporal chain on a node for hours of healthy telemetry
(reproduced on the lab 2026-09-23: db down 156 ticks + 4 blackout ticks →
db read unreachable for 120+ healthy ticks; either alone recovers). So a cycle
with no telemetry for any node is skipped, and the next real one goes out.
"""

import sys
from pathlib import Path

import pytest

DOCKER = Path(__file__).resolve().parents[2] / "docker"
if str(DOCKER) not in sys.path:
    sys.path.insert(0, str(DOCKER))

live_monitor = pytest.importorskip("live_monitor")
from adapters.base import NodeSnapshot, SystemSnapshot  # noqa: E402


def snapshot(metrics_by_node):
    return SystemSnapshot(nodes={nid: NodeSnapshot(node_id=nid, component_type="SERVER_VIRTUAL", metrics=m)
                                 for nid, m in metrics_by_node.items()})


class FakeProm:
    def __init__(self, snaps):
        self.snaps = list(snaps)

    def poll(self):
        return self.snaps.pop(0)


@pytest.fixture
def monitor(monkeypatch):
    mon = live_monitor.LiveMonitor(dry_run=False, pillar1_checkpoint="")
    pushed = []
    monkeypatch.setattr(mon, "_push_to_engine", lambda inputs, snap: pushed.append(snap) or None)
    monkeypatch.setattr(mon, "_print_cycle", lambda *a, **kw: None)
    return mon, pushed


def test_a_cycle_without_any_telemetry_is_skipped_not_posted(monitor):
    mon, pushed = monitor
    healthy = {"db": {"up": 1.0}, "app": {"up": 1.0}}
    blackout = {"db": {}, "app": {}}
    mon.prom = FakeProm([snapshot(healthy), snapshot(blackout), snapshot(blackout), snapshot(healthy)])
    for _ in range(4):
        mon._poll_cycle()
    assert len(pushed) == 2                                   # the two blackout cycles never reached SABLE
    assert all(any(n.metrics for n in s.nodes.values()) for s in pushed)
    assert mon._gap_cycles == 0                               # reset once telemetry came back


def test_one_silent_node_is_still_an_observation(monitor):
    """Only a total gap is skipped: a single node with no metrics is data (it
    may really be down) and goes out as before."""
    mon, pushed = monitor
    mon.prom = FakeProm([snapshot({"db": {}, "app": {"up": 1.0}})])
    mon._poll_cycle()
    assert len(pushed) == 1
