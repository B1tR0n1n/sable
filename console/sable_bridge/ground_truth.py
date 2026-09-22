"""The health scorer's reading beside the model's.

The live monitor POSTs `ground_truth` (a state index per node, node order)
to SABLE's /api/live_tick (docker/live_monitor.py:260-277). The engine keeps
it in its history (sable_engine.py:302) and reports only the aggregate
`accuracy` on the tick it broadcasts — the per-node reading surfaces as
`truth` on each trajectory entry of GET /api/node/{idx} (sable_engine.py:
385, get_node_report). So:

    ground_truth_states(tick, topology)   pure: {node_id: state} from a tick
                                          carrying `ground_truth` (indices,
                                          the POST/stub shape) or per-node
                                          `truth`; {} when it carries neither
    attach_ground_truth(tick, client)     a NEW tick dict with `ground_truth`
                                          fetched from /api/node/{idx} for the
                                          tick's cycle (None where unknown;
                                          the whole field None when nothing
                                          was usable, so it is fetched once)

The stub (lab/sable_stub.py) has no model and emits `ground_truth` on the
tick itself (state == truth); the console never needs to fetch there.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .mapper import STATE_NAMES

log = logging.getLogger(__name__)


def _node_id(index: int, node: Optional[dict[str, Any]], topology) -> str:
    nid = None
    if topology is not None:
        try:
            nid = topology.id_at(index)
        except (TypeError, ValueError):
            nid = None
    if nid is None and node:
        nid = node.get("node_id") or node.get("topo_id")
    return str(nid or f"node-{index}")


def _state_name(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value if value in STATE_NAMES else None
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return None
    return STATE_NAMES[idx] if 0 <= idx < len(STATE_NAMES) else None


def ground_truth_states(tick: dict[str, Any], topology) -> dict[str, str]:
    """{node_id: scorer state} for every node the tick knows the truth of."""
    nodes = list(tick.get("nodes") or [])
    by_index = {int(n["id"]): n for n in nodes if "id" in n}
    out: dict[str, str] = {}
    gt = tick.get("ground_truth")
    if isinstance(gt, list):
        for i, raw in enumerate(gt):
            state = _state_name(raw)
            if state is not None:
                out[_node_id(i, by_index.get(i), topology)] = state
    for n in nodes:
        state = _state_name(n.get("truth")) if "truth" in n else None
        if state is not None and "id" in n:
            out[_node_id(int(n["id"]), n, topology)] = state
    return out


def _truth_from_report(report: Any, cycle: Any) -> Optional[int]:
    """The `truth` of the trajectory entry for `cycle`, as a state index."""
    if not isinstance(report, dict):
        return None
    for entry in reversed(report.get("trajectory") or []):
        if entry.get("cycle") == cycle:
            state = _state_name(entry.get("truth"))
            return STATE_NAMES.index(state) if state is not None else None
    return None


def attach_ground_truth(tick: dict[str, Any], client: Any) -> dict[str, Any]:
    """Fetch the scorer's reading for each node of `tick` from SABLE's
    /api/node/{idx}. Returns a new dict; the input is not modified. A tick
    that already carries `ground_truth` is returned as is."""
    if "ground_truth" in tick:
        return tick
    cycle = tick.get("cycle")
    truths: list[Optional[int]] = []
    failed: list[int] = []
    for i, _ in enumerate(tick.get("nodes") or []):
        try:
            truths.append(_truth_from_report(client.node(i), cycle))
        except Exception as e:                    # noqa: BLE001 — one node's report is not the verdict
            failed.append(i)
            log.debug("ground truth for node %s: %s", i, e)
            truths.append(None)
    if failed:
        log.warning("tick %s: no node report for %d node(s) (%s); telemetry unknown there",
                    cycle, len(failed), ", ".join(map(str, failed[:8])))
    known = any(v is not None for v in truths)
    return {**tick, "ground_truth": truths if known else None}
