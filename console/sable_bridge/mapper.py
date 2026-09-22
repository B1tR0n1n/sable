"""SABLE output -> Finding. Pure functions, no I/O.

Inputs are the dicts SABLE's server returns (INTEGRATION-NOTES B1):

    tick   docker/sable_engine.py:205-225, per node :179-196, enriched by
           docker/server.py:110-117 (label/topo_id/component_type) on replay and
           :483-487 (label/topo_id only) on the live path; `source="live"` :480;
           `mc_samples`/`mc_agreement`/`mc_variance` present only when MC dropout
           is on (:193-195, :222-225).
    recs   docker/sable_engine.py:448-453 enriched by docker/server.py:120-160:
           {summary, total_affected, actions[...], root_cause: int|None,
            root_cause_label?, root_cause_id?, root_cause_type?}.

The rules below are the ones decided in INTEGRATION-NOTES Part D (D1, D3, D4).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from console.contracts import (
    Confidence,
    ConfidenceMethod,
    DetectionMode,
    Evidence,
    Finding,
    NodeRef,
    Severity,
)
from console.topology import Topology

STATE_NAMES = ["healthy", "degraded", "failed", "unreachable", "oscillating"]
CRITICAL_DEPENDENTS = 3     # failed/unreachable root with >= this many affected dependents

_TICK_RE = re.compile(r"(?:at|since) tick (\d+)")
_CHANGES_RE = re.compile(r"\((\d+) transitions?\)")


class MappingError(ValueError):
    """The tick and the recommendations do not describe the same state
    (e.g. the root-cause index is missing from the tick)."""


# ---------------------------------------------------------------- helpers


def tick_time(tick: dict[str, Any]) -> datetime:
    """SABLE ticks carry no timestamp (the engine adds none; the server adds
    only inference_ms). Honour one if a future server adds it — ISO string or
    epoch seconds under `timestamp`/`time` — else use now."""
    raw = tick.get("timestamp", tick.get("time"))
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    if isinstance(raw, str):
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def node_id_for(index: int, tick_node: Optional[dict[str, Any]], topology: Topology) -> str:
    nid = topology.id_at(index)
    if nid is None and tick_node:
        nid = tick_node.get("topo_id")
    return nid or f"node-{index}"


def component_type_for(node_id: str, tick_node: Optional[dict[str, Any]], topology: Topology) -> str:
    ct = topology.component_type(node_id)
    if ct == "UNKNOWN" and tick_node and tick_node.get("component_type"):
        return str(tick_node["component_type"])
    return ct


def _nodes_by_index(tick: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(n["id"]): n for n in tick.get("nodes", []) if "id" in n}


def node_info_from_recs(recs: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Per-node trajectory facts SABLE computes (first_affected_tick,
    n_state_changes, ticks_in_current_state — sable_engine.py:350-356) but
    only exposes inside action `reason` strings ("First failure at tick 2",
    "Degraded since tick 3", "(4 transitions)"). Parsed back here, keyed by
    target_id. If a future server publishes `recs["nodes"]` verbatim, that
    list is used first."""
    out: dict[str, dict[str, int]] = {}
    for n in recs.get("nodes", []) or []:
        key = str(n.get("node_id", n.get("node")))
        facts = {k: int(n[k]) for k in ("first_affected_tick", "n_state_changes", "ticks_in_current_state") if k in n}
        if facts:
            out[key] = facts
    for a in recs.get("actions", []) or []:
        key = a.get("target_id")
        if not key or key in out:
            continue
        facts: dict[str, int] = {}
        m = _TICK_RE.search(a.get("reason", ""))
        if m:
            facts["first_affected_tick"] = int(m.group(1))
        m = _CHANGES_RE.search(a.get("reason", ""))
        if m:
            facts["n_state_changes"] = int(m.group(1))
        if facts:
            out[key] = facts
    return out


def root_cause_index(recs: dict[str, Any]) -> Optional[int]:
    rc = recs.get("root_cause")
    return None if rc is None else int(rc)


def dedup_key_for(tick: dict[str, Any], recs: dict[str, Any], topology: Topology,
                  site_id: str) -> Optional[tuple[str, str, str]]:
    """The would-be Finding's dedup key without building it: (site, root
    node id, root state). None when there is no root cause."""
    idx = root_cause_index(recs)
    if idx is None:
        return None
    node = _nodes_by_index(tick).get(idx)
    if node is None:
        raise MappingError(f"root cause index {idx} not in tick {tick.get('cycle')}")
    return (site_id, node_id_for(idx, node, topology), str(node["state"]))


# ---------------------------------------------------------------- pieces


def confidence_for(tick: dict[str, Any], node: dict[str, Any]) -> Confidence:
    """Confidence in the root-cause node's STATE CLASSIFICATION (D1) — not in
    the causal claim. SABLE's root cause is simply the first node to leave
    `healthy` (sable_engine.py:364,452); nothing in SABLE scores that choice.
    MC dropout on (tick.mc_samples > 0): agreement over the samples, carrying
    the sample count. Off: the engine's max-softmax confidence."""
    samples = int(tick.get("mc_samples") or 0)
    if samples > 0 and "mc_agreement" in node:
        return Confidence(score=_clamp(node["mc_agreement"]), method=ConfidenceMethod.mc_dropout,
                          samples=samples)
    return Confidence(score=_clamp(node["confidence"]), method=ConfidenceMethod.engine_native)


def detection_mode_for(tick: dict[str, Any], root_node: dict[str, Any],
                       upstream_unreachable: bool = False) -> DetectionMode:
    """`unmonitored_gap` when the root-cause node is `unreachable`: the
    diagnosis rests on the ABSENCE of telemetry (adapters/prometheus.py:281-289
    marks a node unreachable when `up` < 1 or it returned no metrics, and the
    scorer short-circuits that to the `unreachable` state). Everything else is
    `live_feed`. A replay tick (`source` != "live"; replays are unlabelled,
    server.py:400-413) is still `live_feed` — there is no better signal, and
    replayed data is only for mapping/dedup tests (D2).

    The served engine only ever names a FAILED node as root cause
    (sable_engine.py:452), so a gap can also sit UPSTREAM of it: when a node
    the root cause depends on is `unreachable`, the diagnosis still rests on
    absent telemetry — `upstream_unreachable` (computed by to_finding from
    the topology) flags that without overriding SABLE's root cause."""
    if str(root_node.get("state")) == "unreachable" or upstream_unreachable:
        return DetectionMode.unmonitored_gap
    return DetectionMode.live_feed


def severity_for(root_state: str, affected_dependents: int) -> Optional[Severity]:
    if root_state in ("failed", "unreachable"):
        return Severity.critical if affected_dependents >= CRITICAL_DEPENDENTS else Severity.high
    if root_state in ("degraded", "oscillating"):
        return Severity.medium
    return None   # healthy: nothing to report


def evidence_for(node_id: str, node: dict[str, Any], facts: dict[str, int], ts: datetime) -> list[Evidence]:
    """What SABLE exposes per node, prefixed with the node id since Evidence
    has no node field: `<id>:state_confidence`, the top-2 `<id>:p_<state>`,
    and the trajectory facts from the recommendations when present."""
    ev = [Evidence(metric=f"{node_id}:state_confidence", value=_clamp(node.get("confidence", 0.0)),
                   unit="p", timestamp=ts)]
    probs = node.get("probs") or {}
    for state, p in sorted(probs.items(), key=lambda kv: -float(kv[1]))[:2]:
        ev.append(Evidence(metric=f"{node_id}:p_{state}", value=float(p), unit="p", timestamp=ts))
    if "mc_agreement" in node:
        ev.append(Evidence(metric=f"{node_id}:mc_agreement", value=_clamp(node["mc_agreement"]),
                           unit="p", timestamp=ts))
    for key in ("first_affected_tick", "ticks_in_current_state", "n_state_changes"):
        if key in facts:
            ev.append(Evidence(metric=f"{node_id}:{key}", value=float(facts[key]), unit="ticks", timestamp=ts))
    return ev


def _clamp(x: Any) -> float:
    return max(0.0, min(1.0, float(x)))


# ---------------------------------------------------------------- main


def to_finding(tick: dict[str, Any], recs: dict[str, Any], topology: Topology, site_id: str,
               engine_version: str, previous: Optional[Finding] = None,
               summarize: Optional[Callable[[Finding], str]] = None) -> Optional[Finding]:
    """Map one SABLE (tick, recommendations) pair to a Finding.

    root_cause       recs["root_cause"] (node index) -> topology.id_at(idx); None -> no Finding.
                     Its state is the tick's per-node state. `healthy` -> no Finding.
    confidence       the root node's classification confidence (see confidence_for).
    detection_mode   see detection_mode_for.
    affected_nodes   every non-healthy node in the tick except the root, in index order.
    evidence         per root + affected node (see evidence_for).
    severity         failed/unreachable root -> critical when >= 3 affected nodes are
                     topology.dependents(root), else high; degraded/oscillating -> medium.
    summary          recs["summary"] verbatim, summary_generated=False — unless `summarize`
                     is given, whose text is used with summary_generated=True.
    previous         an OPEN Finding with the same dedup_key: the result is an updated copy
                     (same id/created_at/status/receipt_ids, occurrences+1, updated_at=now,
                     refreshed affected/evidence/confidence/severity/summary).

    Raises MappingError when the recs' root index is not in the tick, or when
    `previous` has a different dedup_key.
    """
    idx = root_cause_index(recs)
    if idx is None:
        return None
    by_idx = _nodes_by_index(tick)
    root_node = by_idx.get(idx)
    if root_node is None:
        raise MappingError(f"root cause index {idx} not in tick {tick.get('cycle')}")
    root_state = str(root_node["state"])
    if root_state == "healthy":
        return None

    ts = tick_time(tick)
    facts = node_info_from_recs(recs)
    root_id = node_id_for(idx, root_node, topology)
    root = NodeRef(node_id=root_id, component_type=component_type_for(root_id, root_node, topology),
                   state=root_state)

    affected: list[NodeRef] = []
    evidence = evidence_for(root_id, root_node, facts.get(root_id, {}), ts)
    for i in sorted(by_idx):
        n = by_idx[i]
        if i == idx or str(n.get("state", "healthy")) == "healthy":
            continue
        nid = node_id_for(i, n, topology)
        affected.append(NodeRef(node_id=nid, component_type=component_type_for(nid, n, topology),
                                state=str(n["state"])))
        evidence.extend(evidence_for(nid, n, facts.get(nid, {}), ts))

    dependents = set(topology.dependents(root_id))
    severity = severity_for(root_state, sum(1 for a in affected if a.node_id in dependents))
    if severity is None:
        return None

    fields: dict[str, Any] = dict(
        root_cause=root,
        affected_nodes=affected,
        evidence=evidence,
        confidence=confidence_for(tick, root_node),
        severity=severity,
        summary=str(recs.get("summary", "")),
        summary_generated=False,
    )

    if previous is not None:
        if previous.dedup_key != (site_id, root_id, root_state):
            raise MappingError(f"previous finding {previous.id} has a different dedup_key")
        finding = previous.model_copy(update={
            **fields,
            "occurrences": previous.occurrences + 1,
            "updated_at": datetime.now(timezone.utc),
        })
    else:
        upstream = set(topology.dependencies(root_id))
        upstream_gap = any(a.state == "unreachable" and a.node_id in upstream for a in affected)
        finding = Finding(site_id=site_id,
                          detection_mode=detection_mode_for(tick, root_node, upstream_gap),
                          engine_version=engine_version, **fields)

    if summarize is not None:
        finding = finding.model_copy(update={"summary": str(summarize(finding)), "summary_generated": True})
    return finding
