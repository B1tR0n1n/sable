"""
SABLE Graph Builder
=====================
Constructs the infrastructure dependency graph from topology sources.

Prometheus tells you what's happening. The graph builder tells you what's
connected. Without the graph, the GNN has no edges and the cascade
predictions have no propagation paths.

Supported sources:
  - Manual YAML/JSON config (always available, MSP default)
  - Netbox API (if available)
  - ServiceNow CMDB (future)

The graph builder produces EdgeSnapshots that get merged into the
SystemSnapshot alongside Prometheus metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .base import EdgeSnapshot, Criticality, DEPENDENCY_TYPES

log = logging.getLogger(__name__)


def load_topology_from_file(path: str | Path) -> list[EdgeSnapshot]:
    """Load topology from a YAML or JSON config file.

    Expected format:
    ```yaml
    edges:
      - source: core-sw-1
        target: acc-sw-1
        type: NETWORK_PATH
        criticality: HARD

      - source: web-server-1
        target: core-sw-1
        type: NETWORK_PATH
        criticality: HARD

      - source: web-server-1
        target: dns-1
        type: DNS_DEPENDENCY
        criticality: SOFT
    ```
    """
    path = Path(path)
    if not path.exists():
        log.error("Topology file not found: %s", path)
        return []

    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            log.error("PyYAML required for YAML topology files: pip install pyyaml")
            return []
        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        log.error("Unknown topology file format: %s", path.suffix)
        return []

    edges = []
    for entry in data.get("edges", []):
        dep_type = entry.get("type", "NETWORK_PATH")
        if dep_type not in DEPENDENCY_TYPES:
            log.warning("Unknown dependency type: %s — defaulting to NETWORK_PATH", dep_type)
            dep_type = "NETWORK_PATH"

        crit_str = entry.get("criticality", "SOFT").upper()
        try:
            crit = Criticality(crit_str)
        except ValueError:
            crit = Criticality.SOFT

        edges.append(EdgeSnapshot(
            source_id=entry["source"],
            target_id=entry["target"],
            dep_type=dep_type,
            criticality=crit,
            confidence=entry.get("confidence", 0.9),
            metadata=entry.get("metadata", {}),
        ))

    log.info("Loaded %d edges from %s", len(edges), path)
    return edges


def load_topology_from_netbox(
    url: str,
    token: str,
    site: Optional[str] = None,
    timeout: float = 15.0,
) -> list[EdgeSnapshot]:
    """Pull topology from Netbox API.

    Maps Netbox cables and circuits to SABLE dependency edges.
    Netbox is common at MSPs for DCIM/IPAM.

    Args:
        url:   Netbox base URL (e.g. https://netbox.example.com)
        token: API token
        site:  Optional site filter (slug)
    """
    import requests

    session = requests.Session()
    session.headers["Authorization"] = f"Token {token}"
    session.headers["Accept"] = "application/json"

    edges = []

    # Fetch cables (physical connectivity)
    params = {"limit": 1000}
    if site:
        params["site"] = site

    try:
        resp = session.get(
            f"{url.rstrip('/')}/api/dcim/cables/",
            params=params,
            timeout=timeout,
        )
        resp.raise_for_status()
        cables = resp.json().get("results", [])
    except requests.RequestException as e:
        log.error("Netbox cables fetch failed: %s", e)
        return edges

    for cable in cables:
        a_terms = cable.get("a_terminations", [])
        b_terms = cable.get("b_terminations", [])
        if not a_terms or not b_terms:
            continue

        source_id = _netbox_term_to_node(a_terms[0])
        target_id = _netbox_term_to_node(b_terms[0])
        if not source_id or not target_id:
            continue

        edges.append(EdgeSnapshot(
            source_id=source_id,
            target_id=target_id,
            dep_type="NETWORK_PATH",
            criticality=Criticality.HARD,
            confidence=0.95,
            metadata={"source": "netbox", "cable_id": str(cable.get("id", ""))},
        ))

    # Fetch circuits (WAN links)
    try:
        resp = session.get(
            f"{url.rstrip('/')}/api/circuits/circuits/",
            params={"limit": 500, "status": "active"},
            timeout=timeout,
        )
        resp.raise_for_status()
        circuits = resp.json().get("results", [])

        for circuit in circuits:
            # Circuits with two terminations represent WAN links
            term_a = circuit.get("termination_a")
            term_z = circuit.get("termination_z")
            if not term_a or not term_z:
                continue

            site_a = term_a.get("site", {}).get("slug", "")
            site_z = term_z.get("site", {}).get("slug", "")
            if site_a and site_z:
                edges.append(EdgeSnapshot(
                    source_id=f"wan-{site_a}",
                    target_id=f"wan-{site_z}",
                    dep_type="NETWORK_PATH",
                    criticality=Criticality.HARD,
                    confidence=0.9,
                    metadata={"source": "netbox", "circuit": circuit.get("cid", "")},
                ))
    except requests.RequestException as e:
        log.warning("Netbox circuits fetch failed: %s", e)

    log.info("Discovered %d edges from Netbox", len(edges))
    return edges


def _netbox_term_to_node(termination: dict) -> Optional[str]:
    """Extract a node ID from a Netbox cable termination."""
    obj = termination.get("object", {})
    device = obj.get("device", {})
    name = device.get("name", "")
    return name if name else None
