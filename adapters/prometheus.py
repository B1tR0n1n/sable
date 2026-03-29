"""
SABLE Telemetry Adapter — Prometheus
======================================
Pulls metrics from a Prometheus/VictoriaMetrics instance via PromQL and
converts them into SABLE's SystemSnapshot format.

Prometheus is the most common monitoring backend at MSPs — free, ubiquitous,
and covers compute/network/storage metrics via node_exporter, snmp_exporter,
blackbox_exporter, and cAdvisor.

Configuration:
    adapter = PrometheusAdapter(
        url="http://prometheus:9090",
        node_map={...},        # Maps Prometheus targets → SABLE component types
    )
    snapshot = adapter.poll()

The adapter queries Prometheus for current metric values across all targets,
maps each target to a SABLE component type, and returns a SystemSnapshot
with raw metrics. The health scorer runs afterward to classify states.

Topology discovery is NOT Prometheus's job — that comes from CMDB/Netbox/manual
config via graph_builder.py. Prometheus just provides the metric values.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin

import requests

from .base import (
    TelemetryAdapter, NodeSnapshot, EdgeSnapshot, SystemSnapshot,
)

log = logging.getLogger(__name__)


# ── Node Mapping ─────────────────────────────────────────────────────────


@dataclass
class NodeMapping:
    """Maps a Prometheus target to a SABLE component.

    Can be configured per-customer or auto-generated from Prometheus
    target labels.
    """
    node_id: str                  # SABLE node ID
    component_type: str           # One of COMPONENT_TYPES
    instance: str                 # Prometheus instance label (host:port)
    job: str = ""                 # Prometheus job label
    label_filters: dict[str, str] = field(default_factory=dict)


@dataclass
class PrometheusConfig:
    """Configuration for Prometheus adapter.

    node_map: Maps Prometheus instances to SABLE nodes.
              If empty, auto-discovery from targets is attempted.
    queries:  PromQL queries to execute per poll cycle.
              Keys are metric names, values are PromQL expressions.
    """
    url: str = "http://localhost:9090"
    timeout: float = 10.0

    # Explicit node mappings. Key = Prometheus instance label.
    node_map: dict[str, NodeMapping] = field(default_factory=dict)

    # Auto-discovery: infer component type from Prometheus job/labels
    auto_discover: bool = True

    # Job name → component type mapping for auto-discovery
    job_type_map: dict[str, str] = field(default_factory=lambda: {
        "node": "SERVER_PHYSICAL",
        "node_exporter": "SERVER_PHYSICAL",
        "windows": "SERVER_PHYSICAL",
        "vmware": "HYPERVISOR",
        "esxi": "HYPERVISOR",
        "snmp": "CORE_SWITCH",             # Refined by sysDescr label
        "snmp_network": "CORE_SWITCH",
        "blackbox": "APPLICATION_SERVICE",
        "cadvisor": "SERVER_VIRTUAL",
        "docker": "SERVER_VIRTUAL",
        "kubernetes-nodes": "SERVER_PHYSICAL",
        "kubernetes-pods": "APPLICATION_SERVICE",
        "mysql": "APPLICATION_SERVICE",
        "postgres": "APPLICATION_SERVICE",
        "redis": "APPLICATION_SERVICE",
        "nginx": "LOAD_BALANCER",
        "haproxy": "LOAD_BALANCER",
        "bind": "DNS_SERVER",
        "coredns": "DNS_SERVER",
        "isc-dhcp": "DHCP_SERVER",
    })

    # Jobs that use Windows exporter queries instead of Linux node_exporter
    windows_jobs: set[str] = field(default_factory=lambda: {"windows"})

    # Label-based type refinement for SNMP targets
    snmp_type_map: dict[str, str] = field(default_factory=lambda: {
        "cisco": "CORE_SWITCH",
        "juniper": "ROUTER",
        "arista": "CORE_SWITCH",
        "palo alto": "FIREWALL",
        "fortinet": "FIREWALL",
        "fortigate": "FIREWALL",
        "f5": "LOAD_BALANCER",
        "netapp": "STORAGE_ARRAY",
        "dell emc": "STORAGE_ARRAY",
        "pure": "STORAGE_ARRAY",
        "nimble": "STORAGE_ARRAY",
    })

    # PromQL queries per metric. {metric_name: promql_expression}
    # These are instant queries — they return the current value.
    # Multi-value queries (per device/mountpoint) use max by (instance)
    # to take worst-case value per node.
    queries: dict[str, str] = field(default_factory=lambda: {
        # CPU
        "cpu_utilization": (
            '100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
        ),
        # Memory
        "mem_used_pct": (
            '100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)'
        ),
        # Disk — max across mountpoints (worst-case)
        "disk_used_pct": (
            'max by (instance) (100 - (node_filesystem_avail_bytes{fstype!~"tmpfs|overlay"} '
            '/ node_filesystem_size_bytes{fstype!~"tmpfs|overlay"} * 100))'
        ),
        "disk_io_util": (
            'max by (instance) (rate(node_disk_io_time_seconds_total[5m]) * 100)'
        ),
        # Network — sum across real interfaces, then max by instance
        "net_error_rate": (
            'sum by (instance) ('
            'rate(node_network_receive_errs_total{device!~"lo|veth.*|docker.*|br-.*"}[5m]) + '
            'rate(node_network_transmit_errs_total{device!~"lo|veth.*|docker.*|br-.*"}[5m]))'
        ),
        # Uptime / availability (1 = up)
        "up": 'up',
    })

    # Additional PromQL queries for SNMP targets
    snmp_queries: dict[str, str] = field(default_factory=lambda: {
        "interface_errors": (
            'max by (instance) (rate(ifInErrors[5m]) + rate(ifOutErrors[5m]))'
        ),
        "interface_discards": (
            'max by (instance) (rate(ifInDiscards[5m]) + rate(ifOutDiscards[5m]))'
        ),
        "net_bandwidth_util": (
            'max by (instance) ('
            '100 * (rate(ifHCInOctets[5m]) + rate(ifHCOutOctets[5m])) * 8 / ifHighSpeed / 1e6)'
        ),
    })

    # Windows exporter queries (dispatched for windows_jobs targets)
    windows_queries: dict[str, str] = field(default_factory=lambda: {
        "cpu_utilization": (
            '100 - (avg by (instance) (rate(windows_cpu_time_total{mode="idle"}[5m])) * 100)'
        ),
        "mem_used_pct": (
            '100 * (1 - windows_os_physical_memory_free_bytes / windows_cs_physical_memory_bytes)'
        ),
        "disk_used_pct": (
            'max by (instance) ('
            '100 * (1 - windows_logical_disk_free_bytes{volume!~"HarddiskVolume.*"} '
            '/ windows_logical_disk_size_bytes{volume!~"HarddiskVolume.*"}))'
        ),
    })


# ── Prometheus Adapter ───────────────────────────────────────────────────


class PrometheusAdapter(TelemetryAdapter):
    """Fetches metrics from Prometheus and produces SystemSnapshots.

    Usage:
        config = PrometheusConfig(url="http://prometheus:9090")
        adapter = PrometheusAdapter(config)
        snapshot = adapter.poll()  # Returns SystemSnapshot with raw metrics
    """

    def __init__(self, config: Optional[PrometheusConfig] = None):
        self.config = config or PrometheusConfig()
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"
        self._last_targets: dict[str, NodeMapping] = {}

    def poll(self) -> SystemSnapshot:
        """Query Prometheus for current metrics across all targets.

        1. Discover targets (if auto_discover)
        2. Run PromQL queries (Linux, Windows, SNMP dispatched separately)
        3. Map results to NodeSnapshots
        4. Mark unreachable only if zero metrics received
        """
        now = time.time()

        # Step 1: Build node map from configured mappings + auto-discovery
        node_map = self._build_node_map()

        # Initialize nodes from map
        nodes: dict[str, NodeSnapshot] = {}
        for inst, mapping in node_map.items():
            nodes[mapping.node_id] = NodeSnapshot(
                node_id=mapping.node_id,
                component_type=mapping.component_type,
                timestamp=now,
                labels={"instance": inst, "job": mapping.job},
            )

        # Step 2: Dispatch queries by exporter type
        self._dispatch_queries(node_map, nodes, now)

        # Step 3: Mark unreachable based on actual data received
        self._mark_unreachable(nodes)

        self._last_targets = node_map
        return SystemSnapshot(nodes=nodes, timestamp=now, source="prometheus")

    def _build_node_map(self) -> dict[str, NodeMapping]:
        """Merge configured node mappings with auto-discovered targets."""
        node_map = dict(self.config.node_map)
        if self.config.auto_discover:
            discovered = self._discover_targets()
            for inst, mapping in discovered.items():
                if inst not in node_map:
                    node_map[inst] = mapping

        # Detect duplicate node_ids and warn
        seen_ids: dict[str, str] = {}
        for inst, mapping in node_map.items():
            if mapping.node_id in seen_ids:
                log.warning(
                    "Node ID collision: '%s' mapped by both '%s' and '%s'. "
                    "Second instance will overwrite metrics.",
                    mapping.node_id, seen_ids[mapping.node_id], inst,
                )
            seen_ids[mapping.node_id] = inst

        return node_map

    def _dispatch_queries(
        self,
        node_map: dict[str, NodeMapping],
        nodes: dict[str, NodeSnapshot],
        now: float,
    ):
        """Classify targets by exporter type and run the appropriate query sets."""
        snmp_instances = set()
        windows_instances = set()
        linux_instances = set()
        for inst, m in node_map.items():
            if m.job in ("snmp", "snmp_network"):
                snmp_instances.add(inst)
            elif m.job in self.config.windows_jobs:
                windows_instances.add(inst)
            else:
                linux_instances.add(inst)

        if linux_instances:
            self._run_queries(self.config.queries, node_map, nodes, now)
        if windows_instances:
            self._run_queries(self.config.windows_queries, node_map, nodes, now)
        if snmp_instances:
            self._run_queries(self.config.snmp_queries, node_map, nodes, now)
        if "up" not in self.config.queries:
            self._run_queries({"up": "up"}, node_map, nodes, now)

    @staticmethod
    def _mark_unreachable(nodes: dict[str, NodeSnapshot]):
        """Flag nodes as unreachable based on 'up' metric or absence of data."""
        for node in nodes.values():
            up = node.metrics.get("up")
            if up is not None and up < 1.0:
                node.reachable = False
            elif not node.metrics:
                node.reachable = False
                node.stale_seconds = 999.0

    def discover_topology(self) -> list[EdgeSnapshot]:
        """Prometheus doesn't know topology — returns empty.

        Topology comes from graph_builder.py (CMDB, Netbox, manual config).
        This method exists to satisfy the interface.
        """
        return []

    # ── Query Execution ──────────────────────────────────────────────────

    def _run_queries(
        self,
        queries: dict[str, str],
        node_map: dict[str, NodeMapping],
        nodes: dict[str, NodeSnapshot],
        _now: float,
    ):
        """Execute a set of PromQL queries and populate node metrics.

        For multi-value results (multiple series per instance), takes the
        first match. PromQL queries should use aggregation (max/avg by instance)
        to collapse multi-series results before they get here.
        """
        for metric_name, promql in queries.items():
            results = self._query_instant(promql)
            if results is None:
                continue

            # Track which nodes already got this metric (first-write wins)
            seen_nodes: set[str] = set()

            for sample in results:
                instance = sample.get("metric", {}).get("instance", "")
                value = self._extract_value(sample)
                if value is None:
                    continue

                mapping = node_map.get(instance)
                if mapping is None:
                    continue

                node = nodes.get(mapping.node_id)
                if node is None:
                    continue

                # First value wins — prevents clobbering from multi-series results
                # that escaped PromQL aggregation
                if mapping.node_id in seen_nodes:
                    continue
                seen_nodes.add(mapping.node_id)

                node.metrics[metric_name] = value

    # ── Prometheus API ────────────────────────────────────────────────────

    def _query_instant(self, promql: str) -> Optional[list[dict]]:
        """Execute an instant PromQL query. Returns result list or None."""
        try:
            url = urljoin(self.config.url.rstrip("/") + "/", "api/v1/query")
            resp = self._session.get(
                url,
                params={"query": promql},
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "success":
                log.warning("PromQL query failed: %s — %s", promql, data.get("error", ""))
                return None

            return data.get("data", {}).get("result", [])

        except requests.RequestException as e:
            log.warning("Prometheus request failed: %s — %s", promql, e)
            return None

    def _extract_value(self, sample: dict) -> Optional[float]:
        """Extract the numeric value from a Prometheus sample."""
        val = sample.get("value", [None, None])
        if len(val) < 2:
            return None
        try:
            v = float(val[1])
            # NaN/Inf → None
            if v != v or v == float("inf") or v == float("-inf"):
                return None
            return v
        except (ValueError, TypeError):
            return None

    # ── Target Discovery ─────────────────────────────────────────────────

    def _discover_targets(self) -> dict[str, NodeMapping]:
        """Auto-discover nodes from Prometheus targets API."""
        targets = self._fetch_active_targets()
        mappings: dict[str, NodeMapping] = {}

        for target in targets:
            mapping = self._target_to_mapping(target)
            if mapping is not None:
                mappings[mapping.instance] = mapping

        return mappings

    def _fetch_active_targets(self) -> list[dict]:
        """Retrieve the active targets list from the Prometheus API."""
        try:
            url = urljoin(self.config.url.rstrip("/") + "/", "api/v1/targets")
            resp = self._session.get(url, timeout=self.config.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            log.warning("Target discovery failed: %s", e)
            return []

        return data.get("data", {}).get("activeTargets", [])

    def _target_to_mapping(self, target: dict) -> Optional[NodeMapping]:
        """Convert a single Prometheus target dict into a NodeMapping, or None."""
        instance = target.get("labels", {}).get("instance", "")
        job = target.get("labels", {}).get("job", "")
        if not instance:
            return None

        comp_type = self._infer_component_type(target)
        node_id = self._make_node_id(instance, job)

        return NodeMapping(
            node_id=node_id,
            component_type=comp_type,
            instance=instance,
            job=job,
        )

    def _infer_component_type(self, target: dict) -> str:
        """Infer SABLE component type from Prometheus target metadata."""
        labels = target.get("labels", {})
        job = labels.get("job", "")

        # Direct job mapping
        comp_type = self.config.job_type_map.get(job)
        if comp_type:
            # Refine SNMP targets by sysDescr
            if job in ("snmp", "snmp_network"):
                comp_type = self._refine_snmp_type(labels, comp_type)
            return comp_type

        # Fallback: guess from labels
        if "kubernetes" in job.lower() or "k8s" in job.lower():
            return "APPLICATION_SERVICE"

        return "SERVER_PHYSICAL"  # Conservative default

    def _refine_snmp_type(self, labels: dict, default: str) -> str:
        """Refine SNMP target type from sysDescr or other labels."""
        sys_descr = labels.get("sysDescr", "").lower()
        sys_name = labels.get("sysName", "").lower()
        combined = sys_descr + " " + sys_name

        for keyword, comp_type in self.config.snmp_type_map.items():
            if keyword in combined:
                return comp_type

        return default

    def _make_node_id(self, instance: str, job: str) -> str:
        """Generate a stable node ID from Prometheus instance label."""
        # Strip port from instance (e.g. "server1:9100" → "server1")
        host = instance.split(":")[0] if ":" in instance else instance
        # Prefix with job for disambiguation (e.g. two exporters on same host)
        if job and job not in ("node", "node_exporter", "windows"):
            return f"{host}_{job}"
        return host
