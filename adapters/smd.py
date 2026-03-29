"""
SABLE Telemetry Adapter - Server Machine Dataset (SMD)
========================================================
Converts the SMD real server telemetry dataset into SABLE SystemSnapshots.

SMD: 28 server machines, 3 groups, 38 features per machine.
Features are normalized [0,1] metrics: CPU, memory, network I/O, disk, etc.

The 38 features map to SABLE's health scoring as follows:
  - Features 0-7: CPU-related (utilization across cores)
  - Features 8-15: Memory-related
  - Features 16-23: Network I/O
  - Features 24-31: Disk I/O
  - Features 32-37: System-level (load, processes, etc.)

Each machine becomes a SABLE node. Machine groups (1, 2, 3) define
topology clusters with inter-group network dependencies and intra-group
service dependencies.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from .base import (
    TelemetryAdapter, NodeSnapshot, EdgeSnapshot, SystemSnapshot,
    Criticality, COMPONENT_TYPES,
)


# SMD feature groups (indices into the 38-dim feature vector)
# These are approximate - SMD doesn't document exact feature names,
# but the ordering follows standard server metrics
FEATURE_GROUPS = {
    "cpu": list(range(0, 8)),
    "memory": list(range(8, 16)),
    "network": list(range(16, 24)),
    "disk": list(range(24, 32)),
    "system": list(range(32, 38)),
}

# Map SMD machine groups to SABLE component types
# Group 1: physical servers (heavier workloads)
# Group 2: application servers (VMs)
# Group 3: infrastructure services
GROUP_TYPE_MAP = {
    "1": [
        "SERVER_PHYSICAL", "SERVER_PHYSICAL", "HYPERVISOR",
        "SERVER_PHYSICAL", "HYPERVISOR", "SERVER_PHYSICAL",
        "SERVER_PHYSICAL", "STORAGE_ARRAY",
    ],
    "2": [
        "SERVER_VIRTUAL", "SERVER_VIRTUAL", "APPLICATION_SERVICE",
        "SERVER_VIRTUAL", "APPLICATION_SERVICE", "SERVER_VIRTUAL",
        "APPLICATION_SERVICE", "SERVER_VIRTUAL", "LOAD_BALANCER",
    ],
    "3": [
        "SERVER_VIRTUAL", "DNS_SERVER", "DOMAIN_CONTROLLER",
        "APPLICATION_SERVICE", "SERVER_VIRTUAL", "APPLICATION_SERVICE",
        "MONITORING_SERVER", "SERVER_VIRTUAL", "APPLICATION_SERVICE",
        "SERVER_VIRTUAL", "CORE_SWITCH",
    ],
}


class SMDAdapter(TelemetryAdapter):
    """Loads SMD data and produces SystemSnapshots tick-by-tick.

    Usage:
        adapter = SMDAdapter("/path/to/ServerMachineDataset")
        adapter.load(split="test")  # Load test data with labels

        for tick in range(adapter.n_ticks):
            snapshot = adapter.get_tick(tick)
            # snapshot.nodes has 28 machines with health/state scored
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.machines: list[str] = []
        self.data: dict[str, np.ndarray] = {}  # machine_name -> (T, 38)
        self.labels: dict[str, np.ndarray] = {}  # machine_name -> (T,)
        self.n_ticks = 0
        self._edges: list[EdgeSnapshot] = []

    def load(self, split: str = "test", max_ticks: int = 0):
        """Load SMD data for all machines.

        Args:
            split: "train" or "test"
            max_ticks: limit number of timesteps (0 = all)
        """
        data_path = self.data_dir / split
        label_path = self.data_dir / "test_label"

        self.machines = sorted([f.stem for f in data_path.glob("machine-*.txt")])

        for name in self.machines:
            d = np.loadtxt(str(data_path / f"{name}.txt"), delimiter=",")
            if max_ticks > 0:
                d = d[:max_ticks]
            self.data[name] = d

            if split == "test" and (label_path / f"{name}.txt").exists():
                l = np.loadtxt(str(label_path / f"{name}.txt"), delimiter=",")
                if max_ticks > 0:
                    l = l[:max_ticks]
                self.labels[name] = l

        self.n_ticks = min(d.shape[0] for d in self.data.values())
        self._edges = self._build_topology()

    def poll(self) -> SystemSnapshot:
        """Not used for offline data - use get_tick() instead."""
        return self.get_tick(0)

    def discover_topology(self) -> list[EdgeSnapshot]:
        return self._edges

    def get_tick(self, tick: int) -> SystemSnapshot:
        """Get a SystemSnapshot for a specific timestep.

        Health is derived from the 38 feature dimensions:
        - Composite health = 1.0 - weighted average of feature groups
        - State thresholds: healthy > 0.7, degraded > 0.3, failed <= 0.3
        """
        nodes = {}
        for i, name in enumerate(self.machines):
            features = self.data[name][tick]  # (38,)
            group = name.split("-")[1]
            idx_in_group = int(name.split("-")[2]) - 1

            # Component type from group mapping
            type_list = GROUP_TYPE_MAP.get(group, ["SERVER_PHYSICAL"])
            comp_type = type_list[idx_in_group % len(type_list)]

            # Compute per-group metric averages
            cpu_avg = features[FEATURE_GROUPS["cpu"]].mean()
            mem_avg = features[FEATURE_GROUPS["memory"]].mean()
            net_avg = features[FEATURE_GROUPS["network"]].mean()
            disk_avg = features[FEATURE_GROUPS["disk"]].mean()
            sys_avg = features[FEATURE_GROUPS["system"]].mean()

            # Health: inverse of weighted metric average
            # SMD features are normalized [0,1] where higher = more utilization
            # Weight CPU and memory higher than disk/network
            weighted = (
                cpu_avg * 1.5 +
                mem_avg * 1.3 +
                disk_avg * 0.8 +
                net_avg * 0.7 +
                sys_avg * 1.0
            ) / 5.3
            health = max(0.0, min(1.0, 1.0 - weighted * 2.5))

            # State classification
            if health >= 0.7:
                state = "healthy"
            elif health >= 0.3:
                state = "degraded"
            else:
                state = "failed"

            # Check if this is a labeled anomaly
            is_anomaly = False
            if name in self.labels and tick < len(self.labels[name]):
                is_anomaly = self.labels[name][tick] > 0

            # Override state if anomaly label says so and health is borderline
            if is_anomaly and state == "healthy" and health < 0.85:
                state = "degraded"
                health = min(health, 0.5)

            metrics = {
                "cpu_utilization": float(cpu_avg * 100),
                "mem_used_pct": float(mem_avg * 100),
                "disk_io_util": float(disk_avg * 100),
                "net_bandwidth_util": float(net_avg * 100),
                "system_load": float(sys_avg * 100),
            }

            nodes[name] = NodeSnapshot(
                node_id=name,
                component_type=comp_type,
                metrics=metrics,
                health=health,
                state=state,
                timestamp=float(tick),
                labels={"group": group, "anomaly": str(is_anomaly)},
            )

        return SystemSnapshot(
            nodes=nodes,
            edges=self._edges,
            timestamp=float(tick),
            source="smd",
        )

    def get_ground_truth(self, tick: int) -> dict[str, bool]:
        """Get anomaly labels for a specific tick."""
        gt = {}
        for name in self.machines:
            if name in self.labels and tick < len(self.labels[name]):
                gt[name] = bool(self.labels[name][tick] > 0)
            else:
                gt[name] = False
        return gt

    def _build_topology(self) -> list[EdgeSnapshot]:
        """Build infrastructure topology from machine groups.

        Group 1 (physical) provides hosting for Group 2 (VMs).
        Group 2 (apps) depends on Group 3 (services).
        Intra-group machines share network paths.
        """
        edges = []
        by_group: dict[str, list[str]] = {}
        for name in self.machines:
            group = name.split("-")[1]
            by_group.setdefault(group, []).append(name)

        # Intra-group: network path dependencies (mesh within group)
        for group, members in by_group.items():
            for i, a in enumerate(members):
                for b in members[i + 1:i + 3]:  # Connect to next 2 in group
                    edges.append(EdgeSnapshot(
                        source_id=a, target_id=b,
                        dep_type="NETWORK_PATH",
                        criticality=Criticality.SOFT,
                    ))

        # Group 2 VMs hosted on Group 1 physical servers
        g1 = by_group.get("1", [])
        g2 = by_group.get("2", [])
        for i, vm in enumerate(g2):
            host = g1[i % len(g1)] if g1 else None
            if host:
                edges.append(EdgeSnapshot(
                    source_id=vm, target_id=host,
                    dep_type="HOSTING_DEPENDENCY",
                    criticality=Criticality.HARD,
                ))

        # Group 2 apps depend on Group 3 services
        g3 = by_group.get("3", [])
        for i, app in enumerate(g2):
            svc = g3[i % len(g3)] if g3 else None
            if svc:
                edges.append(EdgeSnapshot(
                    source_id=app, target_id=svc,
                    dep_type="SERVICE_DEPENDENCY",
                    criticality=Criticality.SOFT,
                ))

        # Group 1 depends on Group 3 DNS/DC
        for phys in g1:
            if g3:
                edges.append(EdgeSnapshot(
                    source_id=phys, target_id=g3[1 % len(g3)],  # DNS
                    dep_type="DNS_DEPENDENCY",
                    criticality=Criticality.SOFT,
                ))

        return edges
