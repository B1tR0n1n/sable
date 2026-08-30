"""
SABLE Telemetry Adapter — Alibaba Cluster Trace 2018
======================================================
Loads Alibaba's public production trace (~4000 machines, 8 days) into
SABLE's SystemSnapshot format. Unlike SMD, this dataset provides REAL
topology via container→machine assignments and app_du grouping, plus
real machine lifecycle/failure events.

Source: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018

Required files (download separately, see datasets/download_alibaba.sh):
  machine_meta.csv      ~92KB    lifecycle events + failure_domain topology
  machine_usage.csv     ~1.7GB   per-machine telemetry (CPU/mem/net/disk)
  container_meta.csv    ~2.4MB   container→machine + app_du grouping

Mapping decisions:
  Nodes:    machines only (containers explode node count to ~70K)
  Topology: edges from (a) shared app_du co-location → SERVICE_DEPENDENCY
                       (b) shared failure_domain_2  → POWER_DEPENDENCY
                       (c) shared failure_domain_1  → NETWORK_PATH
  Failures: machine_meta.status transitions out of "alive"/working state
  Health:   composite of cpu/mem/disk/net utilization
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .base import (
    Criticality,
    EdgeSnapshot,
    NodeSnapshot,
    SystemSnapshot,
    TelemetryAdapter,
)


# Alibaba machine_meta.status values observed in the public trace.
# Anything other than these "live" states counts as a failure event.
HEALTHY_STATUSES = {"add", "alive", ""}


class AlibabaAdapter(TelemetryAdapter):
    """Streams Alibaba 2018 cluster trace as SABLE SystemSnapshots.

    Usage:
        adapter = AlibabaAdapter("/data/alibaba_2018")
        adapter.load(tick_seconds=300, max_machines=500)
        for tick in range(adapter.n_ticks):
            snap = adapter.get_tick(tick)
            gt   = adapter.get_ground_truth(tick)
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.tick_seconds: int = 300
        self.machines: list[str] = []
        self.usage: pd.DataFrame | None = None
        self.events: pd.DataFrame | None = None
        self.containers: pd.DataFrame | None = None
        self._edges: list[EdgeSnapshot] = []
        self._tick_starts: np.ndarray = np.array([])
        self.n_ticks: int = 0

    # ── Loading ──────────────────────────────────────────────────────────

    def load(
        self,
        tick_seconds: int = 300,
        max_machines: int = 0,
        max_ticks: int = 0,
    ) -> None:
        """Load CSVs and bucket telemetry into fixed-width ticks.

        Args:
            tick_seconds: seconds per SABLE tick (300 = 5 min, matches Prom default)
            max_machines: cap nodes for fast iteration (0 = all ~4000)
            max_ticks: cap simulated time horizon (0 = full 8 days = ~2300 ticks)
        """
        self.tick_seconds = tick_seconds

        meta = pd.read_csv(
            self.data_dir / "machine_meta.csv",
            header=None,
            names=[
                "machine_id", "time_stamp", "failure_domain_1",
                "failure_domain_2", "cpu_num", "mem_size", "status",
            ],
            dtype={"machine_id": "string", "status": "string",
                   "failure_domain_2": "string"},
        )

        usage = pd.read_csv(
            self.data_dir / "machine_usage.csv",
            header=None,
            names=[
                "machine_id", "time_stamp", "cpu_util_percent",
                "mem_util_percent", "mem_gps", "mkpi",
                "net_in", "net_out", "disk_io_percent",
            ],
            dtype={"machine_id": "string"},
        )

        containers = pd.read_csv(
            self.data_dir / "container_meta.csv",
            header=None,
            names=[
                "container_id", "machine_id", "time_stamp", "app_du",
                "status", "cpu_request", "cpu_limit", "mem_size",
            ],
            dtype={"container_id": "string", "machine_id": "string",
                   "app_du": "string", "status": "string"},
        )

        # Subsample machines for development runs
        if max_machines > 0:
            keep = sorted(usage["machine_id"].unique())[:max_machines]
            usage = usage[usage["machine_id"].isin(keep)]
            meta = meta[meta["machine_id"].isin(keep)]
            containers = containers[containers["machine_id"].isin(keep)]

        self.machines = sorted(usage["machine_id"].unique().tolist())
        self.usage = usage.sort_values("time_stamp").reset_index(drop=True)
        self.events = meta.sort_values("time_stamp").reset_index(drop=True)
        self.containers = containers

        t_max = float(self.usage["time_stamp"].max())
        n = int(t_max // tick_seconds) + 1
        if max_ticks > 0:
            n = min(n, max_ticks)
        self.n_ticks = n
        self._tick_starts = np.arange(n) * tick_seconds

        # Pre-bucket usage by tick for O(1) get_tick()
        self.usage["tick"] = (self.usage["time_stamp"] // tick_seconds).astype(int)
        self._usage_by_tick = {
            t: g for t, g in self.usage.groupby("tick") if t < n
        }

        self._edges = self._build_topology()

    # ── Adapter interface ────────────────────────────────────────────────

    def poll(self) -> SystemSnapshot:
        return self.get_tick(0)

    def discover_topology(self) -> list[EdgeSnapshot]:
        return self._edges

    def get_tick(self, tick: int) -> SystemSnapshot:
        if tick >= self.n_ticks:
            raise IndexError(f"tick {tick} >= n_ticks {self.n_ticks}")

        df = self._usage_by_tick.get(tick, pd.DataFrame(columns=self.usage.columns))
        # Average each metric per machine within the tick window
        agg = df.groupby("machine_id").agg({
            "cpu_util_percent": "mean",
            "mem_util_percent": "mean",
            "net_in": "mean",
            "net_out": "mean",
            "disk_io_percent": "mean",
        }).to_dict("index")

        nodes: dict[str, NodeSnapshot] = {}
        ts = float(self._tick_starts[tick])
        failed = self._failures_active_at(ts)

        for mid in self.machines:
            m = agg.get(mid, {})
            cpu = _clean(m.get("cpu_util_percent"))
            mem = _clean(m.get("mem_util_percent"))
            disk = _clean(m.get("disk_io_percent"))
            net_in = _clean(m.get("net_in"))
            net_out = _clean(m.get("net_out"))

            reachable = mid in agg
            health = _composite_health(cpu, mem, disk, max(net_in, net_out))

            if mid in failed:
                state = "failed"
                health = min(health, 0.1)
            elif not reachable:
                state = "unreachable"
                health = 0.0
            elif health >= 0.7:
                state = "healthy"
            elif health >= 0.3:
                state = "degraded"
            else:
                state = "failed"

            nodes[mid] = NodeSnapshot(
                node_id=mid,
                component_type="SERVER_PHYSICAL",
                metrics={
                    "cpu_utilization": cpu,
                    "mem_used_pct": mem,
                    "disk_io_util": disk,
                    "net_in_pct": net_in,
                    "net_out_pct": net_out,
                },
                health=health,
                state=state,
                reachable=reachable,
                timestamp=ts,
                labels={"alibaba_status": "failed" if mid in failed else "alive"},
            )

        return SystemSnapshot(
            nodes=nodes, edges=self._edges, timestamp=ts, source="alibaba_2018",
        )

    # ── Ground truth ─────────────────────────────────────────────────────

    def get_ground_truth(self, tick: int) -> dict[str, bool]:
        """Return {machine_id: failed} for the tick.

        A machine is considered failed at tick T if the most recent
        machine_meta row at or before T*tick_seconds has a non-healthy status.
        """
        ts = float(self._tick_starts[tick])
        failed = self._failures_active_at(ts)
        return {mid: (mid in failed) for mid in self.machines}

    def iter_ticks(self) -> Iterator[tuple[int, SystemSnapshot, dict[str, bool]]]:
        for t in range(self.n_ticks):
            yield t, self.get_tick(t), self.get_ground_truth(t)

    # ── Internals ────────────────────────────────────────────────────────

    def _failures_active_at(self, ts: float) -> set[str]:
        """Set of machine_ids whose last-seen status before `ts` is non-healthy."""
        if self.events is None or self.events.empty:
            return set()
        recent = self.events[self.events["time_stamp"] <= ts]
        if recent.empty:
            return set()
        latest = recent.groupby("machine_id").tail(1)
        bad = latest[~latest["status"].isin(HEALTHY_STATUSES)]
        return set(bad["machine_id"].tolist())

    def _build_topology(self) -> list[EdgeSnapshot]:
        """Build real topology from container assignments + failure domains.

        Three edge sources, in order of confidence:
          1. SERVICE_DEPENDENCY: machines hosting containers in the same app_du
          2. POWER_DEPENDENCY:   machines sharing failure_domain_2 (rack/row)
          3. NETWORK_PATH:       machines sharing failure_domain_1 (zone)
        """
        edges: list[EdgeSnapshot] = []
        machine_set = set(self.machines)

        # 1. App-DU co-location edges (real workload-level dependency)
        if self.containers is not None and not self.containers.empty:
            by_app = defaultdict(set)
            for _, row in self.containers.iterrows():
                if row["machine_id"] in machine_set and pd.notna(row["app_du"]):
                    by_app[row["app_du"]].add(row["machine_id"])
            for app, mids in by_app.items():
                mids = sorted(mids)
                # Cap fanout per app to avoid n^2 explosion on big apps
                for i, a in enumerate(mids[:50]):
                    for b in mids[i + 1:i + 6]:
                        edges.append(EdgeSnapshot(
                            source_id=a, target_id=b,
                            dep_type="SERVICE_DEPENDENCY",
                            criticality=Criticality.SOFT,
                            confidence=0.9,
                            metadata={"app_du": app},
                        ))

        # 2 + 3. Failure-domain edges (physical topology)
        if self.events is not None and not self.events.empty:
            latest = self.events.groupby("machine_id").tail(1)
            by_fd2 = defaultdict(list)
            by_fd1 = defaultdict(list)
            for _, row in latest.iterrows():
                if row["machine_id"] not in machine_set:
                    continue
                if pd.notna(row["failure_domain_2"]):
                    by_fd2[row["failure_domain_2"]].append(row["machine_id"])
                if pd.notna(row["failure_domain_1"]):
                    by_fd1[row["failure_domain_1"]].append(row["machine_id"])

            for fd, mids in by_fd2.items():
                mids = sorted(mids)
                for i, a in enumerate(mids[:30]):
                    for b in mids[i + 1:i + 4]:
                        edges.append(EdgeSnapshot(
                            source_id=a, target_id=b,
                            dep_type="POWER_DEPENDENCY",
                            criticality=Criticality.HARD,
                            confidence=1.0,
                            metadata={"failure_domain_2": str(fd)},
                        ))

            for fd, mids in by_fd1.items():
                mids = sorted(mids)
                for i, a in enumerate(mids[:30]):
                    for b in mids[i + 1:i + 3]:
                        edges.append(EdgeSnapshot(
                            source_id=a, target_id=b,
                            dep_type="NETWORK_PATH",
                            criticality=Criticality.SOFT,
                            confidence=0.7,
                            metadata={"failure_domain_1": str(fd)},
                        ))

        return edges


# ── Helpers ──────────────────────────────────────────────────────────────


def _clean(x) -> float:
    """Coerce missing / Alibaba-sentinel (-1, 101) values to 0."""
    if x is None:
        return 0.0
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v) or v < 0 or v > 100:
        return 0.0
    return v


def _composite_health(cpu: float, mem: float, disk: float, net: float) -> float:
    """Health = 1 - weighted utilization. CPU/mem weighted higher."""
    weighted = (cpu * 1.5 + mem * 1.3 + disk * 0.8 + net * 0.7) / 4.3
    # Normalize: util 0 → health 1.0, util 100 → health ~0
    return max(0.0, min(1.0, 1.0 - (weighted / 100.0) * 1.5))
