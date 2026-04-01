#!/usr/bin/env -S python -u
"""
SABLE Live Monitor
====================
Standalone service that polls a Prometheus instance on a timer, encodes
telemetry into pillar inputs, runs inference through the SABLE engine,
and pushes results to the dashboard via the existing server API.

This is the bridge between real infrastructure and SABLE. Turn it on,
point it at Prometheus, and SABLE starts reasoning about your environment.

Usage:
    # Start the SABLE server first (port 8080)
    python server.py &

    # Then start the live monitor
    python live_monitor.py --prometheus http://prometheus:9090 --interval 30

    # Or with a config file
    python live_monitor.py --config live_monitor.yaml

    # Dry run (poll + encode, don't send to engine)
    python live_monitor.py --prometheus http://prometheus:9090 --dry-run

Config file format (live_monitor.yaml):
    prometheus_url: http://prometheus:9090
    sable_url: http://localhost:8080
    poll_interval: 30
    topology: ../adapters/topologies/msp_demo.yaml
    health_overrides:
      cpu_utilization:
        degraded: 85
        failed: 98
"""

import argparse
import json
import logging
import signal
import sys
import time
import threading
from pathlib import Path

import numpy as np
import requests

# Resolve adapter imports
_here = Path(__file__).parent
_root = _here.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "adapters"))
sys.path.insert(0, str(_root / "pillar1"))
sys.path.insert(0, str(_root / "pillar3"))

from adapters.prometheus import PrometheusAdapter, PrometheusConfig
from adapters.health_scorer import HealthScorer, HealthConfig, ThresholdSet
from adapters.encode import PillarEncoder, MAMBA_MAX_NODES, MAMBA_NODE_FEAT_DIM
from adapters.graph_builder import load_topology_from_file
from adapters.base import SystemSnapshot

log = logging.getLogger("sable.live")

# ── ANSI Colors ──────────────────────────────────────────────────────────

GOLD = "\033[38;2;201;162;39m"
DIM = "\033[38;2;138;127;110m"
RED = "\033[38;2;200;60;60m"
GREEN = "\033[38;2;80;180;80m"
RESET = "\033[0m"
BOLD = "\033[1m"

BANNER = f"""
{GOLD}
  ┌─────────────────────────────────────────┐
  │                                         │
  │   S A B L E    L I V E    M O N I T O R │
  │                                         │
  │   Real-Time Infrastructure Reasoning    │
  │                                         │
  └─────────────────────────────────────────┘
{RESET}"""


# ── Live Monitor ─────────────────────────────────────────────────────────


class LiveMonitor:
    """Polls Prometheus, encodes for SABLE, pushes to engine.

    Runs as a standalone process alongside server.py. Communicates with
    the SABLE engine either via HTTP API or direct engine import.
    """

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        sable_url: str = "http://localhost:8080",
        poll_interval: float = 30.0,
        topology_path: str | None = None,
        health_config: HealthConfig | None = None,
        direct_engine: bool = False,
        dry_run: bool = False,
    ):
        self.prometheus_url = prometheus_url
        self.sable_url = sable_url.rstrip("/")
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self.direct_engine = direct_engine

        # Adapter stack
        self.prom = PrometheusAdapter(PrometheusConfig(url=prometheus_url))
        self.scorer = HealthScorer(health_config)
        self.encoder = PillarEncoder()

        # Topology (edges from YAML/Netbox)
        self.topology_edges = []
        if topology_path:
            self.topology_edges = load_topology_from_file(topology_path)

        # State
        self._prev_snapshot: SystemSnapshot | None = None
        self._running = False
        self._stop_event = threading.Event()
        self._cycle = 0
        self._engine = None  # Direct engine reference (optional)

    def start(self):
        """Start the polling loop. Blocks until stop() is called."""
        self._running = True
        self._stop_event.clear()

        print(BANNER)
        print(f"  Prometheus:    {self.prometheus_url}")
        print(f"  SABLE Engine:  {self.sable_url}")
        print(f"  Poll interval: {self.poll_interval}s")
        print(f"  Topology:      {len(self.topology_edges)} edges")
        print(f"  Dry run:       {self.dry_run}")
        print(f"  Mode:          {'direct' if self.direct_engine else 'HTTP API'}")
        print()

        # Test connectivity
        if not self.dry_run:
            if not self._test_sable():
                log.error("Cannot reach SABLE engine at %s", self.sable_url)
                return
        if not self._test_prometheus():
            log.error("Cannot reach Prometheus at %s", self.prometheus_url)
            return

        print(f"  {GREEN}Live monitor active. Ctrl+C to stop.{RESET}")
        print()

        while not self._stop_event.is_set():
            try:
                self._poll_cycle()
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error("Poll cycle failed: %s", e, exc_info=True)

            self._stop_event.wait(timeout=self.poll_interval)

        self._running = False
        print(f"\n  {DIM}Live monitor stopped.{RESET}")

    def stop(self):
        """Signal the polling loop to stop."""
        self._stop_event.set()

    def _poll_cycle(self):
        """Execute one poll-encode-infer cycle."""
        t0 = time.time()

        # Step 1: Poll Prometheus
        snapshot = self.prom.poll()
        t_poll = time.time() - t0

        if not snapshot.nodes:
            log.warning("No nodes discovered from Prometheus")
            return

        # Step 2: Score health
        for node in snapshot.nodes.values():
            if node.health is None:
                node.health, node.state = self.scorer.score(node)

        # Step 3: Attach topology edges
        snapshot.edges = list(self.topology_edges)

        # Step 4: Encode for pillars
        inputs = self.encoder.encode(snapshot, self._prev_snapshot)
        t_encode = time.time() - t0 - t_poll

        # Step 5: Push to engine (or dry-run print)
        result = None
        t_infer = 0
        if self.dry_run:
            self._print_dry_run(snapshot, inputs)
        else:
            t_infer_start = time.time()
            result = self._push_to_engine(inputs, snapshot)
            t_infer = time.time() - t_infer_start

        # Step 6: Store for next cycle's temporal delta
        self._prev_snapshot = snapshot
        self._cycle += 1

        # Step 7: Print cycle summary
        total = time.time() - t0
        self._print_cycle(snapshot, result, t_poll, t_encode, t_infer, total)

    def _push_to_engine(self, inputs, snapshot) -> dict | None:
        """Send encoded inputs to the SABLE engine via HTTP API."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            gnn_data = inputs.gnn.x.tolist()
            pomdp_data = {nid: v.tolist() for nid, v in inputs.pomdp.beliefs.items()}
            mamba_data = inputs.mamba.x_input.tolist()

            # Build ground truth from health scorer states
            node_ids = inputs.node_ids
            ground_truth = []
            state_map = {"healthy": 0, "degraded": 1, "failed": 2, "unreachable": 3, "oscillating": 4}
            for nid in node_ids:
                node = snapshot.nodes.get(nid)
                if node and node.state:
                    ground_truth.append(state_map.get(node.state, 0))
                else:
                    ground_truth.append(0)

            resp = requests.post(
                f"{self.sable_url}/api/live_tick",
                json={
                    "gnn": gnn_data,
                    "pomdp": pomdp_data,
                    "mamba": mamba_data,
                    "node_ids": node_ids,
                    "n_nodes": inputs.gnn.n_nodes,
                    "ground_truth": ground_truth,
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            log.error("Failed to push to engine: %s", e)
            return None

    def _test_sable(self) -> bool:
        """Test connectivity to the SABLE engine."""
        try:
            resp = requests.get(f"{self.sable_url}/api/status", timeout=5)
            data = resp.json()
            print(f"  SABLE engine:  {GREEN}connected{RESET} ({data.get('device', '?')})")
            return True
        except Exception as e:
            print(f"  SABLE engine:  {RED}unreachable{RESET} ({e})")
            return False

    def _test_prometheus(self) -> bool:
        """Test connectivity to Prometheus."""
        try:
            resp = requests.get(
                f"{self.prometheus_url}/api/v1/status/config",
                timeout=5,
            )
            if resp.status_code == 200:
                # Also count targets
                tresp = requests.get(
                    f"{self.prometheus_url}/api/v1/targets",
                    timeout=5,
                )
                targets = tresp.json().get("data", {}).get("activeTargets", [])
                print(f"  Prometheus:    {GREEN}connected{RESET} ({len(targets)} active targets)")
                return True
            print(f"  Prometheus:    {RED}HTTP {resp.status_code}{RESET}")
            return False
        except Exception as e:
            print(f"  Prometheus:    {RED}unreachable{RESET} ({e})")
            return False

    def _print_dry_run(self, snapshot, inputs):
        """Print encoded data without sending to engine."""
        print(f"\n  {GOLD}--- Dry Run Cycle {self._cycle} ---{RESET}")
        print(f"  Nodes discovered: {snapshot.n_nodes}")
        print(f"  Edges:            {snapshot.n_edges}")
        print(f"  GNN features:     {inputs.gnn.x.shape}")
        print(f"  POMDP beliefs:    {len(inputs.pomdp.beliefs)} nodes")
        print(f"  Mamba input:      {inputs.mamba.x_input.shape}")
        print()

        # State breakdown
        states = {}
        for node in snapshot.nodes.values():
            s = node.state or "unknown"
            states[s] = states.get(s, 0) + 1
        state_str = " | ".join(f"{k}: {v}" for k, v in sorted(states.items()))
        print(f"  Health states: {state_str}")

        # Show degraded/failed nodes
        for node in snapshot.nodes.values():
            if node.state not in ("healthy", None):
                h = node.health if node.health is not None else 0
                print(f"    {RED}[{node.state.upper():>12}]{RESET} {node.node_id:>25} "
                      f"({node.component_type}) health={h:.2f}")
        print()

    def _print_cycle(self, snapshot, result, t_poll, t_encode, t_infer, total):
        """Print one-line cycle summary."""
        n = snapshot.n_nodes
        states = {}
        for node in snapshot.nodes.values():
            s = node.state or "healthy"
            states[s] = states.get(s, 0) + 1

        state_parts = []
        for s in ["healthy", "degraded", "failed", "unreachable", "oscillating"]:
            c = states.get(s, 0)
            if c > 0:
                color = GREEN if s == "healthy" else RED if s in ("failed", "unreachable") else GOLD
                state_parts.append(f"{color}{s[0].upper()}:{c}{RESET}")

        state_str = " ".join(state_parts)

        acc_str = ""
        if result and result.get("accuracy") is not None:
            acc_str = f" | acc={result['accuracy']:.0%}"

        conf_str = ""
        if result and result.get("avg_confidence") is not None:
            conf_str = f" | conf={result['avg_confidence']:.0%}"

        timing = f"poll={t_poll*1000:.0f}ms enc={t_encode*1000:.0f}ms inf={t_infer*1000:.0f}ms"

        print(f"  {DIM}[{self._cycle:>4}]{RESET} {n} nodes | {state_str} "
              f"{acc_str}{conf_str} | {timing} | total={total*1000:.0f}ms")


# ── Config Loading ───────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    """Load live monitor config from YAML file."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_health_config(overrides: dict | None) -> HealthConfig | None:
    """Build HealthConfig with optional per-metric threshold overrides."""
    if not overrides:
        return None

    config = HealthConfig()
    for metric, thresholds in overrides.items():
        if isinstance(thresholds, dict):
            config.thresholds[metric] = ThresholdSet(
                degraded=thresholds.get("degraded", 80),
                failed=thresholds.get("failed", 95),
                invert=thresholds.get("invert", False),
                binary=thresholds.get("binary", False),
            )
    return config


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="SABLE Live Monitor - Real-time infrastructure reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prometheus", default="http://localhost:9090",
                        help="Prometheus URL (default: http://localhost:9090)")
    parser.add_argument("--sable", default="http://localhost:8080",
                        help="SABLE engine URL (default: http://localhost:8080)")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Poll interval in seconds (default: 30)")
    parser.add_argument("--topology", default=None,
                        help="Path to topology YAML file")
    parser.add_argument("--config", default=None,
                        help="Path to config YAML file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Poll and encode only, don't send to engine")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=f"{DIM}%(asctime)s{RESET} %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Config file overrides CLI args
    config = {}
    if args.config:
        config = load_config(args.config)

    prom_url = config.get("prometheus_url", args.prometheus)
    sable_url = config.get("sable_url", args.sable)
    interval = config.get("poll_interval", args.interval)
    topology = config.get("topology", args.topology)
    health_overrides = config.get("health_overrides")
    dry_run = config.get("dry_run", args.dry_run)

    # Resolve topology path relative to config file
    if topology and args.config and not Path(topology).is_absolute():
        topology = str(Path(args.config).parent / topology)

    # Default topology if none specified
    if not topology:
        default_topo = _root / "adapters" / "topologies" / "msp_demo.yaml"
        if default_topo.exists():
            topology = str(default_topo)

    health_config = build_health_config(health_overrides)

    monitor = LiveMonitor(
        prometheus_url=prom_url,
        sable_url=sable_url,
        poll_interval=interval,
        topology_path=topology,
        health_config=health_config,
        dry_run=dry_run,
    )

    # Graceful shutdown
    def handle_signal(sig, frame):
        monitor.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    monitor.start()


if __name__ == "__main__":
    main()
