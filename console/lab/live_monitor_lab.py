#!/usr/bin/env python3
"""Run SABLE's live monitor against the lab with an EXPLICIT PrometheusConfig.

    python3 console/lab/live_monitor_lab.py [--config console/lab/sable_prometheus.yaml] [--dry-run]

Why this exists: docker/live_monitor.py builds `PrometheusAdapter(PrometheusConfig(url=...))`
(:112) and its config file carries no node_map or queries, so run plainly it auto-discovers
node ids such as `app_app` (adapters/prometheus.py:_make_node_id) that match nothing in
topology.yaml. This wrapper reads the same YAML, builds the PrometheusConfig from its
`prometheus:` block, and swaps it into an otherwise unchanged LiveMonitor. SABLE's code is
not modified; the adapter, scorer, encoder and /api/live_tick path are the stock ones.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # the SABLE checkout
for p in (ROOT, ROOT / "docker", ROOT / "adapters", ROOT / "pillar1", ROOT / "pillar3"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEFAULT_CONFIG = HERE / "sable_prometheus.yaml"


def load_lab_config(path: str | Path) -> dict[str, Any]:
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_prometheus_config(cfg: dict[str, Any]) -> "PrometheusConfig":
    """`cfg` is the whole YAML; its `prometheus:` block becomes the PrometheusConfig.
    Imports lazily: `adapters` pulls in numpy through adapters/__init__.py."""
    from adapters.prometheus import NodeMapping, PrometheusConfig
    block = dict(cfg.get("prometheus") or {})
    url = block.get("url") or cfg.get("prometheus_url") or "http://localhost:9090"
    node_map: dict[str, NodeMapping] = {}
    for instance, m in (block.get("node_map") or {}).items():
        node_map[str(instance)] = NodeMapping(
            node_id=str(m["node_id"]),
            component_type=str(m["component_type"]),
            instance=str(instance),
            job=str(m.get("job", "")),
            label_filters=dict(m.get("label_filters") or {}),
        )
    kwargs: dict[str, Any] = {
        "url": str(url),
        "timeout": float(block.get("timeout", 10.0)),
        "node_map": node_map,
        "auto_discover": bool(block.get("auto_discover", not node_map)),
    }
    if block.get("queries"):
        kwargs["queries"] = {str(k): str(v) for k, v in block["queries"].items()}
    return PrometheusConfig(**kwargs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SABLE live monitor with the lab's explicit Prometheus mapping")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--dry-run", action="store_true", help="poll, score and encode; do not post to SABLE")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    import logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    import live_monitor                                   # docker/live_monitor.py (numpy, adapters)
    from adapters.prometheus import PrometheusAdapter

    cfg = load_lab_config(args.config)
    # inside the console container Prometheus and SABLE are not on localhost
    if os.environ.get("PROMETHEUS_URL"):
        cfg["prometheus_url"] = os.environ["PROMETHEUS_URL"]
        cfg.setdefault("prometheus", {})["url"] = os.environ["PROMETHEUS_URL"]
    if os.environ.get("SABLE_URL"):
        cfg["sable_url"] = os.environ["SABLE_URL"]
    prom_cfg = build_prometheus_config(cfg)
    topology = cfg.get("topology") or "topology.yaml"
    if not Path(topology).is_absolute():
        topology = str(Path(args.config).resolve().parent / topology)

    monitor = live_monitor.LiveMonitor(
        prometheus_url=prom_cfg.url,
        sable_url=str(cfg.get("sable_url", "http://localhost:8080")),
        poll_interval=float(cfg.get("poll_interval", 5)),
        topology_path=topology,
        health_config=live_monitor.build_health_config(cfg.get("health_overrides")),
        dry_run=bool(cfg.get("dry_run", False) or args.dry_run),
    )
    monitor.prom = PrometheusAdapter(prom_cfg)             # the only difference from stock
    print(f"  Node map:      {len(prom_cfg.node_map)} explicit instances, "
          f"auto_discover={prom_cfg.auto_discover}, queries={sorted(prom_cfg.queries)}")

    signal.signal(signal.SIGINT, lambda *_: monitor.stop())
    signal.signal(signal.SIGTERM, lambda *_: monitor.stop())
    monitor.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
