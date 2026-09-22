"""FindingEmitter: SABLE ticks in, Findings into the store.

    on_tick(tick)      GET /api/recommendations, map, dedup against the open
                       finding with the same key, upsert.
    run(stop_event)    subscribe to SABLE's /ws and feed every tick/live_tick
                       to on_tick, reconnecting with backoff until stopped.
    poll_once(tick)    refresh topology + engine version, then on_tick — for
                       tests and a CLI. SABLE has no "latest tick" GET, so the
                       tick must be supplied (from POST /api/tick, /api/live_tick
                       or a fixture).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Optional

from console.contracts import Finding
from console.topology import Topology

from .client import SableError
from .mapper import MappingError, dedup_key_for, to_finding
from .store import FindingStore

log = logging.getLogger(__name__)


class FindingEmitter:
    def __init__(self, client: Any, store: FindingStore, site_id: str,
                 summarize: Optional[Callable[[Finding], str]] = None,
                 topology: Optional[Topology] = None, engine_sha: Optional[str] = None,
                 reconnect_delay: float = 2.0):
        self.client = client
        self.store = store
        self.site_id = site_id
        self.summarize = summarize
        self.topology = topology
        self.engine_sha = engine_sha or os.getenv("SABLE_GIT_SHA")
        self.reconnect_delay = reconnect_delay
        self._engine_version: Optional[str] = None
        self.ticks_seen = 0
        self.last_error: Optional[str] = None

    # ---------------------------------------------------------------- context

    def refresh_topology(self) -> Topology:
        self.topology = Topology.from_api(self.client.topology())
        return self.topology

    def engine_version(self, refresh: bool = False) -> str:
        """`sable@<sha> engine/<version> device/<cpu|cuda>`. The sha comes from
        the constructor / SABLE_GIT_SHA or a `git_sha` key if /api/status ever
        publishes one (it does not today, D5); the version is FastAPI's
        (server.py:24) via /openapi.json; the device is get_summary()["device"]."""
        if self._engine_version is None or refresh:
            status = self.client.status()
            sha = self.engine_sha or status.get("git_sha") or status.get("sha") or "unknown"
            version = "unknown"
            info = getattr(self.client, "version_info", None)
            if callable(info):
                version = str((info() or {}).get("version") or version)
            device = str(status.get("device") or "unknown")
            self._engine_version = f"sable@{sha} engine/{version} device/{device}"
        return self._engine_version

    # ---------------------------------------------------------------- ticks

    def on_tick(self, tick: dict[str, Any]) -> Optional[Finding]:
        """Map one tick to a Finding and store it. Returns the stored Finding
        (new or updated) or None when there is nothing to report."""
        self.ticks_seen += 1
        if self.topology is None or len(self.topology.nodes) != len(tick.get("nodes", [])):
            self.refresh_topology()
        recs = self.client.recommendations()
        try:
            key = dedup_key_for(tick, recs, self.topology, self.site_id)
        except MappingError as e:
            self.last_error = str(e)
            log.warning("tick %s: %s", tick.get("cycle"), e)
            return None
        if key is None:
            return None
        previous = self.store.find_open(key)
        finding = to_finding(tick, recs, self.topology, self.site_id, self.engine_version(),
                             previous=previous, summarize=self._summarizer_for(previous, tick))
        if finding is None:
            return None
        return self.store.upsert(finding)

    def _summarizer_for(self, previous: Optional[Finding], tick: dict[str, Any]):
        """Summaries can be model-written and cost a call each: generate one
        for a new finding, and again only when the affected set changed."""
        if self.summarize is None:
            return None
        if previous is None:
            return self.summarize
        now = {(int(n["id"]), n.get("state")) for n in tick.get("nodes", []) if n.get("state") != "healthy"}
        before = {(self.topology.index_of(a.node_id), a.state) for a in previous.affected_nodes}
        before.add((self.topology.index_of(previous.root_cause.node_id), previous.root_cause.state))
        if now == before and previous.summary_generated:
            return lambda _f: previous.summary       # unchanged condition: keep the generated text
        return self.summarize

    def poll_once(self, tick: dict[str, Any]) -> Optional[Finding]:
        self.refresh_topology()
        self.engine_version(refresh=True)
        return self.on_tick(tick)

    # ---------------------------------------------------------------- loop

    async def run(self, stop_event: asyncio.Event) -> None:
        """Consume SABLE's WebSocket until `stop_event` is set. HTTP work per
        tick runs in a thread so the loop stays responsive."""
        async def handle(msg: dict[str, Any]) -> None:
            try:
                await asyncio.to_thread(self.on_tick, msg)
            except (SableError, MappingError) as e:
                self.last_error = str(e)
                log.warning("tick %s not emitted: %s", msg.get("cycle"), e)

        delay = self.reconnect_delay
        while not stop_event.is_set():
            try:
                await self.client.subscribe_ticks(handle, stop=stop_event)
                delay = self.reconnect_delay          # clean close: reset backoff
            except SableError as e:
                self.last_error = str(e)
                log.warning("SABLE websocket: %s (retry in %.1fs)", e, delay)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                delay = min(delay * 2, 30.0)
