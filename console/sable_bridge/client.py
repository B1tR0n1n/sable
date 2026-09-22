"""A thin client over SABLE's served API (docker/server.py).

Every method returns the plain dict (or list) the route returns and raises
`SableError` on transport or HTTP failure. Nothing here interprets the
payloads — that is `mapper.py`'s job.

Routes, as read in docker/server.py:
    GET  /api/status            :203-218   engine flags + get_summary() (device, cycle, ...)
    GET  /api/recommendations   :299-301   enrich_recommendations(engine.get_recommendations())
    GET  /api/node/{idx}        :290-296   engine.get_node_report(idx) + label/topo_id/component_type
    GET  /api/topology          :319-333   {name, nodes[], edges[]} from the first YAML in TOPOLOGY_DIR
    GET  /api/scenarios         :221-238   only scenarios with source == "smd"
    POST /api/tick              :259-268   one replay cycle -> enriched tick dict
    POST /api/live_tick         :418-495   body {gnn, pomdp, mamba, node_ids, n_nodes, ground_truth}
                                           (docker/live_monitor.py:232-242) -> tick dict, source="live"
    POST /api/mc_dropout        :504-509   {samples} -> {mc_dropout, samples}, clamped 0..20
    WS   /ws                    :338-357   sends {"type":"status",...} on connect, then
                                           {"type":"tick",...} per replay/autoplay cycle,
                                           {"type":"live_tick",...} per live cycle (:490),
                                           {"type":"complete","cycle"} at scenario end (:382)
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Awaitable, Callable, Optional, Union

import httpx

TICK_MESSAGE_TYPES = ("tick", "live_tick")

TickCallback = Callable[[dict[str, Any]], Union[None, Awaitable[None]]]


class SableError(Exception):
    """Transport or HTTP failure talking to SABLE."""

    def __init__(self, message: str, status_code: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class SableClient:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ---------------------------------------------------------------- plumbing

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "SableClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def ws_url(self) -> str:
        scheme = "wss" if self.base_url.startswith("https") else "ws"
        return f"{scheme}://{self.base_url.split('://', 1)[1]}/ws"

    def _request(self, method: str, path: str, json_body: Any = None) -> Any:
        try:
            resp = self._http.request(method, path, json=json_body)
        except httpx.HTTPError as e:
            raise SableError(f"{method} {path}: {e}") from e
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        if resp.status_code >= 400:
            msg = body.get("error") if isinstance(body, dict) else body
            raise SableError(f"{method} {path}: HTTP {resp.status_code}: {msg}",
                             status_code=resp.status_code, body=body)
        return body

    def get(self, path: str) -> Any:
        return self._request("GET", path)

    def post(self, path: str, body: Any = None) -> Any:
        return self._request("POST", path, body if body is not None else {})

    # ---------------------------------------------------------------- routes

    def status(self) -> dict[str, Any]:
        return self.get("/api/status")

    def recommendations(self) -> dict[str, Any]:
        return self.get("/api/recommendations")

    def node(self, idx: int) -> dict[str, Any]:
        return self.get(f"/api/node/{int(idx)}")

    def topology(self) -> dict[str, Any]:
        return self.get("/api/topology")

    def scenarios(self) -> list[dict[str, Any]]:
        return self.get("/api/scenarios")

    def set_scenario(self, name: str) -> dict[str, Any]:
        return self.post("/api/scenario", {"name": name})

    def reset(self) -> dict[str, Any]:
        return self.post("/api/reset")

    def tick(self) -> dict[str, Any]:
        """Advance the loaded replay scenario one cycle. 400 (as SableError)
        when no scenario is loaded or it is complete."""
        return self.post("/api/tick")

    def live_tick(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST pre-encoded pillar inputs, exactly what live_monitor._push_to_engine
        sends: {gnn: [[...]] (N x 1044), pomdp: {node_id: [8 floats]},
        mamba: [[[...]]] (1 x 2 x max_nodes*26), node_ids: [str], n_nodes: int,
        ground_truth: [int]} — returns the tick dict with source="live"."""
        missing = [k for k in ("gnn", "pomdp", "mamba", "node_ids", "n_nodes") if k not in payload]
        if missing:
            raise SableError(f"live_tick payload missing {missing}")
        return self.post("/api/live_tick", payload)

    def mc_dropout(self, samples: int) -> dict[str, Any]:
        return self.post("/api/mc_dropout", {"samples": int(samples)})

    def version_info(self) -> dict[str, Any]:
        """FastAPI's own {title, version} — the only version SABLE publishes
        (server.py:24). Best effort; empty dict when unavailable."""
        try:
            spec = self.get("/openapi.json")
        except SableError:
            return {}
        return dict(spec.get("info", {})) if isinstance(spec, dict) else {}

    # ---------------------------------------------------------------- websocket

    async def subscribe_ticks(self, on_tick: TickCallback,
                              stop: Optional[asyncio.Event] = None,
                              on_message: Optional[TickCallback] = None) -> None:
        """Connect to /ws and call `on_tick(tick_dict)` for every `tick` or
        `live_tick` message until the socket closes or `stop` is set. The
        dict is the message as sent (the `type` key is kept). `on_message`,
        if given, sees every message (status/complete included). Sync or
        async callbacks are accepted. Raises SableError on connection failure."""
        try:
            import websockets
        except ImportError as e:  # pragma: no cover
            raise SableError("the 'websockets' package is required for subscribe_ticks") from e

        try:
            async with websockets.connect(self.ws_url, open_timeout=self.timeout) as ws:
                while not (stop and stop.is_set()):
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    try:
                        msg = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(msg, dict):
                        continue
                    if on_message is not None:
                        await _call(on_message, msg)
                    if msg.get("type") in TICK_MESSAGE_TYPES:
                        await _call(on_tick, msg)
        except websockets.exceptions.ConnectionClosedOK:
            return
        except (OSError, websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
            raise SableError(f"websocket {self.ws_url}: {e}") from e


async def _call(cb: TickCallback, msg: dict[str, Any]) -> None:
    result = cb(msg)
    if inspect.isawaitable(result):
        await result
