"""The console's Finding routes (PLAN.md Phase 2).

    GET  /findings?status=open|closed|reopened   newest first
    GET  /findings/{id}                          404 -> {"error": ...}
    WS   /findings/stream                        {"type":"finding","finding":{...}} on every
                                                 store change (upsert / close / reopen)

The router reads the store from `app.state.store`; `create_app(store)` wires
one up. The store notifies subscribers synchronously, from whatever thread
wrote to it; the websocket bridges that into its own loop with
`loop.call_soon_threadsafe` onto an asyncio.Queue.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from console.contracts import Finding, FindingStatus

from .store import FindingStore

router = APIRouter()


def _store(request_or_ws: Any) -> FindingStore:
    return request_or_ws.app.state.store


@router.get("/findings")
async def list_findings(request: Request, status: Optional[str] = None) -> Any:
    if status:
        try:
            FindingStatus(status)
        except ValueError:
            return JSONResponse({"error": f"unknown status {status!r}"}, status_code=400)
    return [f.model_dump(mode="json") for f in _store(request).list(status or None)]


@router.get("/findings/{finding_id}")
async def get_finding(request: Request, finding_id: str) -> Any:
    f = _store(request).get(finding_id)
    if f is None:
        return JSONResponse({"error": f"finding {finding_id} not found"}, status_code=404)
    return f.model_dump(mode="json")


@router.websocket("/findings/stream")
async def findings_stream(ws: WebSocket) -> None:
    store = _store(ws)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def on_change(finding: Finding) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, finding.model_dump(mode="json"))

    store.subscribe(on_change)          # before accept(): nothing written after the
    try:                                # handshake completes can be missed
        await ws.accept()
        recv = asyncio.ensure_future(ws.receive())
        try:
            while True:
                nxt = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait({recv, nxt}, return_when=asyncio.FIRST_COMPLETED)
                if recv in done:        # client spoke or went away
                    nxt.cancel()
                    msg = recv.result()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    recv = asyncio.ensure_future(ws.receive())
                if nxt in done:
                    await ws.send_json({"type": "finding", "finding": nxt.result()})
        finally:
            recv.cancel()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        store.unsubscribe(on_change)


def create_app(store: FindingStore) -> FastAPI:
    app = FastAPI(title="console findings", version="0.1")
    app.state.store = store
    app.include_router(router)
    return app
