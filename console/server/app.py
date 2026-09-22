"""FastAPI translation of console/docs/API.md over a Loop."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from console.contracts import Receipt
from console.planner import NoTemplate, PlanValidationError

from .loop import Loop

UI_DIST = Path(__file__).resolve().parent.parent / "ui" / "dist"


def _err(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


class Broadcaster:
    """Fan events out to every WebSocket; emit() is safe from any thread."""

    def __init__(self):
        self.clients: set[asyncio.Queue] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def emit(self, ev: dict) -> None:
        if self.loop is None or self.loop.is_closed():
            return
        def _put():
            for q in list(self.clients):
                if q.qsize() < 1000:
                    q.put_nowait(ev)
        try:
            self.loop.call_soon_threadsafe(_put)
        except RuntimeError:
            pass


def receipt_markdown(r: Receipt) -> str:
    approvals = [f"- {a.timestamp.isoformat()} **{a.actor}** {a.decision}" for a in r.approvals] or ["- none"]
    steps = [f"| {s.step_id} | {s.status} | {s.started_at or ''} | {s.ended_at or ''} | "
             f"`{(s.output_digest or '')[:16]}` | `{s.session_id or ''}` |" for s in r.steps]
    snapshots = [f"- {s.node_id} → `{s.snapshot_ref}`" for s in r.snapshots] or ["- none"]
    if r.verification:
        verification = [f"- status: **{r.verification.status}**", f"- checked: {r.verification.checked_at.isoformat()}",
                        "- observed:", "```json", json.dumps(r.verification.observed, indent=2), "```"]
    else:
        verification = ["- not verified"]
    rollback = [f"- performed: {r.rollback.performed}"] + [
        f"- {s.step_id}: {s.action_id} → {s.status}{(' (' + s.error + ')') if s.error else ''}" for s in r.rollback.steps]
    lines = [f"# Receipt {r.id}", "", f"- plan: `{r.plan_id}`", f"- finding: `{r.finding_id}`",
             f"- session: `{r.session_id}`", f"- created: {r.created_at.isoformat()}", "",
             "## Approvals", *approvals, "",
             "## Steps", "| step | status | started | ended | output sha256 | session |", "|---|---|---|---|---|---|",
             *steps, "",
             "## Snapshots", *snapshots, "",
             "## Verification", *verification, "",
             "## Rollback", *rollback, "",
             "## Chain", f"- prev: `{r.prev_receipt_hash}`", f"- hash: `{r.receipt_hash}`",
             f"- audit_ref: `{json.dumps(r.audit_ref) if r.audit_ref else 'none'}`", ""]
    return "\n".join(lines)


def create_app(loop: Loop, broadcaster: Optional[Broadcaster] = None, serve_ui: bool = True,
               ui_dist: Optional[Path] = None) -> FastAPI:
    bc = broadcaster or Broadcaster()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bc.loop = asyncio.get_running_loop()
        stop, feed = asyncio.Event(), None
        if loop.sable is not None and loop.emitter is not None:
            async def _feed():
                try:
                    await loop.sable.subscribe_ticks(lambda tick: loop.on_tick(tick), stop=stop)
                except Exception as e:                    # noqa: BLE001 — SABLE may be down; the API still serves
                    loop.note(f"SABLE tick stream ended: {e}", level="warn")
            feed = asyncio.create_task(_feed())
        try:
            yield
        finally:
            stop.set()
            if feed is not None:
                feed.cancel()

    app = FastAPI(title="OVERLORD × SABLE Console", version="0.1.0", lifespan=lifespan)
    inner_emit = loop._emit
    loop._emit = lambda ev: (inner_emit(ev), bc.emit(ev))
    app.state.loop, app.state.bc = loop, bc

    # ---------------------------------------------------------------- REST

    @app.get("/api/state")
    async def state():
        return loop.state()

    @app.get("/api/findings")
    async def findings(status: Optional[str] = None):
        try:
            rows = loop.store.list(status) if status else loop.store.list()
        except ValueError as e:
            return _err(400, str(e))
        return [f.model_dump(mode="json") for f in rows]

    @app.get("/api/findings/{fid}")
    async def finding(fid: str):
        f = loop.store.get(fid)
        return f.model_dump(mode="json") if f else _err(404, f"no finding {fid}")

    @app.get("/api/findings/{fid}/plan")
    async def finding_plan(fid: str):
        if loop.store.get(fid) is None:
            return _err(404, f"no finding {fid}")
        p = loop.plan_for(fid)
        return p if p else _err(404, f"no plan for {fid}")

    @app.post("/api/findings/{fid}/plan")
    async def replan(fid: str, request: Request):
        body = await request.json() if await request.body() else {}
        if loop.store.get(fid) is None:
            return _err(404, f"no finding {fid}")
        try:
            plan = await asyncio.to_thread(loop.plan, fid, body.get("planner", "template"), body.get("force_action"))
        except PlanValidationError as e:
            return JSONResponse({"error": "plan rejected", "reasons": list(getattr(e, "reasons", [str(e)]))}, status_code=422)
        except NoTemplate as e:
            return _err(422, f"no template: {e}")
        return loop.plan_for(fid) or plan.model_dump(mode="json")

    @app.get("/api/plans/{pid}")
    async def plan(pid: str):
        st = loop.plans.get(pid)
        return loop.plan_for(st.finding_id) if st else _err(404, f"no plan {pid}")

    @app.post("/api/plans/{pid}/approve")
    async def approve(pid: str, request: Request):
        body = await request.json()
        actor, decision = str(body.get("actor") or "operator"), str(body.get("decision") or "")
        if decision not in ("approve", "reject", "hold", "execute_now"):
            return _err(400, "decision must be approve | reject | hold | execute_now")
        try:
            return loop.approve(pid, actor, decision)
        except KeyError:
            return _err(404, f"no plan {pid}")

    @app.get("/api/receipts")
    async def receipts(finding_id: Optional[str] = None):
        return [r.model_dump(mode="json") for r in loop.chain.list(finding_id)]

    @app.get("/api/receipts/verify")
    async def receipts_verify():
        return loop.chain.verify()

    @app.get("/api/receipts/{rid}")
    async def receipt(rid: str):
        r = loop.chain.get(rid)
        return r.model_dump(mode="json") if r else _err(404, f"no receipt {rid}")

    @app.get("/api/receipts/{rid}/export")
    async def receipt_export(rid: str, format: str = "json"):
        r = loop.chain.get(rid)
        if r is None:
            return _err(404, f"no receipt {rid}")
        if format == "md":
            return PlainTextResponse(receipt_markdown(r), media_type="text/markdown",
                                     headers={"Content-Disposition": f'attachment; filename="{rid}.md"'})
        return JSONResponse(r.model_dump(mode="json"),
                            headers={"Content-Disposition": f'attachment; filename="{rid}.json"'})

    @app.get("/api/policy")
    async def policy():
        return loop.policy.model_dump()

    @app.put("/api/policy")
    async def put_policy(request: Request):
        body = await request.json()
        actor = str(body.pop("_actor", "operator")) if isinstance(body, dict) else "operator"
        try:
            return loop.set_policy(body, actor=actor).model_dump()
        except Exception as e:                        # noqa: BLE001 — a bad policy is a 422 with the reason
            return _err(422, str(e))

    @app.get("/api/topology")
    async def topology():
        return {"name": loop.topology.name, "nodes": list(loop.topology.nodes.values()), "edges": loop.topology.edges}

    @app.get("/api/log")
    async def log(limit: int = 200):
        return list(loop.log)[-limit:]

    @app.post("/api/analyst/chat")
    async def analyst_chat(request: Request):
        body = await request.json()
        try:
            return await asyncio.to_thread(loop.analyst_chat, body.get("finding_id"), str(body.get("message") or ""),
                                           list(body.get("history") or []))
        except Exception as e:                        # noqa: BLE001
            return _err(502, f"analyst unavailable: {e}")

    @app.post("/api/lab/fault")
    async def lab_fault(request: Request):
        body = await request.json()
        try:
            return await asyncio.to_thread(loop.lab_fault, str(body.get("name") or ""), list(body.get("args") or []))
        except PermissionError as e:
            return _err(403, str(e))
        except KeyError as e:
            return _err(404, str(e).strip("'\""))
        except ValueError as e:
            return _err(400, str(e))

    # ---------------------------------------------------------------- WS

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue()
        bc.clients.add(q)
        try:
            await sock.send_json({"type": "hello", "state": loop.state()})
            while True:
                ev = await q.get()
                await sock.send_json(ev)
        except WebSocketDisconnect:
            pass
        finally:
            bc.clients.discard(q)

    dist = ui_dist or UI_DIST
    if serve_ui and dist.is_dir():
        # the UI is a single-page app with pushState routes (/receipts/:id):
        # a real file is served as itself, anything else gets index.html
        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            target = (dist / path).resolve() if path else dist / "index.html"
            if path and target.is_file() and str(target).startswith(str(dist.resolve())):
                return FileResponse(str(target))
            return FileResponse(str(dist / "index.html"))
    return app
