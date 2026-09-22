#!/usr/bin/env python3
"""Lab key-value service (stdlib only). Runs as `db` (primary) or `db-replica` (replica).

Env: KV_ROLE=primary|replica, KV_PORT=8100, KV_PRIMARY=db.lab:8100 (replica only: the
primary it polls for replication lag; only informational, the replica stays healthy alone).

    GET  /health        200 {"ok": true, "role": ..., "keys": N}
    GET  /metrics       up, kv_role{role=...}, kv_keys, kv_requests_total, kv_replication_ok (replica)
    GET  /kv/<key>      200 value | 404
    PUT  /kv/<key>      store body as value (primary only; 405 on a replica)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROLE = os.environ.get("KV_ROLE", "primary")
PORT = int(os.environ.get("KV_PORT", "8100"))
PRIMARY = os.environ.get("KV_PRIMARY", "")
STORE: dict[str, str] = {"lab": "sable"}
STATS = {"requests_total": 0, "replication_ok": 0}
LOCK = threading.Lock()


def replication_loop() -> None:
    """Replica only: mirror the primary's keys every few seconds (best effort)."""
    while True:
        ok = 0
        try:
            with urllib.request.urlopen(f"http://{PRIMARY}/kv/", timeout=2) as resp:
                data = json.loads(resp.read().decode() or "{}")
            with LOCK:
                STORE.update({k: str(v) for k, v in data.items()})
            ok = 1
        except Exception:
            ok = 0
        with LOCK:
            STATS["replication_ok"] = ok
        time.sleep(5)


class Handler(BaseHTTPRequestHandler):
    server_version = f"sable-lab-kv-{ROLE}/1"

    def log_message(self, fmt, *args):
        if self.path not in ("/metrics", "/health"):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: str, ctype: str = "text/plain; version=0.0.4") -> None:
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        with LOCK:
            STATS["requests_total"] += 1
            keys, stats = dict(STORE), dict(STATS)
        if self.path == "/health":
            self._send(200, json.dumps({"ok": True, "role": ROLE, "keys": len(keys)}) + "\n", "application/json")
        elif self.path == "/metrics":
            lines = [
                "# HELP up 1 while the kv process serves.", "# TYPE up gauge", "up 1",
                "# HELP kv_role The role this instance runs as.", "# TYPE kv_role gauge",
                f'kv_role{{role="{ROLE}"}} 1',
                "# HELP kv_keys Keys held.", "# TYPE kv_keys gauge", f"kv_keys {len(keys)}",
                "# HELP kv_requests_total Requests served.", "# TYPE kv_requests_total counter",
                f"kv_requests_total {stats['requests_total']}",
            ]
            if ROLE == "replica":
                lines += ["# HELP kv_replication_ok 1 when the last pull from the primary succeeded.",
                          "# TYPE kv_replication_ok gauge", f"kv_replication_ok {stats['replication_ok']}"]
            self._send(200, "\n".join(lines) + "\n")
        elif self.path == "/kv/":
            self._send(200, json.dumps(keys) + "\n", "application/json")
        elif self.path.startswith("/kv/"):
            key = self.path[4:]
            if key in keys:
                self._send(200, keys[key] + "\n")
            else:
                self._send(404, "not found\n")
        else:
            self._send(200, f"sable-lab kv role={ROLE}\n")

    def do_PUT(self):
        if not self.path.startswith("/kv/") or len(self.path) <= 4:
            return self._send(404, "not found\n")
        if ROLE != "primary":
            return self._send(405, "read-only replica\n")
        length = int(self.headers.get("Content-Length") or 0)
        value = self.rfile.read(length).decode() if length else ""
        with LOCK:
            STORE[self.path[4:]] = value
            STATS["requests_total"] += 1
        self._send(200, "ok\n")


def main() -> int:
    print(f"[kv] role={ROLE} port={PORT} primary={PRIMARY or '-'}", flush=True)
    if ROLE == "replica" and PRIMARY:
        threading.Thread(target=replication_loop, name="replication", daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
