#!/usr/bin/env python3
"""Lab `app` service (stdlib only).

Reads key=value config from $APP_CONF (config/app.conf):
    db_host=db.lab        the primary's hostname, resolved through the lab's dns container
    db_port=8100          (the container's resolver is dnsmasq, see docker-compose.yml)
    dns_canary=app.lab    a name that always exists: proves the resolver itself works
    check_interval_s=3

A background thread re-checks both dependencies every check_interval_s seconds so that
/health and /metrics answer instantly even while a lookup is timing out.

    GET /health   200 {"ok": true, ...} when dns and db are both fine, else 503
    GET /metrics  Prometheus text: up, app_dependency_ok{dep="dns"|"db"},
                  app_requests_total, app_errors_total, app_db_target_info
    GET /         a short banner

`up 1` is exposed because the lab contract asks for it; note Prometheus synthesises its own
`up` per scrape (1 while this process answers, 0 when it does not) and that is the series
SABLE's adapter reads. A broken dependency therefore shows as app_dependency_ok 0 (mapped
onto SABLE's `healthy` metric), not as up 0: the app is reachable but degraded.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def read_conf(path: str) -> dict[str, str]:
    conf: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                conf[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"[app] config {path} not found; using defaults", flush=True)
    return conf


CONF_PATH = os.environ.get("APP_CONF", "/lab/config/app.conf")
CONF = read_conf(CONF_PATH)
DB_HOST = CONF.get("db_host", "db.lab")
DB_PORT = int(CONF.get("db_port", "8100"))
DNS_CANARY = CONF.get("dns_canary", "app.lab")
CHECK_INTERVAL = float(CONF.get("check_interval_s", "3"))
PORT = int(os.environ.get("APP_PORT", "8000"))

STATE = {
    "dns_ok": False,
    "db_ok": False,
    "db_addr": None,
    "detail": {"dns": "not checked yet", "db": "not checked yet"},
    "checked_at": 0.0,
    "requests_total": 0,
    "errors_total": 0,
}
LOCK = threading.Lock()


def check_dependencies() -> None:
    detail: dict[str, str] = {}
    # dns: can the resolver answer at all? (the canary always exists in lab.hosts)
    try:
        socket.gethostbyname(DNS_CANARY)
        dns_ok, detail["dns"] = True, f"resolved {DNS_CANARY}"
    except OSError as exc:
        dns_ok, detail["dns"] = False, f"cannot resolve {DNS_CANARY}: {exc}"
    # db: resolve the configured host, then call its /health
    db_ok, db_addr = False, None
    try:
        db_addr = socket.gethostbyname(DB_HOST)
        with urllib.request.urlopen(f"http://{db_addr}:{DB_PORT}/health", timeout=2) as resp:
            body = json.loads(resp.read().decode() or "{}")
            db_ok = resp.status == 200 and bool(body.get("ok", True))
            detail["db"] = f"{DB_HOST} -> {db_addr}:{DB_PORT} role={body.get('role', '?')}"
    except OSError as exc:            # includes URLError, timeouts, connection refused, gaierror
        detail["db"] = f"{DB_HOST}" + (f" -> {db_addr}" if db_addr else "") + f": {exc}"
    except ValueError as exc:
        detail["db"] = f"{DB_HOST} -> {db_addr}: bad health body ({exc})"
    with LOCK:
        STATE.update(dns_ok=dns_ok, db_ok=db_ok, db_addr=db_addr, detail=detail, checked_at=time.time())


def checker_loop() -> None:
    while True:
        try:
            check_dependencies()
        except Exception as exc:  # never let the checker die
            print(f"[app] checker error: {exc}", flush=True)
        time.sleep(CHECK_INTERVAL)


class Handler(BaseHTTPRequestHandler):
    server_version = "sable-lab-app/1"

    def log_message(self, fmt, *args):  # quieter logs
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
            STATE["requests_total"] += 1
            snap = dict(STATE)
        if self.path == "/health":
            ok = snap["dns_ok"] and snap["db_ok"]
            if not ok:
                with LOCK:
                    STATE["errors_total"] += 1
            body = {"ok": ok, "dns": snap["dns_ok"], "db": snap["db_ok"], "db_host": DB_HOST,
                    "db_addr": snap["db_addr"], "detail": snap["detail"],
                    "checked_age_s": round(time.time() - snap["checked_at"], 1) if snap["checked_at"] else None}
            self._send(200 if ok else 503, json.dumps(body) + "\n", "application/json")
        elif self.path == "/metrics":
            lines = [
                "# HELP up 1 while the app process serves (Prometheus also synthesises its own up).",
                "# TYPE up gauge",
                "up 1",
                "# HELP app_dependency_ok 1 when the dependency answered correctly on the last check.",
                "# TYPE app_dependency_ok gauge",
                f'app_dependency_ok{{dep="dns"}} {int(snap["dns_ok"])}',
                f'app_dependency_ok{{dep="db"}} {int(snap["db_ok"])}',
                "# HELP app_requests_total Requests served.",
                "# TYPE app_requests_total counter",
                f"app_requests_total {snap['requests_total']}",
                "# HELP app_errors_total Health checks that reported a broken dependency.",
                "# TYPE app_errors_total counter",
                f"app_errors_total {snap['errors_total']}",
                "# HELP app_db_target_info The configured primary.",
                "# TYPE app_db_target_info gauge",
                f'app_db_target_info{{host="{DB_HOST}",port="{DB_PORT}"}} 1',
            ]
            self._send(200, "\n".join(lines) + "\n")
        else:
            self._send(200, f"sable-lab app: db={DB_HOST}:{DB_PORT} (config {CONF_PATH})\n")


def main() -> int:
    print(f"[app] db_host={DB_HOST} db_port={DB_PORT} canary={DNS_CANARY} port={PORT}", flush=True)
    threading.Thread(target=checker_loop, name="checker", daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
