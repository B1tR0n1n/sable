#!/usr/bin/env python3
"""End-to-end runner for the lab (Phase 8 acceptance), driven through the console server's
documented API (console/docs/API.md) and nothing else. stdlib only.

    python3 -m console.lab.e2e [--only LABEL] [--skip-negative] [--console-url URL]

For every positive scenario: reset the lab (POST /api/lab/fault heal_all), inject the fault,
wait for an open Finding whose root cause is the expected node, get (or request) its Plan,
approve it as actor `e2e`, wait for the Receipt and require verification.status == "pass"
and the Finding `closed`. Then the negative scenario: inject corrupt_config while the fix is
unavailable (the console started with CONSOLE_DISABLE_ACTIONS=set_config_value), approve the
plan the template planner produces anyway, and require verification "fail",
rollback.performed and the Finding `reopened`.

The API has no "force a wrong plan" knob (POST /api/findings/{id}/plan takes only
{planner}), so the negative scenario relies on that environment variable. Set
E2E_CONSOLE_CMD (e.g. "python3 -m console.server") and this runner starts the console
itself with CONSOLE_LAB=1 and restarts it with the variable for the negative phase;
otherwise the positive phase runs against whatever is listening and the negative phase
fails with a clear line if the plan still contains set_config_value.

Environment: CONSOLE_URL (http://127.0.0.1:7780), E2E_FINDING_TIMEOUT (180 s),
E2E_RECEIPT_TIMEOUT (240 s), E2E_SETTLE_TIMEOUT (90 s), E2E_CONSOLE_CMD, E2E_CONSOLE_BOOT (60 s).
Exit status: 0 only when every scenario passed.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

LAB_DIR = Path(__file__).resolve().parent
ROOT = LAB_DIR.parents[1]
FAULTS_DIR = LAB_DIR / "faults"

CONSOLE_URL = os.environ.get("CONSOLE_URL", "http://127.0.0.1:7780").rstrip("/")
FINDING_TIMEOUT = float(os.environ.get("E2E_FINDING_TIMEOUT", "180"))
RECEIPT_TIMEOUT = float(os.environ.get("E2E_RECEIPT_TIMEOUT", "240"))
SETTLE_TIMEOUT = float(os.environ.get("E2E_SETTLE_TIMEOUT", "90"))
CONSOLE_BOOT = float(os.environ.get("E2E_CONSOLE_BOOT", "60"))
POLL_S = 2.0
NEGATIVE_DISABLED_ACTION = "set_config_value"


@dataclass(frozen=True)
class Scenario:
    label: str
    fault: str            # faults/<fault>.sh, run by POST /api/lab/fault {"name": fault}
    expect_root: str      # Finding.root_cause.node_id
    negative: bool = False


SCENARIOS: list[Scenario] = [
    Scenario("stop_service dns", "stop_service", "dns"),      # stop_service.sh defaults to dns
    Scenario("corrupt_config", "corrupt_config", "app"),
    Scenario("kill_primary", "kill_primary", "db"),
    Scenario("poison_dns", "poison_dns", "dns"),
]
NEGATIVE = Scenario("corrupt_config with the fix disabled (wrong plan)", "corrupt_config", "app", negative=True)


# ------------------------------------------------------------------ http


class ApiError(Exception):
    def __init__(self, status: int, path: str, body: Any):
        super().__init__(f"{status} {path}: {body}")
        self.status, self.path, self.body = status, path, body


def api(method: str, path: str, body: Optional[dict[str, Any]] = None, timeout: float = 15.0) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(CONSOLE_URL + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode() or "null"
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = raw
        raise ApiError(exc.code, path, parsed) from None


def log(msg: str) -> None:
    print(f"[e2e] {msg}", flush=True)


# ------------------------------------------------------------------ console lifecycle (optional)


class Console:
    """Starts/restarts the console server when E2E_CONSOLE_CMD is set; otherwise a no-op."""

    def __init__(self, cmd: Optional[str]):
        self.cmd = cmd
        self.proc: Optional[subprocess.Popen[bytes]] = None

    def start(self, extra_env: Optional[dict[str, str]] = None) -> None:
        if not self.cmd:
            return
        self.stop()
        env = {**os.environ, "CONSOLE_LAB": "1", **(extra_env or {})}
        log(f"starting console: {self.cmd} " + " ".join(f"{k}={v}" for k, v in (extra_env or {}).items()))
        self.proc = subprocess.Popen(shlex.split(self.cmd), cwd=str(ROOT), env=env)
        deadline = time.time() + CONSOLE_BOOT
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"console exited early with {self.proc.returncode}")
            try:
                api("GET", "/api/state", timeout=3)
                return
            except (ApiError, OSError):
                time.sleep(1)
        raise RuntimeError(f"console did not answer /api/state within {CONSOLE_BOOT:.0f}s")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


# ------------------------------------------------------------------ loop steps


def open_findings() -> list[dict[str, Any]]:
    return list(api("GET", "/api/findings?status=open") or [])


def reset_lab() -> None:
    log("reset: heal_all")
    api("POST", "/api/lab/fault", {"name": "heal_all"}, timeout=240)
    deadline = time.time() + SETTLE_TIMEOUT
    while time.time() < deadline:
        left = open_findings()
        if not left:
            return
        time.sleep(POLL_S)
    log(f"warning: {len(open_findings())} finding(s) still open after reset; continuing with a baseline")


def inject(fault: str) -> None:
    log(f"inject: {fault}")
    api("POST", "/api/lab/fault", {"name": fault}, timeout=120)


def wait_finding(expect_root: str, baseline: dict[str, int]) -> dict[str, Any]:
    deadline = time.time() + FINDING_TIMEOUT
    seen: set[str] = set()
    while time.time() < deadline:
        for f in open_findings():
            root = f["root_cause"]["node_id"]
            seen.add(root)
            fresh = f["id"] not in baseline or int(f.get("occurrences", 1)) > baseline[f["id"]]
            if root == expect_root and fresh:
                return f
        time.sleep(POLL_S)
    raise AssertionError(f"no open finding with root cause {expect_root!r} within {FINDING_TIMEOUT:.0f}s "
                         f"(roots seen: {sorted(seen) or 'none'})")


def get_plan(finding_id: str) -> dict[str, Any]:
    deadline = time.time() + 60
    requested = False
    while time.time() < deadline:
        try:
            plan = api("GET", f"/api/findings/{finding_id}/plan")
            if plan and plan.get("gate"):
                return plan
        except ApiError as exc:
            if exc.status != 404:
                raise
            if not requested:                       # the template planner did not run: ask for it
                requested = True
                api("POST", f"/api/findings/{finding_id}/plan", {"planner": "template"}, timeout=60)
                continue
        time.sleep(POLL_S)
    raise AssertionError(f"no gated plan for finding {finding_id} within 60s")


def approve(plan: dict[str, Any]) -> None:
    pid = plan["id"]
    decision = plan["gate"]["decision"]
    if decision == "report_only":
        raise AssertionError(f"plan {pid} is report_only ({plan['gate']['rule_id']}); nothing can execute")
    result = api("POST", f"/api/plans/{pid}/approve", {"actor": "e2e", "decision": "approve"}, timeout=60)
    started = bool(result.get("started"))
    if not started and decision == "delay":
        result = api("POST", f"/api/plans/{pid}/approve", {"actor": "e2e", "decision": "execute_now"}, timeout=60)
        started = bool(result.get("started"))
    if not started and decision == "auto":
        started = True                              # auto plans executed on creation
    log(f"approved plan {pid} (gate={decision}, rule={plan['gate']['rule_id']}, started={started})")


def wait_receipt(finding_id: str, plan_id: str) -> dict[str, Any]:
    deadline = time.time() + RECEIPT_TIMEOUT
    while time.time() < deadline:
        for r in api("GET", f"/api/receipts?finding_id={finding_id}") or []:
            if r.get("plan_id") == plan_id and r.get("verification"):
                return r
        time.sleep(POLL_S)
    raise AssertionError(f"no verified receipt for plan {plan_id} within {RECEIPT_TIMEOUT:.0f}s")


def wait_finding_status(finding_id: str, want: str, timeout: float = 60) -> str:
    deadline = time.time() + timeout
    status = "?"
    while time.time() < deadline:
        status = api("GET", f"/api/findings/{finding_id}").get("status", "?")
        if status == want:
            return status
        time.sleep(POLL_S)
    return status


def step_actions(plan: dict[str, Any]) -> list[str]:
    return [s["action_id"] for s in plan.get("steps", [])]


# ------------------------------------------------------------------ scenarios


def run_scenario(sc: Scenario) -> None:
    reset_lab()
    baseline = {f["id"]: int(f.get("occurrences", 1)) for f in open_findings()}
    inject(sc.fault)
    finding = wait_finding(sc.expect_root, baseline)
    fid = finding["id"]
    log(f"finding {fid}: root={sc.expect_root} state={finding['root_cause']['state']} "
        f"mode={finding.get('detection_mode')} severity={finding.get('severity')}")
    plan = get_plan(fid)
    actions = step_actions(plan)
    log(f"plan {plan['id']}: {actions} verification={plan['verification']['predicate']}")

    if sc.negative and NEGATIVE_DISABLED_ACTION in actions:
        raise AssertionError(
            f"the plan still uses {NEGATIVE_DISABLED_ACTION}; the negative scenario needs the console "
            f"started with CONSOLE_DISABLE_ACTIONS={NEGATIVE_DISABLED_ACTION} (set E2E_CONSOLE_CMD so e2e "
            f"restarts it, or run with --skip-negative)")

    approve(plan)
    receipt = wait_receipt(fid, plan["id"])
    verification = receipt["verification"]["status"]
    rollback = receipt.get("rollback") or {}
    log(f"receipt {receipt['id']}: verification={verification} rollback.performed={rollback.get('performed')} "
        f"hash={str(receipt.get('receipt_hash'))[:12]}")

    if not sc.negative:
        if verification != "pass":
            raise AssertionError(f"verification {verification!r}, expected 'pass' "
                                 f"(observed={receipt['verification'].get('observed')})")
        status = wait_finding_status(fid, "closed")
        if status != "closed":
            raise AssertionError(f"finding {fid} is {status!r}, expected 'closed'")
    else:
        if verification != "fail":
            raise AssertionError(f"verification {verification!r}, expected 'fail' for the wrong plan")
        if not rollback.get("performed"):
            raise AssertionError("rollback.performed is not true after the failed verification")
        status = wait_finding_status(fid, "reopened")
        if status != "reopened":
            raise AssertionError(f"finding {fid} is {status!r}, expected 'reopened'")


def check_preconditions() -> None:
    missing = [sc.fault for sc in SCENARIOS + [NEGATIVE] if not (FAULTS_DIR / f"{sc.fault}.sh").exists()]
    if missing:
        raise SystemExit(f"[e2e] fault scripts missing in {FAULTS_DIR}: {missing}")
    state = api("GET", "/api/state")
    sable = state.get("sable") or {}
    overlord = state.get("overlord") or {}
    log(f"console {CONSOLE_URL}: sable ok={sable.get('ok')} ({sable.get('device')}), "
        f"overlord ok={overlord.get('ok')} ({overlord.get('version')})")
    if not sable.get("ok") or not overlord.get("ok"):
        raise SystemExit("[e2e] the console reports SABLE or OVERLORD not ok; see console/lab/README.md")


def main(argv: Optional[list[str]] = None) -> int:
    global CONSOLE_URL
    ap = argparse.ArgumentParser(description="lab end-to-end suite (console/lab/e2e.py)")
    ap.add_argument("--only", help="run only the scenario whose label contains this text")
    ap.add_argument("--skip-negative", action="store_true")
    ap.add_argument("--console-url", default=CONSOLE_URL)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    CONSOLE_URL = args.console_url.rstrip("/")

    scenarios = [sc for sc in SCENARIOS if not args.only or args.only in sc.label]
    negative = None if args.skip_negative or (args.only and args.only not in NEGATIVE.label) else NEGATIVE
    if args.list:
        for sc in scenarios + ([negative] if negative else []):
            print(f"{sc.label:50s} fault={sc.fault:15s} root={sc.expect_root} negative={sc.negative}")
        return 0

    console = Console(os.environ.get("E2E_CONSOLE_CMD"))
    results: list[tuple[str, str, str]] = []
    try:
        console.start()
        check_preconditions()
        for sc in scenarios:
            t0 = time.time()
            try:
                run_scenario(sc)
                results.append((sc.label, "PASS", f"{time.time() - t0:.0f}s"))
            except (AssertionError, ApiError, OSError) as exc:
                results.append((sc.label, "FAIL", str(exc)))
            log(f"{results[-1][1]} {sc.label}: {results[-1][2]}")
        if negative:
            if console.cmd:
                console.start({"CONSOLE_DISABLE_ACTIONS": NEGATIVE_DISABLED_ACTION})
            t0 = time.time()
            try:
                run_scenario(negative)
                results.append((negative.label, "PASS", f"{time.time() - t0:.0f}s"))
            except (AssertionError, ApiError, OSError) as exc:
                results.append((negative.label, "FAIL", str(exc)))
            log(f"{results[-1][1]} {negative.label}: {results[-1][2]}")
        try:
            reset_lab()
        except (ApiError, OSError) as exc:
            log(f"final reset failed: {exc}")
    except (RuntimeError, ApiError, OSError, SystemExit) as exc:
        print(f"[e2e] ABORT: {exc}", file=sys.stderr)
        return 2
    finally:
        console.stop()

    print()
    failed = 0
    for label, verdict, detail in results:
        print(f"[e2e] {verdict:4s} {label:50s} {detail}")
        failed += verdict != "PASS"
    print(f"[e2e] {len(results) - failed}/{len(results)} scenarios passed")
    return 1 if failed or not results else 0


if __name__ == "__main__":
    sys.exit(main())
