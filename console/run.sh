#!/usr/bin/env bash
# Start / stop / inspect the whole loop as background processes with logs.
#
#     make -C console up        # lab → OVERLORD daemon → SABLE → live monitor → console
#     make -C console status
#     make -C console logs      # tail every log
#     make -C console down
#
# Processes are tracked by pid files under console/.run/. Environment:
#   ANTHROPIC_API_KEY  set → SABLE runs with Claude as its analyst (SABLE_LLM=claude)
#   CONSOLE_POLICY     path to a policy (default: the shipped default-deny)
#   SITE_ID            default lab
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # the sable checkout
RUN="$HERE/console/.run"
mkdir -p "$RUN"
ok()   { printf '  \033[32mok\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33m!!\033[0m   %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; }

alive() { [[ -f "$RUN/$1.pid" ]] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }

start() {  # start <name> <command...>   (cwd = sable root, log = .run/<name>.log)
    local name="$1"; shift
    if alive "$name"; then ok "$name already running (pid $(cat "$RUN/$name.pid"))"; return 0; fi
    ( cd "$HERE" && nohup "$@" > "$RUN/$name.log" 2>&1 & echo $! > "$RUN/$name.pid" )
    sleep 1
    if alive "$name"; then ok "$name started (pid $(cat "$RUN/$name.pid"), log console/.run/$name.log)"
    else fail "$name exited at once — tail console/.run/$name.log"; tail -5 "$RUN/$name.log" | sed 's/^/       /'; return 1; fi
}

wait_http() {  # wait_http <name> <url> <seconds>
    local i=0
    until curl -fsS "$2" > /dev/null 2>&1; do
        i=$((i+1)); [[ $i -ge $3 ]] && { fail "$1 not answering at $2 after ${3}s"; return 1; }
        sleep 1
    done
    ok "$1 answering at $2"
}

stop() {
    local name="$1"
    if [[ -f "$RUN/$name.pid" ]]; then
        local pid; pid="$(cat "$RUN/$name.pid")"
        if kill -0 "$pid" 2>/dev/null; then
            pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null
            for _ in 1 2 3 4 5; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
            ok "$name stopped"
        fi
        rm -f "$RUN/$name.pid"
    fi
}

cmd_up() {
    printf '== lab\n'
    ( cd "$HERE" && make -s -C console lab-up ) || { fail "lab did not come up (make -C console lab-status)"; return 1; }
    printf '== OVERLORD daemon\n'
    start overlord overlord daemon || return 1
    printf '== SABLE\n'
    for f in fusion.pt temporal.pt; do
        [[ -f "$HERE/docker/checkpoints/$f" ]] || { fail "docker/checkpoints/$f missing — SABLE would run on untrained weights; not starting"; return 1; }
    done
    cp -n "$HERE/console/lab/topology.yaml" "$HERE/adapters/topologies/00-lab.yaml" 2>/dev/null || true
    if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then export SABLE_LLM=claude; ok "analyst: Claude (SABLE_LLM=claude)"; else warn "ANTHROPIC_API_KEY not set: SABLE's analyst is the local Nemotron path"; fi
    start sable python3 docker/server.py || return 1
    wait_http SABLE http://127.0.0.1:8080/api/status 90 || return 1
    printf '== live monitor (Prometheus → SABLE)\n'
    start monitor python3 -m console.lab.live_monitor_lab || return 1
    printf '== console\n'
    export CONSOLE_LAB=1 SITE_ID="${SITE_ID:-lab}" SABLE_URL="${SABLE_URL:-http://127.0.0.1:8080}"
    start console python3 -m console.server || return 1
    wait_http console http://127.0.0.1:7780/api/state 30 || return 1
    printf '\n'
    curl -fsS http://127.0.0.1:7780/api/state | python3 -c '
import sys, json; s = json.load(sys.stdin)
print("  SABLE   ok=%s provider=%s device=%s" % (s["sable"]["ok"], s["sable"].get("provider"), s["sable"].get("device")))
print("  OVERLORD ok=%s version=%s" % (s["overlord"]["ok"], s["overlord"].get("version")))
print("  policy   default_deny=%s   lab=%s" % (s["policy"]["default_deny"], s["lab"]))'
    printf '\n\033[32mUp.\033[0m  http://127.0.0.1:7780   break something: make -C console lab-fault FAULT=stop_service\n'
}

cmd_down() {
    for n in console monitor sable overlord; do stop "$n"; done
    ( cd "$HERE" && make -s -C console lab-down ) && ok "lab down"
}

cmd_status() {
    for n in overlord sable monitor console; do
        if alive "$n"; then ok "$n running (pid $(cat "$RUN/$n.pid"))"; else warn "$n not running"; fi
    done
    curl -fsS http://127.0.0.1:7780/api/state 2>/dev/null | python3 -m json.tool 2>/dev/null | head -40 || warn "console not answering on :7780"
    ( cd "$HERE" && make -s -C console lab-status ) 2>/dev/null | tail -20
}

cmd_logs() { tail -n 20 -F "$RUN"/*.log; }

case "${1:-}" in
    up) cmd_up ;;
    down) cmd_down ;;
    status) cmd_status ;;
    logs) cmd_logs ;;
    *) echo "usage: $0 up|down|status|logs"; exit 2 ;;
esac
