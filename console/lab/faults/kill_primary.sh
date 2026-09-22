#!/usr/bin/env bash
# kill_primary.sh   Kills the db primary (SIGKILL, no graceful stop). Heal: catalog
# restart_service {service: db}, or failover_to_replica {primary: db, replica: db-replica,
# proxy: proxy} for the proxy's path, or faults/heal_all.sh. What SABLE sees: db up 0 ->
# unreachable; app_dependency_ok{dep="db"} 0 -> app degraded; the blackbox http probe of
# http://proxy:80/health (the fronted primary) fails.
set -euo pipefail
FAULT_NAME=kill_primary
# shellcheck source=_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

if service_running db; then
    compose kill db >/dev/null
    say "killed db (SIGKILL)"
else
    say "db is already down; nothing to do"
fi
