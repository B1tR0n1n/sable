#!/usr/bin/env bash
# stop_service.sh [service]   (default: dns, so the console's `{"name":"stop_service"}` stops dns)
# Stops one lab container. Heal: `docker compose restart <service>` (catalog: restart_service)
# or faults/heal_all.sh. What SABLE sees: `up 0` for the service's instance (for dns:
# probe_success 0 from the blackbox dns probe) -> node unreachable; app_dependency_ok drops
# for whatever depended on it.
set -euo pipefail
FAULT_NAME=stop_service
# shellcheck source=_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

SERVICE="${1:-${SERVICE:-dns}}"
if ! service_defined "$SERVICE"; then
    say "unknown compose service '$SERVICE' (see: docker compose -f $COMPOSE_FILE config --services)"
    exit 2
fi
if service_running "$SERVICE"; then
    compose stop -t 2 "$SERVICE" >/dev/null
    say "stopped $SERVICE"
else
    say "$SERVICE is already stopped; nothing to do"
fi
