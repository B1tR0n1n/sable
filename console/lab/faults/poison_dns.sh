#!/usr/bin/env bash
# poison_dns.sh   Plants a wrong A record for db.lab in the dns container's LIVE records
# (/run/dnsmasq/hosts, the runtime copy of config/lab.hosts) and HUPs dnsmasq directly so it
# serves the poison. The pristine bind-mounted files are untouched, because dnsmasq never
# re-reads dnsmasq.conf on SIGHUP: a poison written there could not be healed by the catalog.
# Heal: catalog clear_dns_cache {service: dns} (`docker compose kill -s HUP dns`: PID 1 is the
# entrypoint, which regenerates the live records and re-HUPs dnsmasq), restart_service dns,
# or faults/heal_all.sh. What SABLE sees: blackbox's dns probe validates the db.lab answer, so
# probe_success{instance="dns:53"} 0 -> dns unreachable; app resolves db.lab to the wrong IP
# -> app_dependency_ok{dep="db"} 0 -> app degraded.
set -euo pipefail
FAULT_NAME=poison_dns
# shellcheck source=_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

POISON_IP="${POISON_IP:-172.28.0.99}"
if ! service_running dns; then
    say "dns is not running; start the lab first (make -C console lab-up)"
    exit 2
fi
# POSIX-only sed/grep patterns: the container's tools are busybox.
result="$(compose exec -T dns sh -c '
    LIVE=/run/dnsmasq/hosts
    if grep -q "^'"$POISON_IP"'[[:space:]][[:space:]]*db\.lab$" "$LIVE"; then
        echo already
        exit 0
    fi
    sed -i "s/^[0-9.][0-9.]*[[:space:]][[:space:]]*db\.lab$/'"$POISON_IP"' db.lab/" "$LIVE"
    kill -HUP "$(pidof dnsmasq)"
    echo poisoned
')"
case "$result" in
    *already*)  say "db.lab already resolves to $POISON_IP; nothing to do" ;;
    *poisoned*) say "db.lab now resolves to $POISON_IP (dnsmasq re-read its live records)" ;;
    *)          say "unexpected result from dns container: $result"; exit 1 ;;
esac
