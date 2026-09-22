#!/bin/sh
# dnsmasq wrapper (PID 1).
#   /etc/dnsmasq.d/lab.hosts  pristine records, bind-mounted read-only from config/lab.hosts
#   /run/dnsmasq/hosts        the LIVE records dnsmasq serves ("the cache"); regenerated from
#                             the pristine file at start and whenever PID 1 receives SIGHUP,
#                             which is then forwarded to dnsmasq so it re-reads the file.
# faults/poison_dns.sh edits the live file and HUPs dnsmasq directly (not PID 1), so the
# poison persists until something clears the cache: `docker compose kill -s HUP dns`
# (catalog action clear_dns_cache) or `docker compose restart dns` (restart_service).
set -eu
PRISTINE=/etc/dnsmasq.d/lab.hosts
LIVE=/run/dnsmasq/hosts

regen() {
    mkdir -p /run/dnsmasq
    cp "$PRISTINE" "$LIVE"
    chmod 0644 "$LIVE"
    echo "[dns] live records regenerated from $PRISTINE"
}

regen
dnsmasq --keep-in-foreground --log-facility=- --conf-file=/etc/dnsmasq.conf &
child=$!
trap 'regen; kill -HUP "$child"' HUP
trap 'kill -TERM "$child"' TERM INT
while kill -0 "$child" 2>/dev/null; do
    wait "$child" || true
done
