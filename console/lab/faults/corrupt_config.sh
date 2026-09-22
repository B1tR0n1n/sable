#!/usr/bin/env bash
# corrupt_config.sh   Points the app at a host that does not exist (db_host=nowhere.lab in
# config/app.conf) and restarts app so it re-reads the file. Heal: catalog set_config_value
# {file: config/app.conf, key: db_host, value: db.lab, service: app} or faults/heal_all.sh.
# What SABLE sees: app up 1 but app_dependency_ok{dep="db"} 0 -> `healthy` 0 -> app degraded;
# dns and db stay healthy, so the root cause is app itself.
set -euo pipefail
FAULT_NAME=corrupt_config
# shellcheck source=_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

CONF="$LAB_DIR/config/app.conf"
BAD_HOST="${BAD_HOST:-nowhere.lab}"
if grep -qx "db_host=$BAD_HOST" "$CONF"; then
    say "$CONF already has db_host=$BAD_HOST"
elif grep -q '^db_host=' "$CONF"; then
    sed -i.bak "s/^db_host=.*/db_host=$BAD_HOST/" "$CONF" && rm -f "$CONF.bak"
    say "rewrote db_host=$BAD_HOST in $CONF"
else
    printf 'db_host=%s\n' "$BAD_HOST" >> "$CONF"
    say "appended db_host=$BAD_HOST to $CONF"
fi
compose restart -t 2 app >/dev/null
say "restarted app (it re-reads app.conf on start)"
