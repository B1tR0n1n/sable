#!/usr/bin/env bash
# heal_all.sh   Back to pristine: restores every config from its *.pristine twin, then
# recreates and starts the whole stack and waits for every healthcheck. Recreating (not just
# restarting) also clears in-container state such as a poisoned DNS record and makes the
# proxy re-read proxy/upstream.env. Idempotent; `make -C console lab-reset` runs this.
set -euo pipefail
FAULT_NAME=heal_all
# shellcheck source=_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

restored=0
for pristine in "$LAB_DIR"/config/*.pristine "$LAB_DIR"/proxy/*.pristine; do
    [ -e "$pristine" ] || continue
    target="${pristine%.pristine}"
    if [ ! -e "$target" ] || ! cmp -s "$pristine" "$target"; then
        cp "$pristine" "$target"
        say "restored ${target#"$LAB_DIR"/}"
        restored=$((restored + 1))
    fi
done
[ "$restored" -eq 0 ] && say "all configs already pristine"

say "recreating the stack and waiting for healthchecks"
compose up -d --build --force-recreate --wait --wait-timeout "${WAIT_TIMEOUT:-120}"
say "lab healthy: $(compose ps --services --status running | tr '\n' ' ')"
