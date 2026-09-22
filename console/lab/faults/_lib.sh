# Shared by every fault script: sourced, not executed.
#   LAB_DIR       the lab checkout (console/lab), derived from this file's location
#   COMPOSE_FILE  its docker-compose.yml
#   compose ...   docker compose against that file
#   say ...       one line per thing done, prefixed with the fault name
# shellcheck shell=bash
LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$LAB_DIR/docker-compose.yml"
FAULT_NAME="${FAULT_NAME:-$(basename "${BASH_SOURCE[1]:-fault}" .sh)}"

compose() { docker compose -f "$COMPOSE_FILE" "$@"; }
say() { printf '[%s] %s\n' "$FAULT_NAME" "$*"; }
# Drain the whole pipe (no grep -q): with pipefail, grep exiting early can turn a
# successful compose into an EPIPE failure; and keep compose's stderr, so a
# broken compose invocation says why instead of "unknown compose service".
service_defined() { compose config --services | grep -x -- "$1" >/dev/null; }
service_running() { compose ps --services --status running | grep -x -- "$1" >/dev/null; }
