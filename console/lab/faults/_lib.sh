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
service_defined() { compose config --services 2>/dev/null | grep -qx -- "$1"; }
service_running() { compose ps --services --status running 2>/dev/null | grep -qx -- "$1"; }
