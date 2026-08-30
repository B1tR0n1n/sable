#!/usr/bin/env bash
# SABLE stack launcher: SABLE engine + Nemotron + Grafana.
# Idempotent — skips services that are already healthy.
set -u

PROJECT_DIR="/mnt/vault/projects/sable"
ML_ENV="${HOME}/ml-env"
NEMOTRON_BIN="${HOME}/llama.cpp/build/bin/llama-server"
NEMOTRON_MODEL="/mnt/vault/models/nemotron-nano/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf"
LOG_DIR="/tmp/sable-logs"
SABLE_PORT=8080
NEMOTRON_PORT=8081
GRAFANA_PORT=3000
GRAFANA_NAME="sable-grafana"

mkdir -p "${LOG_DIR}"

GOLD="\033[38;2;201;162;39m"
DIM="\033[38;2;138;127;110m"
GREEN="\033[38;2;74;122;69m"
RED="\033[38;2;166;61;47m"
RESET="\033[0m"

say()  { printf "${DIM}[sable]${RESET} %s\n" "$*"; }
ok()   { printf "${GREEN}[ ok ]${RESET} %s\n" "$*"; }
warn() { printf "${RED}[fail]${RESET} %s\n" "$*"; }

port_open() {
    ss -tlnp 2>/dev/null | grep -qE ":${1}\b"
}

wait_for_port() {
    local port="$1"
    local label="$2"
    local check_url="$3"
    local timeout="${4:-60}"
    local elapsed=0
    while (( elapsed < timeout )); do
        if curl -fsS -o /dev/null --max-time 2 "${check_url}" 2>/dev/null; then
            ok "${label} ready on :${port}"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    warn "${label} did not become ready on :${port} in ${timeout}s"
    return 1
}

printf "${GOLD}\n"
cat <<'BANNER'
  ┌─────────────────────────────────────────┐
  │   S A B L E    S T A C K    L A U N C H │
  │   engine + nemotron + grafana           │
  └─────────────────────────────────────────┘
BANNER
printf "${RESET}\n"

# ── SABLE engine ─────────────────────────────────────────────────────────
if port_open "${SABLE_PORT}"; then
    say "SABLE already on :${SABLE_PORT}"
else
    if [[ ! -f "${ML_ENV}/bin/activate" ]]; then
        warn "ml-env not found at ${ML_ENV}"
        exit 1
    fi
    say "starting SABLE engine"
    # shellcheck disable=SC1091
    source "${ML_ENV}/bin/activate"
    nohup python3 "${PROJECT_DIR}/docker/server.py" \
        > "${LOG_DIR}/server.log" 2>&1 &
    disown
fi
wait_for_port "${SABLE_PORT}" "SABLE" "http://127.0.0.1:${SABLE_PORT}/api/status" 60 || exit 1

# ── Nemotron ─────────────────────────────────────────────────────────────
if port_open "${NEMOTRON_PORT}"; then
    say "Nemotron already on :${NEMOTRON_PORT}"
else
    if [[ ! -x "${NEMOTRON_BIN}" ]]; then
        warn "llama-server not found at ${NEMOTRON_BIN} — skipping Nemotron"
    elif [[ ! -f "${NEMOTRON_MODEL}" ]]; then
        warn "Nemotron model not found at ${NEMOTRON_MODEL} — skipping Nemotron"
    else
        say "starting Nemotron (model load ~30s)"
        nohup "${NEMOTRON_BIN}" \
            -m "${NEMOTRON_MODEL}" \
            --ctx-size 4096 \
            --temp 0.4 \
            -ngl 999 \
            --port "${NEMOTRON_PORT}" \
            > "${LOG_DIR}/nemotron.log" 2>&1 &
        disown
    fi
fi
if port_open "${NEMOTRON_PORT}" || pgrep -f "llama-server.*${NEMOTRON_PORT}" >/dev/null; then
    wait_for_port "${NEMOTRON_PORT}" "Nemotron" "http://127.0.0.1:${NEMOTRON_PORT}/health" 120 || true
fi

# ── Grafana ──────────────────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
    warn "docker not found — skipping Grafana"
else
    if docker ps --format '{{.Names}}' | grep -q "^${GRAFANA_NAME}$"; then
        say "Grafana container already running"
    else
        # Clean up any stopped container with the same name
        docker rm "${GRAFANA_NAME}" >/dev/null 2>&1 || true
        # Make plugins dir writable by Grafana UID 472 (idempotent)
        chmod -R o+w "${PROJECT_DIR}/grafana/plugins" 2>/dev/null || true
        say "starting Grafana container"
        docker run -d --name "${GRAFANA_NAME}" \
            -p "${GRAFANA_PORT}:3000" \
            -e GF_SECURITY_ADMIN_USER=sable \
            -e GF_SECURITY_ADMIN_PASSWORD=sable \
            -e GF_INSTALL_PLUGINS=marcusolsson-json-datasource \
            -e GF_DEFAULT_THEME=dark \
            -e GF_USERS_DEFAULT_THEME=dark \
            -e GF_PLUGINS_ALLOW_LOADING_UNSIGNED_PLUGINS=sable-topology-panel \
            -e GF_AUTH_ANONYMOUS_ENABLED=true \
            -e GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
            -e GF_PANELS_DISABLE_SANITIZE_HTML=true \
            -v "${PROJECT_DIR}/grafana/provisioning:/etc/grafana/provisioning" \
            -v "${PROJECT_DIR}/grafana/dashboards:/var/lib/grafana/dashboards" \
            -v "${PROJECT_DIR}/grafana/plugins:/var/lib/grafana/plugins" \
            -v "${PROJECT_DIR}/grafana/grafana.ini:/etc/grafana/grafana.ini" \
            -v sable-grafana-data:/var/lib/grafana \
            --add-host=host.docker.internal:host-gateway \
            grafana/grafana-oss:11.6.0 >/dev/null
    fi
    wait_for_port "${GRAFANA_PORT}" "Grafana" "http://127.0.0.1:${GRAFANA_PORT}/api/health" 120 || true
fi

printf "\n${GOLD}  STACK READY${RESET}\n"
printf "  ${DIM}SABLE     →${RESET} http://127.0.0.1:${SABLE_PORT}/\n"
printf "  ${DIM}Nemotron  →${RESET} http://127.0.0.1:${NEMOTRON_PORT}/\n"
printf "  ${DIM}Grafana   →${RESET} http://127.0.0.1:${GRAFANA_PORT}/  (sable/sable)\n"
printf "  ${DIM}Logs      →${RESET} ${LOG_DIR}/\n\n"
