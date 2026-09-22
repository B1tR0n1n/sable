#!/usr/bin/env bash
# One-shot bootstrap for the OVERLORD × SABLE console on Ubuntu / WSL2.
#
#     bash console/install.sh            # from the sable checkout
#
# Installs everything the loop needs on THIS Linux box, in order, and says
# what it did and what it could not:
#   1. OVERLORD  — cloned into $OVERLORD_DIR (default ~/projects/overlord, never
#                  /mnt/c) or pulled if present, then `packaging/install.sh`
#   2. Python    — console deps + the Claude bridge + SABLE's engine deps
#                  (CPU torch unless SABLE_TORCH=cuda); system pip, no venv
#   3. Node      — the LINUX node/npm (apt), never the Windows one, then the UI build
#   4. Docker    — must be reachable (Docker Desktop with WSL integration)
#   5. Weights   — docker/checkpoints/{fusion,temporal}.pt must exist; they are
#                  gitignored and are NOT fetched by anything — copy them in
# Re-runnable: every step is idempotent.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # the sable checkout
OVERLORD_DIR="${OVERLORD_DIR:-$HOME/projects/overlord}"
OVERLORD_REPO="${OVERLORD_REPO:-https://github.com/B1tR0n1n/overlord.git}"
SKIP_SABLE_DEPS="${SKIP_SABLE_DEPS:-0}"
TORCH="${SABLE_TORCH:-cpu}"
ok()   { printf '  \033[32mok\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33m!!\033[0m   %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
PIP="python3 -m pip install --quiet --disable-pip-version-check"
pipi() {  # install with the PEP 668 override only when the interpreter demands it
    if ! $PIP "$@" 2>/dev/null; then $PIP --break-system-packages "$@" || $PIP --user --break-system-packages "$@"; fi
}
problems=0

step "1. OVERLORD (into $OVERLORD_DIR)"
if [[ -d "$OVERLORD_DIR/.git" ]]; then
    git -C "$OVERLORD_DIR" pull --ff-only -q && ok "pulled $OVERLORD_DIR" || warn "could not fast-forward $OVERLORD_DIR; installing what is there"
else
    mkdir -p "$(dirname "$OVERLORD_DIR")"
    git clone -q "$OVERLORD_REPO" "$OVERLORD_DIR" && ok "cloned into $OVERLORD_DIR" || { fail "clone failed"; problems=$((problems+1)); }
fi
if [[ -f "$OVERLORD_DIR/packaging/install.sh" ]]; then
    if sudo bash "$OVERLORD_DIR/packaging/install.sh" > /tmp/overlord-install.log 2>&1; then
        ok "overlord $(grep -m1 '^VERSION' "$OVERLORD_DIR/overlord.py" | cut -d'"' -f2) installed (log: /tmp/overlord-install.log)"
    else
        fail "overlord install failed — see /tmp/overlord-install.log"; problems=$((problems+1))
    fi
fi
command -v overlord >/dev/null && ok "overlord on PATH: $(command -v overlord)" || { fail "overlord not on PATH"; problems=$((problems+1)); }

step "2. Python packages (system python3, no venv)"
pipi -r "$HERE/console/requirements.txt" anthropic && ok "console + Claude bridge deps" || { fail "console deps"; problems=$((problems+1)); }
if [[ "$SKIP_SABLE_DEPS" != "1" ]]; then
    if [[ "$TORCH" == "cpu" ]]; then
        pipi torch --index-url https://download.pytorch.org/whl/cpu && ok "torch (cpu)" || { fail "torch"; problems=$((problems+1)); }
    else
        pipi torch && ok "torch (cuda build)" || { fail "torch"; problems=$((problems+1)); }
    fi
    pipi -r "$HERE/docker/requirements.txt" numpy networkx && ok "SABLE engine deps" || { fail "SABLE deps"; problems=$((problems+1)); }
else
    warn "SKIP_SABLE_DEPS=1: SABLE's engine deps not installed"
fi

step "3. Node (Linux) and the UI"
if ! command -v npm >/dev/null || [[ "$(command -v npm)" == /mnt/* ]]; then
    warn "no Linux npm (found: $(command -v npm || echo none)); installing nodejs + npm with apt"
    sudo apt-get install -y -qq nodejs npm > /tmp/node-install.log 2>&1 && ok "apt: nodejs npm" || { fail "apt nodejs npm — see /tmp/node-install.log"; problems=$((problems+1)); }
    hash -r
fi
if command -v npm >/dev/null && [[ "$(command -v npm)" != /mnt/* ]]; then
    ok "npm: $(command -v npm) (node $(node --version 2>/dev/null))"
    ( cd "$HERE/console/ui" && [[ -d node_modules/.bin ]] && [[ "$(find node_modules -maxdepth 1 -name '.package-lock.json' -newer package.json | wc -l)" -gt 0 ]] ) \
        || ( cd "$HERE/console/ui" && rm -rf node_modules && npm install --silent --no-audit --no-fund > /tmp/ui-install.log 2>&1 ) \
        || { fail "npm install — see /tmp/ui-install.log"; problems=$((problems+1)); }
    ( cd "$HERE/console/ui" && npm run build --silent > /tmp/ui-build.log 2>&1 ) && ok "UI built → console/ui/dist" || { fail "UI build — see /tmp/ui-build.log"; problems=$((problems+1)); }
else
    fail "npm is still the Windows one; the UI was not built"; problems=$((problems+1))
fi

step "4. Docker"
if docker info > /dev/null 2>&1 && docker compose version > /dev/null 2>&1; then
    ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null) + compose"
else
    fail "docker is not reachable from WSL — enable Docker Desktop's WSL integration for this distro"; problems=$((problems+1))
fi

step "5. SABLE weights"
missing=0
for f in fusion.pt temporal.pt; do
    if [[ -f "$HERE/docker/checkpoints/$f" ]]; then ok "docker/checkpoints/$f"; else fail "docker/checkpoints/$f is missing"; missing=1; fi
done
if [[ $missing == 1 ]]; then
    # SABLE used to run as `docker run … sable-engine` (docker/run.sh) and its
    # Dockerfile bakes checkpoints/ into the image — recover them from there
    img="${SABLE_IMAGE:-sable-engine}"
    if docker image inspect "$img" > /dev/null 2>&1; then
        warn "recovering weights from the $img image"
        cid="$(docker create "$img" 2>/dev/null)"
        mkdir -p "$HERE/docker/checkpoints"
        if docker cp "$cid:/app/checkpoints/." "$HERE/docker/checkpoints/" 2>/dev/null; then
            docker rm "$cid" > /dev/null 2>&1
            missing=0
            for f in fusion.pt temporal.pt; do
                if [[ -f "$HERE/docker/checkpoints/$f" ]]; then ok "recovered docker/checkpoints/$f from $img"; else fail "$img has no /app/checkpoints/$f"; missing=1; fi
            done
        else
            docker rm "$cid" > /dev/null 2>&1; fail "could not copy /app/checkpoints out of $img"
        fi
    else
        warn "no local docker image named $img (docker images | grep -i sable); set SABLE_IMAGE=<name> if it is called something else"
    fi
fi
if [[ $missing == 1 ]]; then
    problems=$((problems+1))
    warn "the weights are gitignored and never in the repo: copy them from wherever SABLE ran before, e.g."
    warn "  mkdir -p $HERE/docker/checkpoints && cp <old-sable>/docker/checkpoints/*.pt $HERE/docker/checkpoints/"
fi

step "6. Lab topology for SABLE"
# SABLE loads the FIRST *.y*ml in adapters/topologies, sorted; 00-lab sorts first
if cmp -s "$HERE/console/lab/topology.yaml" "$HERE/adapters/topologies/00-lab.yaml" 2>/dev/null; then
    ok "adapters/topologies/00-lab.yaml is the lab topology"
else
    cp "$HERE/console/lab/topology.yaml" "$HERE/adapters/topologies/00-lab.yaml" && ok "installed adapters/topologies/00-lab.yaml (SABLE maps the lab, not msp_demo)"
fi

step "7. Tests"
( cd "$HERE" && python3 -m pytest console/tests -q 2>&1 | tail -1 )

printf '\n'
if [[ $problems == 0 ]]; then
    printf '\033[32mReady.\033[0m  Next:  make -C console up    (then open http://127.0.0.1:7780)\n'
else
    printf '\033[33m%d problem(s) above.\033[0m Fix them and re-run: bash console/install.sh\n' "$problems"
fi
exit $problems
