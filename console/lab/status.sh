#!/usr/bin/env bash
# status.sh   One screen of lab truth: compose state, each service's /health from the host
# ports, Prometheus targets and the blackbox probes. `make -C console lab-status`.
set -euo pipefail
LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROM="${PROM_URL:-http://localhost:9090}"

echo "== docker compose ps"
docker compose -f "$LAB_DIR/docker-compose.yml" ps --format 'table {{.Service}}\t{{.Status}}' || true

echo; echo "== /health (host ports)"
for entry in "app http://localhost:18000/health" "db http://localhost:18100/health" \
             "db-replica http://localhost:18101/health" "proxy->primary http://localhost:18080/health" \
             "proxy /metrics http://localhost:18080/metrics"; do
    name="${entry% *}"; url="${entry##* }"
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 4 "$url" 2>/dev/null || echo 000)"
    printf '  %-18s %s  %s\n' "$name" "$code" "$url"
done
printf '  %-18s ' "dns db.lab"
if command -v dig >/dev/null 2>&1; then
    dig +short +time=2 +tries=1 @127.0.0.1 -p 15353 db.lab 2>/dev/null | tr '\n' ' ' || true; echo
else
    echo "(install dig, or: docker compose exec dns nslookup db.lab 127.0.0.1)"
fi

echo; echo "== prometheus targets ($PROM)"
curl -sS --max-time 4 "$PROM/api/v1/targets" | python3 -c '
import json, sys
data = json.load(sys.stdin)["data"]["activeTargets"]
for t in sorted(data, key=lambda t: (t["labels"]["job"], t["labels"]["instance"])):
    print("  %-12s %-28s %-5s %s" % (t["labels"]["job"], t["labels"]["instance"], t["health"], (t.get("lastError") or "")[:60]))
' || echo "  (prometheus not reachable)"

echo; echo "== what SABLE reads: up / probe_success / app_dependency_ok"
for q in 'probe_success or on(instance, job) up' 'app_dependency_ok'; do
    curl -sS --max-time 4 --get "$PROM/api/v1/query" --data-urlencode "query=$q" | python3 -c '
import json, sys
for r in json.load(sys.stdin)["data"]["result"]:
    m = r["metric"]; print("  %-12s %-28s %-8s %s" % (m.get("job",""), m.get("instance",""), m.get("dep",""), r["value"][1]))
' || true
done
