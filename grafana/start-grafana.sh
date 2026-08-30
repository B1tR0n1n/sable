#!/bin/bash
# Start Grafana for SABLE development
# Assumes SABLE server is running on localhost:8080
#
# Usage:
#   1. Start SABLE: cd /mnt/vault/sable/docker && source ~/ml-env/bin/activate && python3 server.py
#   2. Start Grafana: ./start-grafana.sh
#   3. Open: http://localhost:3000 (login: sable/sable)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Stop existing container if running
docker rm -f sable-grafana 2>/dev/null

docker run -d \
  --name sable-grafana \
  --add-host=host.docker.internal:host-gateway \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_USER=sable \
  -e GF_SECURITY_ADMIN_PASSWORD=sable \
  -e GF_INSTALL_PLUGINS=marcusolsson-json-datasource,yesoreyeram-infinity-datasource \
  -e GF_DEFAULT_THEME=dark \
  -e GF_USERS_DEFAULT_THEME=dark \
  -e GF_PLUGINS_ALLOW_LOADING_UNSIGNED_PLUGINS=sable-topology-panel \
  -e GF_AUTH_ANONYMOUS_ENABLED=true \
  -e GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
  -e GF_PANELS_DISABLE_SANITIZE_HTML=true \
  -v "${SCRIPT_DIR}/provisioning:/etc/grafana/provisioning" \
  -v "${SCRIPT_DIR}/dashboards:/var/lib/grafana/dashboards" \
  -v "${SCRIPT_DIR}/plugins/sable-topology-panel/dist:/var/lib/grafana/plugins/sable-topology-panel" \
  -v "${SCRIPT_DIR}/grafana.ini:/etc/grafana/grafana.ini" \
  grafana/grafana-oss:11.6.0

echo ""
echo "  Grafana starting at http://localhost:3000"
echo "  Login: sable / sable"
echo "  SABLE datasource: http://host.docker.internal:8080/grafana"
echo ""
echo "  Waiting for Grafana to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "  Grafana is ready."
    echo "  Dashboard: http://localhost:3000/d/sable-overview/sable-engine"
    exit 0
  fi
  sleep 1
done
echo "  Grafana may still be starting. Check: docker logs sable-grafana"
