#!/usr/bin/env bash
# Download a manageable slice of the Alibaba Cluster Trace v2018.
#
# Pulls only the three files needed by adapters/alibaba.py:
#   machine_meta.tar.gz     ~92KB    failure events + topology domains
#   machine_usage.tar.gz    ~1.7GB   per-machine telemetry
#   container_meta.tar.gz   ~2.4MB   container→machine + app_du
#
# Skips container_usage (~28GB) and batch_instance (~20GB) — those are
# only needed if you want to model containers as nodes.
#
# Total disk: ~1.7GB compressed, ~12GB uncompressed.
# Bandwidth note: Beijing OSS bucket — expect slow transfers from US (5-50 MB/s).

set -euo pipefail

DEST="${1:-/mnt/vault/projects/sable/data/alibaba_2018}"
BASE="http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces"

mkdir -p "$DEST"
cd "$DEST"

for f in machine_meta.tar.gz machine_usage.tar.gz container_meta.tar.gz; do
  if [[ -f "$f" ]]; then
    echo "[skip] $f already present"
  else
    echo "[fetch] $f"
    wget -c --retry-connrefused --tries=0 --timeout=120 "$BASE/$f"
  fi
done

for f in machine_meta.tar.gz machine_usage.tar.gz container_meta.tar.gz; do
  csv="${f%.tar.gz}.csv"
  if [[ -f "$csv" ]]; then
    echo "[skip] $csv already extracted"
  else
    echo "[extract] $f"
    tar -xzf "$f"
  fi
done

echo
echo "Done. Files in $DEST:"
ls -lh "$DEST"/*.csv
