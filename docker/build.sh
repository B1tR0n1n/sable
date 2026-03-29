#!/bin/bash
set -e

echo ""
echo "  SABLE Engine — Docker Build"
echo "  ═══════════════════════════════════"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SABLE_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate ml-env
source /home/b1tr0n1n/ml-env/bin/activate

# Step 1: Copy engine source
echo "  [1/4] Copying engine source..."
rm -rf "$SCRIPT_DIR/engine"
mkdir -p "$SCRIPT_DIR/engine/fusion" "$SCRIPT_DIR/engine/pillar1" "$SCRIPT_DIR/engine/pillar2" "$SCRIPT_DIR/engine/pillar3" "$SCRIPT_DIR/engine/sim"

# Fusion
cp "$SABLE_ROOT/fusion/temporal_chain.py" "$SCRIPT_DIR/engine/fusion/"
cp "$SABLE_ROOT/fusion/staged_fusion_v3.py" "$SCRIPT_DIR/engine/fusion/"
cp "$SABLE_ROOT/fusion/shared_latent_space.py" "$SCRIPT_DIR/engine/fusion/"

# Pillar 1
cp "$SABLE_ROOT/pillar1/cortex_gnn_model.py" "$SCRIPT_DIR/engine/pillar1/"
cp "$SABLE_ROOT/pillar1/domain_portability_test.py" "$SCRIPT_DIR/engine/pillar1/"
cp "$SABLE_ROOT/pillar1/build_infra_dataset.py" "$SCRIPT_DIR/engine/pillar1/"

# Pillar 2
cp "$SABLE_ROOT/pillar2/pomcp.py" "$SCRIPT_DIR/engine/pillar2/"

# Pillar 3
cp "$SABLE_ROOT/pillar3/sable_mamba.py" "$SCRIPT_DIR/engine/pillar3/"
cp "$SABLE_ROOT/pillar3/sable_mamba_final.py" "$SCRIPT_DIR/engine/pillar3/"
cp "$SABLE_ROOT/pillar3/generate_temporal_data.py" "$SCRIPT_DIR/engine/pillar3/"

# Sim
cp -r "$SABLE_ROOT/sable_sim" "$SCRIPT_DIR/engine/sim/sable_sim"

echo "  [2/4] Copying checkpoints..."
rm -rf "$SCRIPT_DIR/checkpoints"
mkdir -p "$SCRIPT_DIR/checkpoints"
cp "$SABLE_ROOT/fusion/checkpoints/staged_fusion_v3.pt" "$SCRIPT_DIR/checkpoints/fusion.pt"
cp "$SABLE_ROOT/fusion/checkpoints/temporal_chain.pt" "$SCRIPT_DIR/checkpoints/temporal.pt"

echo "  [3/4] Pre-computing scenarios..."
cd "$SCRIPT_DIR"
/home/b1tr0n1n/ml-env/bin/python precompute_scenarios.py

echo "  [4/4] Building Docker image..."
docker build -t sable-engine "$SCRIPT_DIR"

echo ""
echo "  Done. Run with: ./run.sh"
echo ""
