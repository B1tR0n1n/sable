# SABLE — Full System Startup Guide

Everything you need to spin up the complete SABLE stack.

---

## Quick Start (Copy-Paste This)

```bash
# 1. Activate ML environment
source /home/b1tr0n1n/ml-env/bin/activate

# 2. Start GNN server (Pillar 1 — structural reasoning, GPU port 5070)
cd /mnt/vault/sable/pillar1
python gnn_server.py --port 5070 --device cuda &
sleep 3

# 3. Start Dashboard API (backend, port 3001)
cd /mnt/vault/sable/dashboard
python api.py &
sleep 2

# 4. Start Dashboard Frontend (UI, port 3000)
npm run dev -- --host 0.0.0.0 --port 3000 &
sleep 2

# 5. Verify everything
echo "--- Health Check ---"
curl -s http://localhost:5070/health && echo ""
curl -s http://localhost:3001/api/health && echo ""
echo "Dashboard: http://localhost:3000"
```

That's it. Open **http://localhost:3000** in your browser.

---

## What Each Service Does

| Service | Port | Command | What It Does |
|---------|------|---------|-------------|
| **GNN Server** | 5070 | `python gnn_server.py` | Pillar 1 — loads trained GNN, serves link suggestions, contradiction detection, structural scoring |
| **Dashboard API** | 3001 | `python api.py` | FastAPI backend — bridges frontend to CORTEX + GNN server |
| **Dashboard UI** | 3000 | `npm run dev` | React frontend — graph visualization, stats, structural reasoning view |
| **Cloudflared** | random | `cloudflared tunnel` | (Optional) Exposes GNN to internet for Claude.ai access |

Pillars 2 (POMDP) and 3 (Mamba) don't run as services — they're invoked by the orchestrator scripts on demand.

---

## Optional: Claude.ai Access (Cloudflared Tunnel)

Only needed if you want Claude.ai (not Claude Code) to use the GNN tools.

```bash
# Start tunnel
cloudflared tunnel --url http://localhost:5070 &

# Watch output for the tunnel URL (e.g., https://xxx.trycloudflare.com)
# Then update Supabase secret with new URL:
cd /mnt/vault/cortex
supabase secrets set GNN_SERVER_URL=https://NEW-URL.trycloudflare.com

# IMPORTANT: After setting secrets, reconnect the CORTEX connector
# in Claude.ai → Settings → Integrations (secrets restart kills sessions)
```

Claude.ai connector URLs:
- **CORTEX:** `https://lqpvskwevanpgywdksqu.supabase.co/functions/v1/cortex-mcp?key=47c9bd8e01420a7274801548aab409aed6ab3d3614271a96307aadc30bd0065d`
- **SABLE GNN:** `https://lqpvskwevanpgywdksqu.supabase.co/functions/v1/sable-gnn?key=parallax-gnn-2026`

---

## Running Diagnostics (POMDP + Mamba + GNN)

```bash
source /home/b1tr0n1n/ml-env/bin/activate

# Run the orchestrator demo (core switch failure)
cd /mnt/vault/sable/orchestrator
python sable_core.py --device cuda

# Run the complex scenario (Monday Morning Meltdown, 44 nodes)
python complex_scenario.py

# Run cognitive diagnostic on CORTEX knowledge graph
python cortex_diagnostic.py --device cuda

# Run CORTEX graph repair (dry-run first!)
python cortex_repair.py --dry-run --device cuda
# Then execute:
python cortex_repair.py --execute --device cuda
```

---

## Running Benchmarks & Tests

```bash
source /home/b1tr0n1n/ml-env/bin/activate

# GNN benchmark (4 baselines vs GNN on CORTEX test split)
cd /mnt/vault/sable/pillar1
python benchmark.py --with-gnn --device cuda

# Domain portability test (CORTEX-trained GNN on infrastructure)
python domain_portability_test.py --device cuda

# POMDP hard scenarios
cd /mnt/vault/sable/pillar2
python hard_scenarios.py --rollouts 500

# Contradiction finder
cd /mnt/vault/sable/pillar1
python find_contradictions.py --top 20 --threshold 0.15

# Fusion tests
cd /mnt/vault/sable/orchestrator
python fusion_test.py --device cuda           # Static fusion
python sequential_fusion_test.py --device cuda # Sequential fusion
```

---

## Re-Training Models

### GNN (Pillar 1) — after CORTEX grows or after graph repair

```bash
source /home/b1tr0n1n/ml-env/bin/activate
cd /mnt/vault/sable/pillar1

# Kill running GNN server first
pkill -f "gnn_server.py"

# Re-export graph from CORTEX (read-only pull)
python cortex_gnn_export.py --output cortex_graph.pt

# Re-train
python cortex_gnn_model.py --data cortex_graph.pt --epochs 200 --device cuda

# Restart GNN server with new checkpoint
python gnn_server.py --port 5070 --device cuda &
```

### Mamba (Pillar 3) — if you want to regenerate training data

```bash
source /home/b1tr0n1n/ml-env/bin/activate
cd /mnt/vault/sable/pillar3

# Generate new cascade sequences
python generate_temporal_data_v3.py --count 15000 --output temporal_v3_tuned.pt

# Retrain
python sable_mamba_final.py --data temporal_v3_tuned.pt --epochs 100 --device cuda
```

---

## Shutdown

```bash
pkill -f "gnn_server.py"
pkill -f "api.py"
pkill -f "vite"
pkill -f cloudflared
```

Or kill everything SABLE-related:

```bash
pkill -f "gnn_server\|api.py\|vite\|cloudflared"
```

---

## File Map

```
/mnt/vault/sable/
├── STARTUP.md                           ◄── THIS FILE
├── SESSION-1-SUMMARY.md
├── SESSION-2-SUMMARY.md
├── PROJECT-PARALLAX-OPS-MANUAL.md
│
├── pillar1/                             ◄── GNN (Structural Reasoning)
│   ├── cortex_gnn_export.py             # CORTEX → PyG data pipeline
│   ├── cortex_graph.pt                  # Exported dataset
│   ├── cortex_gnn_model.py              # SableGNN model + training
│   ├── checkpoints/best_model.pt        # Trained checkpoint
│   ├── gnn_server.py                    # HTTP server (port 5070)
│   ├── gnn_mcp_server.py               # Claude Code MCP wrapper
│   ├── benchmark.py                     # 4-baseline benchmark suite
│   ├── domain_portability_test.py       # Cross-domain transfer test
│   └── find_contradictions.py           # Contradiction surfacing
│
├── pillar2/                             ◄── POMDP (Decision Planning)
│   ├── pomcp.py                         # POMCP solver + BeliefState
│   └── hard_scenarios.py                # 3 hard diagnostic scenarios
│
├── pillar3/                             ◄── Mamba (Temporal Prediction)
│   ├── sable_mamba.py                   # Pure-PyTorch SSM core
│   ├── sable_mamba_final.py             # Final cascade predictor
│   ├── generate_temporal_data.py        # Sequence generator v1
│   ├── generate_temporal_data_v3.py     # Cascade outcome generator
│   ├── temporal_v3_tuned.pt             # Training data (176 MB)
│   └── checkpoints/best_mamba_final.pt  # Trained checkpoint
│
├── orchestrator/                        ◄── Integration Layer
│   ├── sable_core.py                    # Three-pillar orchestrator
│   ├── complex_scenario.py              # Monday Morning Meltdown (44 nodes)
│   ├── cortex_diagnostic.py             # Cognitive graph analysis
│   ├── cortex_repair.py                 # GNN-driven graph repair
│   ├── fusion_test.py                   # Static fusion experiment
│   └── sequential_fusion_test.py        # Dynamic fusion experiment
│
├── dashboard/                           ◄── Frontend
│   ├── api.py                           # FastAPI backend (port 3001)
│   ├── src/App.jsx                      # React dashboard
│   ├── src/index.css                    # b1tr0n1n design system
│   └── package.json
│
├── sable_sim/                           ◄── Infrastructure Simulator
│   ├── core/                            # Component, graph, state, dependency
│   ├── simulation/                      # Propagation, fog-of-war, failure injection
│   ├── export/                          # PyG/JSON export
│   └── generation/                      # Topology templates
│
└── .git/

/mnt/vault/cortex/supabase/functions/
├── cortex-mcp/                          ◄── CORTEX MCP (13 tools)
│   ├── index.ts                         # Edge Function (includes capture_document)
│   └── agents/true-agents/
│       ├── shared/                      # Extracted agent infrastructure
│       ├── convergence_scanner.py
│       └── retroactive_linker.py
└── sable-gnn/                           ◄── GNN Edge Function (Claude.ai)
    └── index.ts
```

---

## Ports Reference

| Port | Service | Required? |
|------|---------|-----------|
| 3000 | Dashboard frontend | For UI |
| 3001 | Dashboard API | For UI |
| 5070 | GNN server | For GNN tools + diagnostics |
| 8080 | llama-server | Only for convergence scanner / retroactive linker |

---

## Credentials

| What | Value | Where |
|------|-------|-------|
| CORTEX MCP Key | `47c9bd8e01420a...` | Supabase env `MCP_ACCESS_KEY` |
| GNN MCP Key | `parallax-gnn-2026` | Supabase env `GNN_ACCESS_KEY` |
| Supabase Service Key | In `shared/config.py` | REST API calls |

---

## Troubleshooting

**GNN server won't start:** `source /home/b1tr0n1n/ml-env/bin/activate` first. Check GPU: `nvidia-smi`. Check port: `lsof -i :5070`.

**Dashboard shows "Loading":** Is the API running? `curl http://localhost:3001/api/health`. Is CORTEX accessible?

**Dashboard graph not loading:** Large graph takes a few seconds. Check browser console for errors. API must be on port 3001.

**Claude.ai CORTEX connector fails after deploy:** `supabase secrets set` restarts ALL Edge Functions. Reconnect in Claude.ai → Settings → Integrations.

**Tunnel URL changed:** Quick tunnels get new URLs every restart. Update: `supabase secrets set GNN_SERVER_URL=https://NEW-URL.trycloudflare.com`

**POMDP/orchestrator scripts fail:** Make sure you're in the right directory and ml-env is activated. These scripts use relative imports.

**Import errors on orchestrator scripts:** They add parent directories to sys.path. Run from inside the script's directory.
