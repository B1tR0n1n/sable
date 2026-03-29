# Project PARALLAX — Pillar 1 Startup Guide

Everything runs from `/mnt/vault/sable/pillar1/`.

---

## Quick Start (3 commands)

```bash
# 1. Activate the ML environment
source /home/b1tr0n1n/ml-env/bin/activate

# 2. Start the GNN server (runs on GPU, port 5070)
cd /mnt/vault/sable/pillar1
python gnn_server.py --port 5070 --device cuda &

# 3. (Optional) Start the cloudflared tunnel for Claude.ai access
cloudflared tunnel --url http://localhost:5070 &
# Copy the tunnel URL and set it in Supabase secrets if changed
```

That's it. Claude Code picks up the `sable-gnn` MCP server automatically.

---

## What Each Piece Does

| Component | Command | Port | What It Does |
|-----------|---------|------|-------------|
| **GNN Server** | `python gnn_server.py` | 5070 | Loads trained model, serves link suggestions via HTTP |
| **MCP Server** | (automatic via Claude Code) | stdio | Translates MCP tool calls → GNN HTTP calls |
| **Cloudflared** | `cloudflared tunnel --url http://localhost:5070` | random | Exposes GNN server to internet for Claude.ai |

---

## Verify Everything Works

```bash
# Health check
curl http://localhost:5070/health

# Stats
curl http://localhost:5070/stats

# Test a suggestion (use any thought UUID from CORTEX)
curl -X POST http://localhost:5070/suggest \
  -H "Content-Type: application/json" \
  -d '{"thought_id": "ccaf7584-e2d1-41a9-8ce8-5e8f099f6ca2", "top_k": 5}'
```

---

## File Inventory

```
/mnt/vault/sable/pillar1/
├── cortex_gnn_export.py      # Exports CORTEX → PyG dataset
├── cortex_graph.pt           # The exported dataset (5.5 MB)
├── cortex_gnn_model.py       # SableGNN model + training loop
├── checkpoints/
│   └── best_model.pt         # Trained model checkpoint
├── gnn_server.py             # HTTP server exposing the GNN
├── gnn_mcp_server.py         # MCP wrapper for Claude Code
├── find_contradictions.py    # Contradiction surfacing tool
└── STARTUP.md                # This file
```

```
/mnt/vault/cortex/supabase/functions/
├── cortex-mcp/               # Existing CORTEX MCP (12 tools)
│   └── agents/true-agents/
│       ├── shared/            # Extracted shared infrastructure
│       │   ├── config.py
│       │   ├── colors.py
│       │   ├── clients.py
│       │   ├── parsers.py
│       │   └── models.py
│       ├── convergence_scanner.py
│       └── retroactive_linker.py
└── sable-gnn/                # GNN MCP Edge Function (for Claude.ai)
    └── index.ts
```

---

## Re-training the GNN (after CORTEX grows)

```bash
source /home/b1tr0n1n/ml-env/bin/activate
cd /mnt/vault/sable/pillar1

# 1. Re-export the graph (pulls latest from CORTEX, read-only)
python cortex_gnn_export.py --output cortex_graph.pt

# 2. Re-train (overwrites checkpoint)
python cortex_gnn_model.py --data cortex_graph.pt --epochs 200 --device cuda

# 3. Restart the GNN server to load new checkpoint
# Kill old server, then:
python gnn_server.py --port 5070 --device cuda &
```

---

## Claude.ai Connector Setup (One-Time)

After deploying the Edge Function and starting the tunnel:

1. Go to Claude.ai → Settings → Connectors
2. Add new connector:
   - URL: `https://lqpvskwevanpgywdksqu.supabase.co/functions/v1/sable-gnn?key=parallax-gnn-2026`
   - Name: `SABLE GNN`
3. The tunnel URL changes each restart unless you set up a named tunnel

---

## Troubleshooting

**GNN server won't start:** Check that `ml-env` is activated and GPU is available (`nvidia-smi`).

**MCP tools not showing in Claude Code:** Make sure GNN server is running first, then restart Claude Code. The MCP server was registered at `/mnt/vault/sable/pillar1/`.

**Tunnel URL changed:** Quick tunnels get a new URL each time. Update the Supabase secret:
```bash
supabase secrets set GNN_SERVER_URL=https://NEW-URL.trycloudflare.com
```
For a permanent URL, set up a named Cloudflare tunnel with `cloudflared tunnel create sable-gnn`.
