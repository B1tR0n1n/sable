# Project PARALLAX — Operations Manual

### Codename: PARALLAX | Architecture: SABLE | Three perspectives, one truth.

---

## What Was Built (March 22, 2026)

In one session, we built the first operational pillar of SABLE — a domain-portable cognitive architecture that uses three neural pillars (GNN, POMDP, Mamba) sharing a common latent state space. Pillar 1 (Causal Graph Reasoning) is now live, serving structural link suggestions from your GPU to both Claude Code and Claude.ai.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        SABLE Architecture                       │
│                                                                 │
│  Pillar 1 (DONE)        Pillar 2 (NEXT)      Pillar 3 (LATER) │
│  ┌──────────────┐       ┌──────────────┐      ┌──────────────┐ │
│  │ GNN/GAT      │       │ POMDP/MCTS   │      │ Mamba/SSM    │ │
│  │ Structural   │       │ Decision     │      │ Temporal     │ │
│  │ Reasoning    │       │ Under        │      │ State        │ │
│  │              │       │ Uncertainty  │      │ Modeling     │ │
│  │ 5.6M params  │       │ CPU-based    │      │ 100-500M     │ │
│  │ RTX 5070     │       │ 9950X3D      │      │ RTX 5090     │ │
│  └──────┬───────┘       └──────────────┘      └──────────────┘ │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │ CORTEX       │  ◄── Shared Knowledge Graph (1327 thoughts)  │
│  │ (Supabase)   │      1164 typed directional links             │
│  └──────────────┘      12 MCP tools + 4 GNN tools               │
└─────────────────────────────────────────────────────────────────┘
```

### What Each Component Does

| Component | What It Is | What It Does |
|-----------|-----------|-------------|
| `cortex_gnn_export.py` | Data pipeline | Pulls CORTEX graph from Supabase, converts to PyTorch Geometric format |
| `cortex_graph.pt` | Dataset | 1327 nodes (1030-dim features), 1164 edges (8-dim features), train/val/test splits |
| `cortex_gnn_model.py` | Neural network | EdgeConditionedGAT with 3 task heads: link type (7-class), link prediction (binary), contradiction detection (binary) |
| `checkpoints/best_model.pt` | Trained weights | Best model checkpoint (epoch 75, AUC 0.991 link prediction, AUC 0.96 contradiction) |
| `gnn_server.py` | HTTP server | Loads trained GNN, serves predictions via REST API on port 5070 |
| `gnn_mcp_server.py` | MCP wrapper | Translates Claude Code MCP tool calls into HTTP requests to gnn_server |
| `sable-gnn/index.ts` | Edge Function | Supabase Edge Function that proxies Claude.ai requests through cloudflared to gnn_server |
| `find_contradictions.py` | Analysis tool | Scores all edges for contradictions, surfaces candidates for review |
| `shared/` | Common library | Extracted infrastructure (config, colors, clients, parsers, models) for all agents |

---

## File Locations

```
/mnt/vault/sable/pillar1/                    ◄── PILLAR 1 (GNN)
├── cortex_gnn_export.py                      # CORTEX → PyG data pipeline
├── cortex_graph.pt                           # Exported dataset (5.5 MB)
├── cortex_gnn_model.py                       # SableGNN model + training
├── checkpoints/
│   └── best_model.pt                         # Trained model checkpoint
├── gnn_server.py                             # HTTP server (port 5070)
├── gnn_mcp_server.py                         # Claude Code MCP wrapper
├── find_contradictions.py                    # Contradiction surfacing
├── STARTUP.md                                # Quick start guide
└── PROJECT-PARALLAX-OPS-MANUAL.md            # This file

/mnt/vault/cortex/supabase/functions/
├── cortex-mcp/                               ◄── CORTEX MCP (existing, 12 tools)
│   └── agents/true-agents/
│       ├── shared/                           # Extracted shared infrastructure
│       │   ├── __init__.py
│       │   ├── config.py                     # Supabase URLs, keys, model paths
│       │   ├── colors.py                     # b1tr0n1n ANSI palette
│       │   ├── clients.py                    # CortexMCPClient, CortexRESTClient, LLMClient
│       │   ├── parsers.py                    # JSON action parsing from LLM responses
│       │   └── models.py                     # Thought, Link, Convergence, AgentState
│       ├── convergence_scanner.py            # Agentic convergence scanner (Nemotron)
│       └── retroactive_linker.py             # Batch link classifier (Nemotron)
└── sable-gnn/                                ◄── GNN MCP Edge Function (for Claude.ai)
    └── index.ts

/mnt/vault/sable/                             ◄── SABLE PROJECT ROOT
├── sable_sim/                                # Infrastructure simulator (20 files)
├── data/                                     # Simulation data
├── pillar1/                                  # GNN (this session)
└── .git/                                     # Git repo

/home/b1tr0n1n/ml-env/                        ◄── PYTHON ML ENVIRONMENT
                                              # torch 2.10+cu129, torch_geometric 2.7.0
                                              # httpx, networkx, numpy

/mnt/vault/models/
├── nemotron-nano/                            # Nemotron-3-Nano-30B (for agent loops)
└── sable-v1/                                 # QLoRA fine-tune (needs verbosity fix)
```

---

## Startup Procedures

### Cold Boot — Full Stack

Run these in order after turning on the machine:

```bash
# 1. Activate ML environment
source /home/b1tr0n1n/ml-env/bin/activate

# 2. Start GNN server (loads model onto GPU, ~3 seconds)
cd /mnt/vault/sable/pillar1
python gnn_server.py --port 5070 --device cuda &

# 3. Verify GNN server is running
curl http://localhost:5070/health
# Should return: {"status": "ok", "model": "SableGNN"}

# 4. Start cloudflared tunnel (for Claude.ai access)
cloudflared tunnel --url http://localhost:5070 &
# Watch output for the tunnel URL (e.g., https://xxx.trycloudflare.com)

# 5. Update Supabase secret with new tunnel URL (it changes each restart)
cd /mnt/vault/cortex
supabase secrets set GNN_SERVER_URL=https://NEW-TUNNEL-URL.trycloudflare.com

# 6. (Optional) Start llama-server for agent scripts
~/llama.cpp/build/bin/llama-server \
  -m /mnt/vault/models/nemotron-nano/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf \
  --port 8080 -ngl 99 --flash-attn on &
```

### Quick Start — Claude Code Only (No Cloud)

```bash
source /home/b1tr0n1n/ml-env/bin/activate
cd /mnt/vault/sable/pillar1
python gnn_server.py --port 5070 --device cuda &
```

That's it. Claude Code's `sable-gnn` MCP picks up the local server automatically.

### Shutdown

```bash
# Kill GNN server
pkill -f "gnn_server.py"

# Kill cloudflared
pkill -f cloudflared

# Kill llama-server (if running)
pkill -f llama-server
```

---

## How To Use the GNN Tools

### In Claude Code

Start Claude Code from any directory. The `sable-gnn` MCP server is registered globally. Just talk naturally:

**Suggest links:**
> "Use the GNN to suggest links for thought ccaf7584-e2d1-41a9-8ce8-5e8f099f6ca2"
> "What does the GNN think should link to my SABLE architecture thought?"

**Score a pair:**
> "Score the connection between thought X and thought Y using the GNN"

**Find contradictions:**
> "What contradictions does the GNN find in CORTEX?"
> "Run the GNN contradiction detector"

**Get stats:**
> "Show me GNN model stats"

Claude Code translates these into MCP tool calls automatically. The tools are:
- `gnn_suggest_links` — structural link suggestions with predicted relation types
- `gnn_score_pair` — score a specific thought pair
- `gnn_find_contradictions` — surface contradiction candidates
- `gnn_stats` — model and graph stats

### In Claude.ai

Go to **claude.ai → Settings → Integrations** and add a connector:
- **URL:** `https://lqpvskwevanpgywdksqu.supabase.co/functions/v1/sable-gnn?key=parallax-gnn-2026`
- **Name:** SABLE GNN

Then use the same natural language in any conversation. Claude.ai will call the tools through the Edge Function → cloudflared tunnel → your local GNN server.

**Requirement:** Your machine must be on with `gnn_server.py` and `cloudflared` running.

### Direct HTTP (curl, scripts, other agents)

```bash
# Suggest links
curl -X POST http://localhost:5070/suggest \
  -H "Content-Type: application/json" \
  -d '{"thought_id": "UUID-HERE", "top_k": 10, "min_confidence": 0.5}'

# Score a pair
curl -X POST http://localhost:5070/score \
  -H "Content-Type: application/json" \
  -d '{"source_id": "UUID-1", "target_id": "UUID-2"}'

# Find contradictions
curl -X POST http://localhost:5070/contradictions \
  -H "Content-Type: application/json" \
  -d '{"top_k": 15, "threshold": 0.3}'

# Health check
curl http://localhost:5070/health

# Stats
curl http://localhost:5070/stats
```

---

## Request Flow Diagrams

### Claude Code (Local)

```
You → Claude Code → gnn_mcp_server.py (stdio) → HTTP → gnn_server.py:5070 → GPU → response
```

### Claude.ai (Cloud)

```
You → Claude.ai → Supabase Edge Function (sable-gnn)
                        ↓
                  cloudflared tunnel (https://xxx.trycloudflare.com)
                        ↓
                  gnn_server.py:5070 (your machine)
                        ↓
                  RTX 5070 (GPU inference, ~4-28ms)
                        ↓
                  response back up the chain
```

### Convergence Scanner / Other Agents (Local)

```
Agent script → HTTP → gnn_server.py:5070 → GPU → response
```

---

## Re-Training the GNN

Do this when CORTEX has grown significantly (new thoughts, new links from retroactive linker sessions).

```bash
source /home/b1tr0n1n/ml-env/bin/activate
cd /mnt/vault/sable/pillar1

# 1. Kill running GNN server
pkill -f "gnn_server.py"

# 2. Re-export the graph (read-only pull from CORTEX)
python cortex_gnn_export.py --output cortex_graph.pt

# 3. Re-train
python cortex_gnn_model.py --data cortex_graph.pt --epochs 200 --device cuda

# 4. Restart GNN server with new checkpoint
python gnn_server.py --port 5070 --device cuda &
```

### Stats-Only Mode (Check Graph Without Exporting)

```bash
python cortex_gnn_export.py --stats
```

### Find Contradictions After Re-Training

```bash
python find_contradictions.py --top 20 --threshold 0.15
python find_contradictions.py --unlinked --top 30    # Also check unlinked pairs
```

---

## Model Performance (Current Checkpoint)

| Task | Metric | Score |
|------|--------|-------|
| **Link Prediction** | AUC-ROC | **0.991** |
| **Link Prediction** | Accuracy | 0.960 |
| **Link Prediction** | F1 | 0.884 |
| **Link Type (7-class)** | Macro F1 | 0.200 |
| **Link Type — supports** | F1 | 0.551 |
| **Link Type — elaborates** | F1 | 0.465 |
| **Link Type — depends_on** | F1 | 0.222 |
| **Link Type — related** | F1 | 0.167 |
| **Contradiction Detection** | AUC-ROC | **0.960** |

Link prediction is production-ready. Link type classification works well for common types, struggles with rare types (contradicts: 16 examples, supersedes: 11 examples). Contradiction detection ranks correctly (AUC 0.96) but can't threshold on tiny sample sizes — use the ranked output from `find_contradictions.py`.

---

## Credentials & Keys

| What | Value | Where Used |
|------|-------|-----------|
| Supabase URL | `https://lqpvskwevanpgywdksqu.supabase.co` | All agents, export scripts |
| Supabase Service Key | `eyJhbGciOiJIUzI1NiI...` (in config.py) | REST API calls |
| CORTEX MCP Key | `47c9bd8e01420a7274...` | MCP tool calls |
| GNN MCP Access Key | `parallax-gnn-2026` | Edge Function auth |
| GNN Server Port | `5070` | Local HTTP server |
| llama-server Port | `8080` | Nemotron inference |

---

## Edge Function Deployment

The `sable-gnn` Edge Function is deployed separately from `cortex-mcp`. It lives at:
- **Code:** `/mnt/vault/cortex/supabase/functions/sable-gnn/index.ts`
- **URL:** `https://lqpvskwevanpgywdksqu.supabase.co/functions/v1/sable-gnn`
- **Auth:** `?key=parallax-gnn-2026` or header `x-gnn-key: parallax-gnn-2026`

### Re-Deploy After Changes

```bash
cd /mnt/vault/cortex
supabase functions deploy sable-gnn --no-verify-jwt
```

### Update Tunnel URL (After cloudflared Restart)

```bash
cd /mnt/vault/cortex
supabase secrets set GNN_SERVER_URL=https://NEW-URL.trycloudflare.com
```

Quick tunnels get a new URL every restart. For a permanent URL:

```bash
# One-time: create a named tunnel (requires free Cloudflare account)
cloudflared tunnel create sable-gnn
cloudflared tunnel route dns sable-gnn gnn.yourdomain.com

# Then start with:
cloudflared tunnel run sable-gnn
```

---

## Competitive Position (Validated March 22, 2026)

Two deep research passes confirmed: **nothing like SABLE exists.**

No system integrates GNN + POMDP + SSM/Mamba through a shared latent state space as a general cognitive architecture. Closest systems cover at most 2 of 6 defining characteristics.

| Whitespace Claim | Status |
|-----------------|--------|
| Three-pillar neural integration through shared latent space | Zero papers, zero products |
| Architecture-first approach to scaling plateau | Contrarian to industry (LLM+agents) |
| POMDP-native reasoning in any diagnostic domain | Zero papers in IT/infrastructure |
| Consumer hardware cognitive architecture (100-500M params) | Empty design space |
| Domain portability by design (neural, not symbolic) | No existing system |

Key papers informing implementation: CausalMamba (arXiv:2511.16191), GammaZero (arXiv:2510.14035), DyG-Mamba (arXiv:2408.06966), STG-Mamba (arXiv:2403.12418), Causal POMDPs (arXiv:2602.23545).

---

## Build Roadmap

| Phase | Sessions | Status | What |
|-------|----------|--------|------|
| **P1: GNN** | 1-5 | **DONE** | Data export, model, training, shared infra, MCP integration |
| **P2: POMDP** | 6-9 | NEXT | POMCP solver, decision journal agent, contact intel agent |
| **P3: Mamba** | 10-13 | PLANNED | Temporal dataset, Mamba architecture, training |
| **P4: Integration** | 14-16 | PLANNED | Shared latent state, orchestrator, end-to-end demo |
| **P5: Refinement** | 17-18 | PLANNED | sable-v1 fix, feedback loop |

Full plan: `/home/b1tr0n1n/.claude/plans/composed-brewing-sunbeam.md`

---

## Troubleshooting

**GNN server won't start:**
- Is `ml-env` activated? `source /home/b1tr0n1n/ml-env/bin/activate`
- Is GPU available? `nvidia-smi`
- Is port 5070 in use? `lsof -i :5070`

**MCP tools not showing in Claude Code:**
- GNN server must be running first
- MCP registered at `/mnt/vault/sable/pillar1/` scope — check with `claude mcp list`
- Restart Claude Code after registering

**Claude.ai can't reach GNN:**
- Is cloudflared running? `pgrep cloudflared`
- Did the tunnel URL change? Check `cloudflared` output for new URL
- Update the secret: `supabase secrets set GNN_SERVER_URL=https://NEW-URL`

**Export shows 0 nodes:**
- Embeddings come back as strings from pgvector — the export script handles this
- Check Supabase is accessible: `curl https://lqpvskwevanpgywdksqu.supabase.co/rest/v1/thoughts?limit=1 -H "apikey: YOUR_KEY"`

**Training loss not decreasing:**
- Check data loaded correctly: `python -c "import torch; d=torch.load('cortex_graph.pt', weights_only=False); print(d)"`
- Try lower learning rate: `--lr 0.0005`

**CORTEX connector fails after deploying/changing secrets:**
- `supabase secrets set` and `supabase functions deploy` restart all Edge Functions
- The Claude.ai connector's session goes stale
- Fix: disconnect and reconnect the CORTEX connector in Claude.ai → Settings → Integrations

**Tunnel URL keeps changing:**
- Expected with quick tunnels. Set up a named tunnel for permanent URL
- Or use a cron job to update the secret on restart
