# SABLE Engine — Docker + Dashboard Build Instructions

## What You're Building

A Docker container that packages the entire SABLE reasoning engine with a web-based dashboard. When someone runs `docker run --gpus all -p 8080:8080 sable-engine`, a browser dashboard opens showing the engine processing infrastructure cascades in real time. The dashboard follows the b1tr0n1n brand identity exactly.

---

## Architecture

```
Docker Container
├── SABLE Engine (Python, PyTorch, CUDA)
│   ├── GNN (pillar 1) — topology expert
│   ├── POMDP (pillar 2) — belief state expert  
│   ├── Mamba (pillar 3) — temporal expert
│   ├── Staged Fusion v3 — sharp router + hard routing
│   └── Temporal Chain — verdict feedback loop
├── FastAPI server (port 8080)
│   ├── /api/status — engine state
│   ├── /api/tick — advance one inference cycle
│   ├── /api/scenario — load a scenario
│   ├── /api/node/{id} — per-node detail
│   └── WebSocket /ws — live streaming updates
└── Static dashboard (served at localhost:8080)
    └── Single HTML file with embedded JS/CSS
```

---

## Part 1: The Engine Wrapper

Create `sable_engine.py` in the fusion directory. This is the single entry point for all inference.

```python
class SableEngine:
    def __init__(self, device='cuda')
    def load_checkpoints(self, checkpoint_dir)
    def reset_state(self, n_nodes)
    def infer_temporal(self, gnn, pomdp, mamba) -> dict
    def get_node_report(self, node_idx) -> str
    def get_summary(self) -> dict
```

Key requirements:
- Temporal state persists across `infer_temporal()` calls automatically.
- `get_summary()` returns a dict with: per-node predictions, per-node confidence, per-node trajectory (last 8 verdicts), class counts, router decisions, cycle number.
- All inference runs `@torch.no_grad()`.
- Load checkpoints from: `checkpoints/fusion.pt` (staged fusion v3) and `checkpoints/temporal.pt` (temporal chain).

---

## Part 2: The Scenario Generator

Create `scenarios.py`. This provides pre-built cascade scenarios for demo mode and the ability to generate random ones.

Built-in scenarios (each returns per-tick features for all three pillars):

1. **Monday Morning Meltdown** — 40 nodes, 25 ticks. The flagship demo. A routine Monday morning turns catastrophic. Starts fully healthy, subtle degradation begins at tick 3 on a database cluster, cascade failure hits the application tier at tick 8, dependent services go unreachable by tick 12, a partial recovery attempt at tick 18 restores the root cause but leaves orphaned failures downstream, and oscillating nodes appear throughout as load balancers flap between healthy backends and dead ones. This scenario exercises every class, every expert, and every capability of the temporal chain. It's the scenario that scores 0.842 macro F1 — the system's best performance under compound stress.

2. **Silent Killer** — 40 nodes, 20 ticks. A single critical node fails with zero direct observability — no health check reaches it, no telemetry comes back. The system has to detect the failure purely from downstream effects: neighbors start degrading, dependent services show latency spikes, the topology develops a hole. This is the POMDP's showcase — reasoning about what you can't see from what you can. Proves the belief-state layer earns its place in the architecture.

3. **Cascade Whiplash** — 40 nodes, 15 ticks. A root-cause node fails at tick 3, triggering a cascade through its dependents. At tick 8, the root cause recovers — someone rebooted it. But the dependents stay dead because they've already crashed and need manual intervention. The system must correctly show: root = healthy (recovered), dependents = still failed (not recovered). This tests whether the temporal chain understands causality — a recovered cause doesn't mean recovered effects.

4. **Slow Poison** — 40 nodes, 20 ticks. Eight nodes degrade by 5% per tick. No sudden failures, no dramatic cascades — just a slow, steady decline that's invisible in any single snapshot. The system needs temporal context to catch it: each tick looks almost identical to the last, but over 10 ticks the trajectory is unmistakable. This is the scenario that proves the temporal chain detects trends, not just states.

5. **Random Chaos** — 40 nodes, configurable ticks. Random failures, recoveries, oscillations, and cascades. No two runs are identical. Nodes fail without warning, recover unexpectedly, and oscillate at random intervals. This is the stress test for generalization — the system can't memorize this scenario because it never repeats. If the engine handles Random Chaos well, it can handle production.

Each scenario must use `sable_sim` to generate realistic cascade physics. The features come from running all three pillars on the sim output, with fog-of-war applied (30% observability, 15% noise, 2-tick delay) to match training conditions.

Each scenario should display its description in the dashboard when selected from the dropdown — one or two sentences explaining what the user is about to see and what to watch for.

---

## Part 3: The FastAPI Server

Create `server.py`. Lightweight HTTP + WebSocket server.

```
GET  /                  → serves the dashboard HTML
GET  /api/status        → { engine loaded, scenario, cycle, n_nodes }
POST /api/scenario      → { name: "monday_morning" } → loads and resets
POST /api/tick          → advances one inference cycle, returns full state
POST /api/autoplay      → { speed: 1.0 } → starts/stops automatic ticking
GET  /api/node/{id}     → full node detail (trajectory, confidence, probs)
GET  /api/scenarios     → list available scenarios
WS   /ws                → streams tick results in real time during autoplay
```

The server should:
- Load the engine on startup
- Pre-load the Monday Morning Meltdown scenario by default
- Support autoplay mode where ticks advance automatically at configurable speed (default 1 tick/second)
- Stream results over WebSocket during autoplay so the dashboard updates live
- All responses are JSON

Dependencies: `fastapi`, `uvicorn`, `websockets`

---

## Part 4: The Dashboard

Single HTML file served at `GET /`. This is the most important part — it's what people see.

### Brand Identity (MANDATORY)

Follow the b1tr0n1n brand guidelines exactly:

**Colors:**
- Background: `#0a0908` (near-black)
- Panel backgrounds: `#0f0e0b`, `#161410`
- Borders: `#2a2620`
- Primary text: `#c8bda0` (warm parchment)
- Dim text: `#7a7060`
- Bright text: `#ede5d0`
- Gold accent: `#c9a227` (used ONLY for active states, dividers, highlights — earned, not sprinkled)
- Dim gold: `#8b7320` (labels, section markers)
- Success: `#4a7a45`
- Danger: `#a63d2f`

**Typography:**
- Serif: `'Cormorant Garamond', Georgia, serif` — all prose, titles, headings. This is the thinking font.
- Mono: `'JetBrains Mono', 'Fira Code', monospace` — labels, data, metrics, node IDs. This is the operational font.
- Load both from Google Fonts.
- Never swap roles. Mono prose or serif labels violate the identity.

**Visual Texture:**
- Grid background: fixed div with 80px grid lines using border color at 6% opacity
- No gradients, no shadows, no glow effects
- Gold is earned — it marks what matters (active states, critical alerts, section dividers)
- Dark background is non-negotiable. No light mode.

**Section labels:** Format as `"01 — TOPOLOGY"` — two-digit number, em dash, uppercase title. Font: mono, 9px, letter-spacing 5px, color dim gold.

### Dashboard Layout

The dashboard is a single page with four panels arranged in a grid:

```
┌─────────────────────────────────┬──────────────────────────┐
│                                 │                          │
│     TOPOLOGY MAP                │    SYSTEM STATUS         │
│     (node grid with states)     │    (class counts,        │
│                                 │     router decisions,    │
│                                 │     confidence avg)      │
│                                 │                          │
├─────────────────────────────────┼──────────────────────────┤
│                                 │                          │
│     CASCADE TIMELINE            │    NODE DETAIL           │
│     (tick-by-tick progression,  │    (click a node to see  │
│      state changes over time)   │     trajectory, probs,   │
│                                 │     confidence history)  │
│                                 │                          │
└─────────────────────────────────┴──────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  CONTROLS: [▶ Play] [⏸ Pause] [⏭ Step] [Speed ▼] [Scenario ▼]  │
└────────────────────────────────────────────────────────────┘
```

### Panel 1: Topology Map (top-left, ~60% width)

Grid of node cells. Each node is a small rectangle showing:
- Node ID (mono, 10px)
- Current state as background color:
  - Healthy: dim green `#4a7a45` at 30% opacity
  - Degraded: gold `#c9a227` at 30% opacity  
  - Failed: red `#a63d2f` at 50% opacity
  - Unreachable: fully dark `#0a0908` with dashed border
  - Oscillating: alternating pulse animation between green and red at 40% opacity
- Confidence as opacity intensity — higher confidence = more saturated color
- Click a node to select it (gold border highlight) and show detail in Panel 4

Nodes should be arranged in a grid that roughly matches the topology. If the sim provides adjacency info, draw thin lines between connected nodes (border color, 0.5px).

When a node changes state between ticks, flash the gold accent briefly (0.3s) to draw attention to the transition.

### Panel 2: System Status (top-right, ~40% width)

**Class Distribution:**
```
01 — CLASS DISTRIBUTION

healthy      ████████████████████████  28
degraded     ████                       4
failed       ██████                     6
unreachable  ██                         2
oscillating  ─                          0

Cycle: 14 / 25
```

Bar lengths proportional to count. Bar color matches the node state colors.

**Router Decisions:**
```
02 — ROUTER DECISIONS

healthy     → Fusion    (83%)
degraded    → Fusion    (100%)
failed      → Fusion    (98%)
unreachable → POMDP     (54%)
oscillating → Mamba     (71%)
```

Show which expert the router is trusting for each class. Update each tick.

**System Confidence:**
```
03 — CONFIDENCE

Average:  0.847
Lowest:   Node 17 — 0.412 (degraded)
Highest:  Node 03 — 0.991 (healthy)
```

### Panel 3: Cascade Timeline (bottom-left, ~60% width)

Horizontal timeline showing all ticks. Current tick highlighted with gold marker.

Below the timeline, a per-node state history rendered as a heatmap/grid:
- Rows = nodes (show node ID on left)
- Columns = ticks  
- Cell color = state at that tick (same colors as topology map)
- Current tick column has gold left border

This lets you see the cascade propagate visually — a wave of red spreading across the grid as nodes fail.

Only show nodes that have changed state at least once (don't waste space on 28 nodes that stayed healthy the whole time). Sort by first-change tick so the cascade reads top-to-bottom as it propagated.

### Panel 4: Node Detail (bottom-right, ~40% width)

Shows when a node is clicked in the topology map.

```
04 — NODE DETAIL

Node 17                          DEGRADED
Confidence: 0.412

Probabilities:
  healthy      0.231  ████
  degraded     0.412  ████████
  failed       0.298  ██████
  unreachable  0.044  █
  oscillating  0.015  

Trajectory (last 8 cycles):
  healthy → healthy → healthy → degraded → degraded → degraded → degraded → degraded

Trend: DETERIORATING ↓

Expert routing: Fusion (100% weight)
```

### Controls Bar (bottom)

- **Play/Pause** button — starts/stops autoplay
- **Step** button — advance one tick manually
- **Speed** dropdown — 0.5x, 1x, 2x, 5x (ticks per second)
- **Scenario** dropdown — list of available scenarios
- **Reset** button — restart current scenario from tick 0

Controls use the b1tr0n1n button style: mono font, 11px, uppercase, letter-spacing 2px, border `#2a2620`, hover inverts gold/dark.

### Behavior

1. On page load: connect WebSocket, show Monday Morning Meltdown at tick 0, paused.
2. User clicks Play: ticks advance at selected speed. WebSocket streams updates. All panels animate.
3. User clicks a node: Panel 4 populates. Selection persists across ticks.
4. State change animations: when a node transitions between states, its cell flashes gold briefly. The timeline adds the new column. System status updates.
5. At the end of the scenario: autoplay stops. Display a summary overlay:
   - Per-class final F1 (comparing predictions to ground truth)
   - "SABLE correctly identified the cascade at tick X"
   - Total inference time and avg confidence

---

## Part 5: Dockerfile

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# System deps
RUN pip install fastapi uvicorn websockets --no-cache-dir

# Copy engine code
COPY engine/ /app/engine/
COPY checkpoints/ /app/checkpoints/
COPY scenarios.py /app/
COPY server.py /app/
COPY dashboard.html /app/
COPY sable_engine.py /app/

# Entry point
EXPOSE 8080
CMD ["python", "server.py"]
```

The `server.py` should:
- Print the SABLE banner on startup (gold ANSI terminal art)
- Print `SABLE Engine running at http://localhost:8080`
- Auto-open the browser if possible, fall back to printing the URL

---

## Part 6: Launcher Script

Create `run.sh` at the project root:

```bash
#!/bin/bash
echo ""
echo "  Starting SABLE Reasoning Engine..."
echo ""
docker run --gpus all -p 8080:8080 --rm sable-engine
```

And `build.sh`:

```bash
#!/bin/bash
echo ""
echo "  Building SABLE Docker image..."
echo ""
docker build -t sable-engine .
echo ""
echo "  Done. Run with: ./run.sh"
```

---

## File Structure

```
sable-docker/
├── Dockerfile
├── build.sh
├── run.sh
├── server.py              ← FastAPI + WebSocket server
├── sable_engine.py        ← Engine wrapper class
├── scenarios.py           ← Pre-built demo scenarios
├── dashboard.html         ← The b1tr0n1n dashboard (single file)
├── engine/
│   ├── pillar1/           ← GNN source
│   ├── pillar2/           ← POMDP source
│   ├── pillar3/           ← Mamba source
│   ├── fusion/            ← Fusion + temporal chain source
│   └── sim/               ← sable_sim for scenario generation
└── checkpoints/
    ├── fusion.pt
    └── temporal.pt
```

---

## Critical Requirements

1. **All inference is `@torch.no_grad()`.** No gradients during serving.
2. **Dashboard must follow b1tr0n1n brand exactly.** No light mode. No pure white text. Gold is earned. Serif thinks, mono operates.
3. **WebSocket must handle disconnects gracefully.** Don't crash the server if the browser closes.
4. **Autoplay must be stoppable.** The user controls the pace. Never auto-advance without explicit Play.
5. **State transitions should be visually obvious.** The gold flash on node state change is the primary attention mechanism.
6. **The Monday Morning Meltdown is the default demo.** It's the flagship scenario — 0.842 macro F1 on the old 4-class version, the system's best performance under compound stress.
7. **Ground truth should be available but hidden by default.** Add a "Show Truth" toggle that overlays the actual state next to the predicted state so the user can see where the engine is right and wrong.
