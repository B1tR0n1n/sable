# SABLE — Run of Show (BrightWorks pitch)

**Goal:** get BrightWorks to see SABLE as a capability that cuts MTTR and senior-engineer hours across their client base.

**Format:** double-click `index.html`, drive it live in the browser. ~6–8 minutes.

---

## The hook (say this first)

> "Every MSP has the same expensive problem: an incident fires, alarms light up
> from every layer, and it takes a senior engineer hours to find the one root
> cause under the noise. SABLE does that in milliseconds. Let me show you a real one."

## The flow

1. **Open on `monday_morning`.** Set the scene:
   > "Storage array started degrading Friday evening. Nobody noticed. Over the
   > weekend it cascaded — VMs, DNS, domain-controller replication. Monday morning
   > the login storm spikes the core switch. The operator sees apps down, monitoring
   > gaps, alarms everywhere. No clear cause. This is a 2–4 hour senior-engineer ticket."

2. **Hit ▶ play.** Narrate as it runs:
   - Watch the node grid light up as the cascade spreads (healthy → degraded → failed).
   - Point at the **hero bar**: live accuracy, affected-node count, confidence, and
     **inference time (~7–9 ms per cycle)**. "That's the diagnosis speed. Milliseconds."
   - "Three models vote on every node — structural (GNN), decision (POMDP), temporal
     (Mamba) — fused into one call. The router shows which expert is driving each verdict."

3. **Let it finish → recommendations panel.** This is the money shot:
   > "It doesn't just say what's broken. It ranks what to check first by information
   > gain — so a junior tech works the incident like a senior would."

4. **Click a node** (e.g. the core switch) → show its trajectory. "Full prediction
   history vs. ground truth, per node. Auditable."

5. **Switch to `smd_incident_1`.** Land the credibility punch:
   > "That first one was our enterprise scenario. This is *real server telemetry* —
   > 28 machines, actual incident. 96% accuracy. Not a synthetic toy."

## The close

> "For an MSP this is leverage: faster MTTR, fewer escalations, junior techs closing
> tickets that used to eat senior hours, and one pane across every client's noise.
> The portable version you're looking at replays real runs. The workstation build
> recomputes live on new incidents. I want to talk about running a pilot on your stack."

---

## Honest answers to the questions they'll ask

- **"Is this live or canned?"** → "This portable build replays *real* engine output so
  it runs on any machine. Live recompute on brand-new incidents is the workstation build —
  same engine, ~7 ms/inference on a GPU."
- **"What's it trained on?"** → GNN/POMDP/Mamba over infrastructure dependency graphs;
  validated against real server telemetry (the `smd_*` scenarios).
- **"Accuracy?"** → Shown live per scenario (82–96% here). It's on screen, not a slide.

## Fallback (if their machine is locked down)

It's a single self-contained HTML file — no install, no admin, no network. If a browser
opens, the demo runs. If USB is blocked, email yourself `index.html` and open it locally.

## Don't oversell

Adapter/MC-dropout toggles and the Nemotron narrative report are **disabled** in the
portable build (they need the live engine). Leave them alone or say so plainly.
