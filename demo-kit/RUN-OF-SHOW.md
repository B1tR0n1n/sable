# SABLE Demo — Run of Show (honest version)

**Goal:** show that SABLE sees into monitoring blind spots and tracks an incident as it evolves. Lead with the story, back it with honest numbers. ~5 minutes.

**Format:** double-click `index.html`, pick **monday_morning**, hit ▶ play, talk over the one screen.

---

## The hook
> "Every infrastructure team monitors maybe half their estate. When something
> breaks, the alerts you get are symptoms — and the root cause is often in the
> half you're *not* watching. SABLE fills that in from the dependency graph, and
> tracks the cascade as it spreads."

## Walk the screen (play the tape)
1. **Ticks 0–1 — quiet.** "Everything reads healthy. This is the baseline."
2. **Ticks 2–5 — the cascade starts.** "Watch it: one node goes, then three,
   then six. Accuracy dips from 100% to the mid-70s as the picture gets messy —
   that's honest, not a flat fake number. SABLE is propagating the failure
   through the dependency graph in real time, ~7 ms per tick."
3. **Ticks 10–20 — it names a root cause.** "Now it's calling the root cause —
   gateway, DNS, WAN router — and the affected set grows to a dozen-plus nodes.
   This is the cascade mapped before anyone finishes the first ticket."
4. **Expert routing panel** — "It's not one model. Three pillars — a graph net
   for topology, a belief tracker for partial monitoring, a temporal model for
   how it evolves — and a router that picks per node. The routed answer beats
   any single pillar."

## The close
> "Feed it your topology and your existing alerts — it tells you what's breaking
> that your monitoring can't see, and follows it over time. Two modes: one-shot
> triage, and this live-stream mode."

---

## Honest numbers (cite only these)
- **Live accuracy 62–100%** tracking an incident as it evolves — never a flat
  fake 96%.
- **GNN 0.73 / routed fusion 0.65 / temporal 0.63 macro-F1**, all measured with
  the label leak closed.
- **Synthetic scenarios ~76–96%; real server telemetry (SMD) ~57–68%** — the
  real-world number is lower, and that's the one to quote for a pilot.
- **LoRA 0.326**, ~7 ms/diagnosis on an RTX 5090.

## The credibility move (this is your differentiator)
> "One thing I'll be straight about: an earlier version reported 96% accuracy —
> and it was lying to itself. The training data leaked the answer into the
> model's inputs, and a second bug let the graph module read a failure label
> off a single input feature instead of reasoning about it. I found both, tore
> them out, retrained the whole stack, and re-measured honestly. Every number on
> this screen is de-leaked and real. I'd rather show you a true 65% than a fake
> 96%."

That paragraph is worth more than the demo. It shows judgment most people never
demonstrate: finding that the good number was a lie and rebuilding it honest.

## Also in the kit
- `topology-view.html` — a single **snapshot** view (observed alerts vs
  SABLE-inferred hidden problems). It's the static, one-shot cousin of the
  dashboard and catches fewer hidden failures than the temporal run, because it
  has no time dimension — lead with `index.html`, use this only as a diagram.

## Don't
- Don't cite any accuracy figure not in this file or on the screen.
- Don't claim it's production-ready — it's a working prototype with honest,
  measured capability. Sell the capability and the rigor, not a finished SKU.
