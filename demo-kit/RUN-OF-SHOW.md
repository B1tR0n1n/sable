# SABLE Demo — Run of Show (honest version)

**Goal:** show that SABLE sees into monitoring blind spots. Lead with the story, back it with honest numbers. ~5 minutes.

**Format:** double-click `index.html`, talk over the one screen.

---

## The hook
> "Every infrastructure team monitors maybe half their estate. When something
> breaks, the alerts you get are symptoms — and the root cause is often in the
> half you're *not* watching. SABLE fills that in from the dependency graph."

## Walk the screen
1. **The three solid nodes** — "This is all the monitoring saw Monday morning:
   a storage IOPS warning, and two apps down. Three alerts, no clear link."
2. **The glowing dashed nodes** — "SABLE took those three, propagated them
   through the dependency graph, and surfaced six more failures — every one in
   a node the monitoring wasn't watching: four VMs on that storage, an access
   switch, DNS. That's the cascade, mapped, before anyone opened a ticket."
3. **The metrics panel** — "~85% accuracy tracking an incident live; catches
   ~40% of hidden failures that monitoring alone catches zero of."

## The close
> "Feed it your topology and your existing alerts — it tells you what's broken
> that your monitoring can't see. Two modes: one-shot triage like this, and a
> live-stream mode that tracks an incident as it evolves."

---

## The credibility move (this is your differentiator)
> "One thing I'll be straight about: an earlier version of this reported 96%
> accuracy — and it was lying to itself. The training data leaked the answer
> into the model's inputs. I found it, tore it out, fixed a structural bug that
> had a whole reasoning module fed zeros, and re-measured honestly. Every number
> on this screen is de-leaked and real. I'd rather show you a true 40% than a
> fake 96%."

That paragraph is worth more than the demo. It shows judgment most people
never demonstrate: finding the good number was a lie and rebuilding it honest.

## Don't
- Don't cite any accuracy figure not on this screen.
- Don't claim it's production-ready — it's a working prototype with honest,
  measured capability. Sell the capability and the rigor, not a finished SKU.
