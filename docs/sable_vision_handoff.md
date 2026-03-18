# SABLE — PROJECT VISION HANDOFF

**Context Window Handoff // Updated March 13, 2026**

---

## THE PROBLEM WITH CURRENT AI

The entire AI industry is fixated on language. LLMs. Text in, text out. The paradigm assumes intelligence is linguistic — that predicting the next token well enough captures something meaningful about cognition.

But that's not how intelligence actually works. Language is a compression layer — how we communicate thought, not thought itself. A surgeon doesn't think in words when operating. A pilot doesn't narrate decisions when landing in crosswind. A Marine doesn't compose sentences when rounds are coming in. An IT engineer doesn't process a broken network as text — he builds a mental model of components, relationships, dependencies, failure points, and information flows. He simulates causality.

Language models can approximate structural reasoning, but only by routing it through text. They have to narrate the structure to reason about it. An operator sees it directly. That gap is the opportunity.

---

## SABLE'S IDENTITY

Sable is not a better chatbot. Sable is not a fine-tuned language model with a personality. Sable is a **cognitive engine for infrastructure operations** — a model that reasons about systems the way an operator reasons. Language is just the interface. The intelligence underneath is structural, causal, and spatial.

**Three core pillars:**

---

## PILLAR 1: CAUSAL REASONING OVER SYSTEM GRAPHS

The model understands how components relate and how failures cascade. Feed it a network topology, introduce a failure, and it predicts the cascade — not because it read about networking, but because it understands how systems propagate state changes. This is not text prediction. This is state propagation modeling.

When Keith walks into a broken environment, he doesn't troubleshoot linearly. He builds a mental graph of the system, identifies where state has changed, and traces the cascade. The model does the same.

---

## PILLAR 2: DECISION MODELING UNDER UNCERTAINTY

Clausewitz made computational. The model evaluates decision trees with incomplete information, friction, and fog. Not a chatbot — a command system. Feed it the situation, the constraints, the unknowns, and it maps the decision space with tradeoffs and confidence levels.

This is the Fields of Fire connection — Keith studies the game as a decision-system laboratory. The same principles apply: decisions under pressure with incomplete information and cascading consequences.

Most AI systems optimize for the "correct" answer. In real operations, there often is no correct answer — there are tradeoffs. The model presents the decision space, not a single recommendation.

---

## PILLAR 3: MULTIMODAL SYSTEM STATE REPRESENTATION

Instead of text, the model ingests structured data — network telemetry, log files, topology maps, performance metrics, event streams. It builds an internal representation of system health as a state space, not a text summary. It then reasons about interventions within that state space — "if I change X, what happens to Y and Z?"

This is the hardest pillar and the one that pushes Sable beyond what current language models do. It requires moving from token prediction to state space modeling. The language interface sits on top, but the reasoning engine underneath operates on structured representations, not text.

---

## WHY KEITH IS THE ONE TO BUILD THIS

Most people building AI models are ML researchers in labs. They understand the math. They understand the architectures. But they've never operated under the conditions they're trying to model.

Keith has. He maintained combat communications in Afghanistan under fire. He manages 2,000+ endpoints in a healthcare enterprise where failures have real consequences. He's about to walk into a failing IaaS environment at Schneider Geomatics and diagnose it from the ground. He's the operator AND the builder.

That dual perspective — understanding both how systems fail in practice and how models learn in theory — is vanishingly rare. It's why Sable can't be built by a typical ML engineer. It has to be built by someone who lives in the problem space.

---

## THE LABORATORY

Keith is in the final stages of being hired by Brightworks Group as an on-site technical resource at Schneider Geomatics. The role involves assessing a failing IaaS environment (Avatara's VDI platform), managing the vendor relationship, and building a path forward.

**This job IS the laboratory for Sable.** Every day on the ground is lived experience of the exact use case Sable is designed to serve:

- Diagnosing system failures from incomplete information
- Tracing causal chains across infrastructure components
- Making decisions under uncertainty with political and technical constraints
- Ingesting telemetry, logs, and state data to build a picture of system health
- Presenting options and tradeoffs to stakeholders who aren't technical

The job and the model feed each other. The experience informs the architecture. The architecture gives the experience structure. This convergence is not accidental — it's the natural result of someone whose operational instincts and technical ambitions point in the same direction.

---

## NEXT STEPS

- **Immediate (current model):** Build evaluation test set. Measure accuracy per operation class. Determine if the merged model is usable as a foundation or if the approach needs to reset.
- **Short-term (architecture):** Begin designing the three-pillar architecture. Research existing work in causal inference over graphs, decision modeling under uncertainty (POMDPs, Monte Carlo tree search, Bayesian decision networks), and multimodal state representation. Identify what can be adapted vs. what needs to be built from scratch.
- **Medium-term (data strategy):** The current Op0/Op1/Op2 text dataset may not be the right foundation for this vision. A cognitive engine for infrastructure operations needs training data that looks like system graphs, telemetry streams, topology maps, and decision traces — not chat transcripts. The dataset strategy needs to be rebuilt around the new architecture.
- **Long-term (the build):** Sable becomes a system that ingests infrastructure state, models causal relationships, evaluates decision options under uncertainty, and communicates findings through a natural language interface. Language is the last mile, not the core.
- **Ongoing (the laboratory):** Every day at Schneider/Brightworks generates operational knowledge. Document patterns, failure modes, decision processes, and diagnostic workflows. This lived experience becomes the design specification for Sable's architecture.

---

## KEITH'S PHILOSOPHY — IN HIS OWN WORDS

> *"The problem I am seeing is that all AI is mostly language models. That's what AI is about. Language, text. Why is that the focus and how do we push through that to something else?"*

> *"It's about the solution that works best in the moment. Best for the current situation. But moments are fleeting and situations change, so IT must adapt accordingly."*

---

*The model nobody else has built. Built by someone who lives in the problem space. That's Sable. That's Keith.*
