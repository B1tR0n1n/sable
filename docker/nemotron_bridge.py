"""
SABLE-Nemotron Bridge
=======================
Feeds SABLE's structured diagnostic output to Nemotron for natural
language incident reports. SABLE is the source of truth for what's
happening. Nemotron is the translator.

The bridge:
  1. Takes SABLE's tick result or recommendation output (structured JSON)
  2. Formats it into a prompt that grounds Nemotron on the facts
  3. Calls llama-server's completion API
  4. Returns the narrative alongside the original structured data

The operator sees both - the data and the explanation. Nemotron never
overrides SABLE's findings.
"""

import logging
from typing import Optional

import requests

log = logging.getLogger(__name__)

DEFAULT_LLAMA_URL = "http://localhost:8081"


class NemotronBridge:
    """Translates SABLE diagnostics into natural language via Nemotron."""

    def __init__(self, llama_url: str = DEFAULT_LLAMA_URL, timeout: float = 30.0):
        self.llama_url = llama_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def is_available(self) -> bool:
        """Check if llama-server is running."""
        try:
            resp = self._session.get(f"{self.llama_url}/health", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def explain_tick(self, tick_result: dict) -> str:
        """Generate a natural language summary of a single tick.

        Args:
            tick_result: The dict returned by SableEngine.infer() + enrichment
        """
        # Build the grounded prompt
        cycle = tick_result.get("cycle", 0)
        n_nodes = len(tick_result.get("nodes", []))
        accuracy = tick_result.get("accuracy")
        avg_conf = tick_result.get("avg_confidence", 0)
        class_counts = tick_result.get("class_counts", {})
        mc_agree = tick_result.get("mc_avg_agreement")

        # Find notable nodes (non-healthy or low confidence)
        notable = []
        for node in tick_result.get("nodes", []):
            if node.get("state") != "healthy" or node.get("confidence", 1) < 0.8:
                label = node.get("label", f"Node {node['id']}")
                comp_type = node.get("component_type", "")
                notable.append(
                    f"- {label} [{comp_type}]: {node['state']} "
                    f"(confidence {node['confidence']:.2f}, "
                    f"trend: {node.get('trend', 'stable')})"
                )

        facts = f"""Infrastructure monitoring cycle {cycle}:
- {n_nodes} nodes monitored
- State distribution: {class_counts}
- Average confidence: {avg_conf:.3f}"""

        if mc_agree is not None:
            facts += f"\n- MC dropout agreement: {mc_agree:.3f}"

        if accuracy is not None:
            facts += f"\n- Classification accuracy: {accuracy:.3f}"

        if notable:
            facts += "\n\nNodes requiring attention:\n" + "\n".join(notable[:10])
        else:
            facts += "\n\nAll nodes healthy."

        prompt = f"""You are a senior infrastructure operations analyst. Based on SABLE's diagnostic output below, write a brief, actionable status update for the operations team. Be direct and specific. Do not speculate beyond what the data shows.

SABLE DIAGNOSTIC DATA:
{facts}

Write a 2-4 sentence operational status update:"""

        return self._complete(prompt)

    def explain_recommendations(self, recs: dict, max_tokens: int = 512) -> str:
        """Generate a natural language incident report from recommendations.

        Args:
            recs: The dict returned by SableEngine.get_recommendations() + enrichment
        """
        summary = recs.get("summary", "")
        actions = recs.get("actions", [])
        root_cause = recs.get("root_cause_label")
        root_type = recs.get("root_cause_type")
        total_affected = recs.get("total_affected", 0)

        facts = f"Incident summary: {summary}\n"
        if root_cause:
            facts += f"Probable root cause: {root_cause} [{root_type}]\n"
        facts += f"Total affected nodes: {total_affected}\n"

        if actions:
            facts += "\nRecommended actions (priority order):\n"
            for a in actions[:8]:
                facts += (f"  P{a['priority']}: {a['action']} -> {a['target']}"
                          f" [{a.get('target_type', '')}]\n")
                facts += f"    Reason: {a['reason']}\n"
                facts += f"    Fix: {a['recommendation']}\n"

        prompt = f"""Write a brief incident report (3-4 paragraphs max) based on these findings. Use the exact component names. No placeholders. No bullet points. Plain prose only.

{facts}

Incident Report:
Automated diagnostics detected"""

        return self._complete(prompt, max_tokens=max_tokens)

    def explain_scenario_complete(self, tick_history: list[dict], recs: dict) -> str:
        """Generate an after-action report from a completed scenario.

        Args:
            tick_history: List of tick results across the scenario
            recs: Final recommendations
        """
        n_ticks = len(tick_history)
        if not tick_history:
            return "No data available for analysis."

        first = tick_history[0]
        last = tick_history[-1]
        n_nodes = len(first.get("nodes", []))

        # Track accuracy over time
        accs = [t.get("accuracy", 0) for t in tick_history if t.get("accuracy") is not None]
        avg_acc = sum(accs) / len(accs) if accs else 0

        # Track state evolution
        first_counts = first.get("class_counts", {})
        last_counts = last.get("class_counts", {})

        # Find first tick where failures appeared
        first_failure_tick = None
        for t in tick_history:
            counts = t.get("class_counts", {})
            if counts.get("failed", 0) > 0 or counts.get("unreachable", 0) > 0:
                first_failure_tick = t.get("cycle")
                break

        summary = recs.get("summary", "")
        root_cause = recs.get("root_cause_label", "Unknown")

        facts = f"""Scenario analysis ({n_ticks} monitoring cycles, {n_nodes} nodes):

Initial state: {first_counts}
Final state: {last_counts}
Average classification accuracy: {avg_acc:.1%}
First failure detected: {"tick " + str(first_failure_tick) if first_failure_tick is not None else "none"}

SABLE assessment: {summary}
Probable root cause: {root_cause}

Recommended actions:"""
        for a in recs.get("actions", [])[:5]:
            facts += f"\n  {a['priority']}. {a['action']} -> {a['target']}"

        prompt = f"""You are a senior infrastructure operations analyst writing an after-action report. Based on SABLE's analysis of this monitoring scenario, write a concise after-action summary. Cover: timeline of events, impact assessment, root cause, and recommended next steps.

Be factual. Only reference what SABLE observed.

SABLE SCENARIO ANALYSIS:
{facts}

AFTER-ACTION REPORT:"""

        return self._complete(prompt)

    def _complete(self, prompt: str, max_tokens: int = 512) -> str:
        """Call llama-server completion API."""
        try:
            resp = self._session.post(
                f"{self.llama_url}/completion",
                json={
                    "prompt": prompt,
                    "n_predict": max_tokens,
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "repeat_penalty": 1.3,
                    "repeat_last_n": 128,
                    "stop": ["\n\n\n", "SABLE DIAGNOSTIC", "SABLE SCENARIO", "---"],
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("content", "").strip()
        except requests.RequestException as e:
            log.warning("Nemotron completion failed: %s", e)
            return f"[Nemotron unavailable: {e}]"
