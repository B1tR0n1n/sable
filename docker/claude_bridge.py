"""
SABLE — Claude Bridge
=====================
Drop-in alternative to NemotronBridge that routes SABLE's language layer to Claude.

Design rules:
  * SABLE's engine output stays ground truth. Claude explains, summarizes and
    investigates; it never overrides a diagnosis.
  * Report methods (explain_tick / explain_recommendations / explain_scenario_complete)
    are inherited unchanged: they build SABLE's grounded prompts, then call
    _complete(), which this class reroutes to Claude.
  * chat() is upgraded: instead of stuffing all context into the prompt, Claude can
    call read-only SABLE tools (node detail, topology, recommendations) as needed.
  * No tool can change engine state. Actuation belongs to OVERLORD, not here.
  * API errors and tool failures come back as marked strings / is_error tool
    results. Nothing in here raises into the server.

Env:
  SABLE_LLM           "claude" selects this bridge (see select_bridge); anything
                      else keeps the local Nemotron path unchanged
  ANTHROPIC_API_KEY   required for the Claude path; the ONLY key source
  SABLE_CLAUDE_MODEL  optional, default "claude-sonnet-5"
"""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any, Callable, Mapping, Optional

import anthropic

from nemotron_bridge import NemotronBridge

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOOL_ROUNDS = 6
UNAVAILABLE = "[Claude unavailable: ANTHROPIC_API_KEY not set]"
TRUNCATED = " […truncated]"

CHAT_SYSTEM = """You are a senior infrastructure engineer helping an operator diagnose an incident.
SABLE's automated diagnostics are your ground truth. Use the tools to look up node detail,
topology and current recommendations rather than guessing.

Rules:
- Clearly separate what SABLE's data shows from your own inference. Label inference as such.
- Never contradict SABLE's root-cause diagnosis without citing specific data that conflicts with it.
- You cannot change the system. If a fix is needed, describe it as a recommendation for the operator.
- Be direct, specific and actionable. Use exact component names."""

TOOLS = [
    {
        "name": "get_recommendations",
        "description": "Current SABLE diagnosis: root cause, affected components and prioritized recommended actions.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_node",
        "description": "Health, state, metrics and labels for one topology node by its index.",
        "input_schema": {
            "type": "object",
            "properties": {"idx": {"type": "integer", "description": "Node index in the topology"}},
            "required": ["idx"],
        },
    },
    {
        "name": "get_topology",
        "description": "Topology nodes and dependency edges, for reasoning about upstream causes and downstream impact.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ── read-only tool factory ────────────────────────────────────────────────


def make_tools(engine: Any,
               enrich_recommendations: Callable[[dict], dict],
               node_label: Callable[[int], str],
               node_id: Callable[[int], str],
               node_type: Callable[[int], str],
               topo_nodes: list[dict],
               topo_edges: list[dict]) -> dict[str, Callable[..., dict]]:
    """Build the three read-only accessors Claude may call.

    Every accessor wraps the SYNC internals that the server's own routes use
    (server.py: /api/recommendations, /api/node/{idx}, /api/topology) and
    returns a plain, JSON-serialisable *copy*. None of them writes to the
    engine, and the copies mean Claude's loop can never alias engine state.
    """

    def get_recommendations() -> dict:
        return copy.deepcopy(enrich_recommendations(engine.get_recommendations()))

    def get_node(idx: Any) -> dict:
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return {"error": f"idx must be an integer, got {idx!r}"}
        n_nodes = int(getattr(engine, "n_nodes", 0) or 0)
        if not 0 <= idx < n_nodes:
            return {"error": f"node index {idx} out of range (0..{n_nodes - 1})",
                    "n_nodes": n_nodes}
        report = copy.deepcopy(engine.get_node_report(idx))
        report["label"] = node_label(idx)
        report["topo_id"] = node_id(idx)
        report["component_type"] = node_type(idx)
        return report

    def get_topology() -> dict:
        return {
            "nodes": copy.deepcopy(list(topo_nodes)),
            "edges": copy.deepcopy(list(topo_edges)),
        }

    return {
        "get_recommendations": get_recommendations,
        "get_node": get_node,
        "get_topology": get_topology,
    }


# ── provider selection ────────────────────────────────────────────────────


def select_bridge(env: Mapping[str, str] = os.environ) -> "NemotronBridge | ClaudeBridge":
    """SABLE_LLM == "claude" -> ClaudeBridge(); anything else -> the local NemotronBridge."""
    if env.get("SABLE_LLM", "").strip().lower() == "claude":
        return ClaudeBridge(env=env)
    return NemotronBridge()


def bridge_info(bridge: Any) -> dict:
    """Provider descriptor for /api/nemotron/status, for either bridge class."""
    info = getattr(bridge, "provider_info", None)
    if callable(info):
        return info()
    return {"provider": "nemotron", "url": getattr(bridge, "llama_url", None)}


# ── the bridge ────────────────────────────────────────────────────────────


class ClaudeBridge(NemotronBridge):
    def __init__(self, model: Optional[str] = None, timeout: float = 60.0,
                 env: Mapping[str, str] = os.environ):
        super().__init__(timeout=timeout)
        self.model = model or env.get("SABLE_CLAUDE_MODEL") or DEFAULT_MODEL
        self.llama_url = f"anthropic:{self.model}"  # shown by /api/nemotron/status
        api_key = env.get("ANTHROPIC_API_KEY")
        # The key comes ONLY from ANTHROPIC_API_KEY: pass it explicitly so the
        # SDK never falls through to another credential source.
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout) if api_key else None
        self._tools: dict[str, Callable[[dict], dict]] = {}

    # ── wiring ────────────────────────────────────────────────────────────

    def register_tools(self, get_recommendations: Callable[[], dict],
                       get_node: Callable[[int], dict],
                       get_topology: Callable[[], dict]) -> None:
        """Server passes in read-only accessors (see make_tools) after the engine is initialized."""
        self._tools = {
            "get_recommendations": lambda _in: get_recommendations(),
            "get_node": lambda _in: get_node(_in.get("idx")),
            "get_topology": lambda _in: get_topology(),
        }

    def is_available(self) -> bool:
        return self._client is not None

    def provider_info(self) -> dict:
        return {"provider": "claude", "model": self.model}

    # ── report path: reroute inherited prompts to Claude ──────────────────

    def _complete(self, prompt: str, max_tokens: int = 512) -> str:
        if not self._client:
            return UNAVAILABLE
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return self._finish(resp)
        except anthropic.APIError as e:
            log.warning("Claude completion failed: %s", e)
            return f"[Claude error: {e.__class__.__name__}]"
        except Exception as e:  # never let the report path raise into the server
            log.warning("Claude completion failed: %r", e)
            return f"[Claude error: {e.__class__.__name__}]"

    # ── chat path: tool-using investigation ───────────────────────────────

    def chat(self, user_message: str, sable_context: dict, history: list[dict] = None) -> str:
        if not self._client:
            return UNAVAILABLE

        # Seed with the current recommendations so simple questions need no tool call.
        system = CHAT_SYSTEM + "\n\nCurrent SABLE recommendations (ground truth):\n" + \
            json.dumps(sable_context, default=str)[:12000]

        messages = self._normalise_history(history)
        self._append_turn(messages, "user", user_message)

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system,
                    tools=TOOLS if self._tools else [],
                    messages=messages,
                )
                if resp.stop_reason != "tool_use":
                    return self._finish(resp)

                # Echo the assistant turn back verbatim (SDK blocks serialise correctly),
                # then answer every tool_use block in ONE user message.
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        **self._run_tool(block.name, block.input),
                    })
                messages.append({"role": "user", "content": results})

            return "[Investigation stopped: tool-call limit reached]"
        except anthropic.APIError as e:
            log.warning("Claude chat failed: %s", e)
            return f"[Claude error: {e.__class__.__name__}]"
        except Exception as e:  # never let the chat path raise into the server
            log.warning("Claude chat failed: %r", e)
            return f"[Claude error: {e.__class__.__name__}]"

    def _run_tool(self, name: str, tool_input: Any) -> dict:
        """Run one registered tool. Failures are returned to the model, never raised."""
        fn = self._tools.get(name)
        if fn is None:
            return {"content": f"Unknown tool: {name}", "is_error": True}
        try:
            payload = fn(tool_input if isinstance(tool_input, dict) else {})
            return {"content": json.dumps(payload, default=str)[:20000]}
        except Exception as e:  # tool failures go back to Claude, never crash the server
            return {"content": f"Tool error: {e.__class__.__name__}: {e}", "is_error": True}

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_history(history: Optional[list[dict]]) -> list[dict]:
        """Turn the browser's [{role, content}] list into a valid Messages API history.

        * keeps only user/assistant roles (a stray "system" is dropped);
        * drops empty turns and anything before the first user turn;
        * merges consecutive same-role turns (joined with a blank line) so the
          API never sees two user turns in a row — e.g. after a failed request
          left an unanswered user turn in the client's chatHistory.
        """
        out: list[dict] = []
        for m in history or []:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            content = m.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, default=str) if content is not None else ""
            content = content.strip()
            if not content:
                continue
            if not out and role != "user":
                continue  # history must start with a user turn
            ClaudeBridge._append_turn(out, role, content)
        return out

    @staticmethod
    def _append_turn(messages: list[dict], role: str, content: str) -> None:
        if messages and messages[-1]["role"] == role and isinstance(messages[-1]["content"], str):
            messages[-1] = {"role": role, "content": messages[-1]["content"] + "\n\n" + content}
        else:
            messages.append({"role": role, "content": content})

    @staticmethod
    def _finish(resp: Any) -> str:
        """Collect the text blocks of a final assistant message; flag truncation/refusal."""
        text = "".join(getattr(b, "text", "") for b in resp.content if b.type == "text").strip()
        stop = getattr(resp, "stop_reason", None)
        if stop == "max_tokens":
            text += TRUNCATED
        elif stop == "refusal" and not text:
            text = "[Claude declined to answer this request]"
        return text
