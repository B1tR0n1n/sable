"""The LLM-assisted planner: a model picks and parameterises catalog actions;
its output is validated, never repaired, never retried silently.

Per INTEGRATION-NOTES G3/D9 this does NOT use OVERLORD's `run_agent` (which
hands the model shell/write_file). It makes ONE tool-less completion through
`complete_fn`, whose result has the shape of `OverlordClient.complete()`:

    {text, provider, model, prompt_sha256, output_sha256, stop, usage, refusal}

so the call is audited on the OVERLORD side as model.complete and the Plan
carries the same hashes in PlannerProvenance(kind="llm").

Console-owned fields. The model is asked for steps and a verification only.
If its JSON also carries `id`, `created_at`, `gate`, `planner` or
`blast_radius`, those are STRIPPED before validation: ids and timestamps
are minted by the console, the gate is Phase 6's, provenance is this
module's, and the blast radius is computed from the topology (D4) — a model
must not be able to shrink it. `finding_id` is filled in when absent and
must match when present. Everything else is validated strictly by
validate_plan; any failure raises PlanValidationError with its reasons.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Callable, Optional

from console.catalog import Catalog
from console.contracts import Finding, Plan, PlannerProvenance
from console.topology import Topology

from .validate import PlanValidationError, expected_blast_radius, validate_plan

CompleteFn = Callable[..., dict[str, Any]]      # complete_fn(prompt, system, purpose) -> dict

CONSOLE_OWNED_FIELDS = ("id", "created_at", "gate", "planner", "blast_radius")

SYSTEM_PROMPT = """You are the planning analyst for an infrastructure remediation loop.
You propose; you never act. Another component executes plans, and only after a human gate.

Rules — violations are rejected outright, never repaired:
1. Use ONLY the actions listed in the catalog you are given. Do not invent actions, commands,
   scripts or parameters. If no listed action fits, reply with exactly: {"steps": []}
2. Reply with ONE JSON object and nothing else: no prose, no markdown fences, no comments.
3. Each step must copy `reversibility`, `precondition` (one of the action's listed checks)
   and `compensation` (the action's listed compensation with the params it names, or null)
   EXACTLY from the catalog entry. `timeout_s` must not exceed the action's ceiling.
4. `target_node` must be a node id from the topology, and the action must apply to that
   node's component type. `params` must satisfy the action's params schema exactly.
5. `verification.predicate` must be the predicate the chosen action declares.
6. Do not include `id`, `created_at`, `gate`, `planner` or `blast_radius`: the console owns them.
Prefer the smallest, most reversible plan that addresses the root cause."""

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """The FIRST JSON object in `text` (fenced or bare), or None."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    dec = json.JSONDecoder()
    for cand in candidates:
        start = 0
        while True:
            i = cand.find("{", start)
            if i < 0:
                break
            try:
                obj, _ = dec.raw_decode(cand[i:])
            except json.JSONDecodeError:
                start = i + 1
                continue
            if isinstance(obj, dict):
                return obj
            start = i + 1
    return None


class LLMPlanner:
    def __init__(self, catalog: Catalog, topology: Topology, complete_fn: CompleteFn,
                 model: Optional[str] = None, service_map: Optional[dict[str, str]] = None):
        self.catalog = catalog
        self.topology = topology
        self.complete_fn = complete_fn
        self.model = model
        self.service_map = dict(service_map or {})

    # --- prompt

    def applicable_actions(self, finding: Finding) -> list[dict[str, Any]]:
        types = [finding.root_cause.component_type] + [n.component_type for n in finding.affected_nodes]
        seen: set[str] = set()
        out = []
        for ct in dict.fromkeys(types):
            for a in self.catalog.actions_for(ct):
                if a.action_id in seen:
                    continue
                seen.add(a.action_id)
                out.append({
                    "action_id": a.action_id,
                    "description": a.description,
                    "component_types": a.component_types,
                    "params_schema": a.params,
                    "reversibility": a.reversibility,
                    "compensation": a.compensation.model_dump() if a.compensation else None,
                    "preconditions": a.preconditions,
                    "verification": a.verification,
                    "timeout_s_ceiling": a.executor.grants.timeout_s,
                })
        return out

    def build_prompt(self, finding: Finding) -> str:
        nodes = [{"id": nid, "component_type": self.topology.component_type(nid),
                  **({"service": self.service_map[nid]} if nid in self.service_map else {})}
                 for nid in self.topology.nodes]
        edges = [{"source": e["source"], "target": e["target"], "type": e.get("type"),
                  "criticality": e.get("criticality")} for e in self.topology.edges]
        shape = {
            "finding_id": finding.id,
            "steps": [{"action_id": "<catalog action_id>", "target_node": "<topology node id>",
                       "params": {"<param>": "<value>"}, "reversibility": "<from catalog>",
                       "compensation": {"action_id": "<from catalog>", "params": {}},
                       "precondition": "<one of the action's preconditions>",
                       "timeout_s": "<int, at most the ceiling>"}],
            "verification": {"predicate": "<the action's verification>", "window_s": 30},
        }
        parts = [
            "FINDING (from SABLE, the source of truth):",
            json.dumps(finding.model_dump(mode="json"), indent=1, sort_keys=True),
            "",
            "TOPOLOGY (edge source depends on target; `service` is the compose service name to use in params):",
            json.dumps({"nodes": nodes, "edges": edges}, indent=1),
            "",
            "ACTION CATALOG — the only actions you may use:",
            json.dumps(self.applicable_actions(finding), indent=1),
            "",
            "REQUIRED OUTPUT SHAPE (one JSON object, no other text):",
            json.dumps(shape, indent=1),
        ]
        return "\n".join(parts)

    # --- planning

    def plan(self, finding: Finding) -> Plan:
        prompt = self.build_prompt(finding)
        result = self.complete_fn(prompt=prompt, system=SYSTEM_PROMPT, purpose=f"plan:{finding.id}")
        if not isinstance(result, dict):
            raise PlanValidationError([f"complete_fn returned {type(result).__name__}, not a completion result"])
        if result.get("refusal") or result.get("stop") == "refusal":
            raise PlanValidationError([f"model refused to plan: {result.get('refusal') or result.get('stop')}"])
        text = result.get("text") or ""
        if not text.strip():
            raise PlanValidationError([f"model returned no text (stop={result.get('stop')!r})"])
        recomputed = hashlib.sha256(text.encode()).hexdigest()
        output_sha = result.get("output_sha256") or recomputed
        if output_sha != recomputed:
            raise PlanValidationError(["output_sha256 does not match the text returned; refusing an unverifiable completion"])

        obj = extract_json_object(text)
        if obj is None:
            raise PlanValidationError([f"no JSON object in model output (first 120 chars: {text[:120]!r})"])
        obj = copy.deepcopy(obj)
        for k in CONSOLE_OWNED_FIELDS:
            obj.pop(k, None)                   # console-owned: minted or computed below
        obj.setdefault("finding_id", finding.id)
        steps = obj.get("steps")
        if not isinstance(steps, list) or not steps:
            raise PlanValidationError(["model proposed no steps (no applicable catalog action, or malformed output)"])
        targets = [s.get("target_node") for s in steps if isinstance(s, dict) and isinstance(s.get("target_node"), str)]
        nodes = expected_blast_radius(targets, self.topology)
        obj["blast_radius"] = {"nodes": nodes, "count": len(nodes)}
        obj["planner"] = PlannerProvenance(
            kind="llm", provider=result.get("provider"), model=result.get("model") or self.model,
            prompt_sha256=result.get("prompt_sha256"), output_sha256=output_sha,
        ).model_dump()
        return validate_plan(obj, self.catalog, finding, self.topology)
