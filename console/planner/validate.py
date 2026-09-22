"""validate_plan — the gate between any planner and the rest of the loop.

A plan dict (from the template planner, from a model, from an operator)
is parsed with the Plan contract and then checked against the catalog and
the topology. Nothing is repaired: every violation is collected and raised
as PlanValidationError(reasons). The input dict is never modified.

Checks, per step
  action_id      exists in the catalog and is not compensation_only
  params         validate against the action's JSON Schema
  target_node    exists in the topology; its component_type is one the action applies to
  precondition   is one of the checks the catalog declares for that action
  reversibility  equals the catalog's for that action
  compensation   equals catalog.compensation_for(step) (or None)
  timeout_s      does not exceed the executor binding's ceiling
Plan-wide
  finding_id     equals the finding's id
  verification   names a known predicate, and one declared by the plan's actions
  blast_radius   equals the union over steps of topology.blast_radius(target_node)
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from console.catalog import Catalog, CatalogError, check_schema
from console.contracts import Finding, Plan, Reversibility
from console.topology import Topology


class PlanValidationError(ValueError):
    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("plan rejected: " + "; ".join(self.reasons))


def expected_blast_radius(target_nodes: list[str], topology: Topology) -> list[str]:
    """Union of each target's blast radius, first-seen order, no duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for t in target_nodes:
        for n in topology.blast_radius(t):
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


def _format_pydantic(e: ValidationError) -> list[str]:
    out = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", ())) or "plan"
        out.append(f"{loc}: {err.get('msg')}")
    return out


def validate_plan(plan_dict: dict[str, Any], catalog: Catalog, finding: Finding, topology: Topology) -> Plan:
    if not isinstance(plan_dict, dict):
        raise PlanValidationError([f"plan must be a JSON object, got {type(plan_dict).__name__}"])
    try:
        plan = Plan.model_validate(plan_dict)
    except ValidationError as e:
        raise PlanValidationError(_format_pydantic(e)) from None

    reasons: list[str] = []
    if plan.finding_id != finding.id:
        reasons.append(f"finding_id {plan.finding_id!r} is not the finding being planned for ({finding.id!r})")

    predicates_declared: set[str] = set()
    for i, step in enumerate(plan.steps):
        tag = f"steps[{i}] ({step.step_id})"
        if not catalog.has_action(step.action_id):
            reasons.append(f"{tag}: unknown action {step.action_id!r}; the catalog knows "
                           f"{sorted(a for a, s in catalog.actions.items() if not s.compensation_only)}")
            continue
        action = catalog.get(step.action_id)
        if action.compensation_only:
            reasons.append(f"{tag}: {step.action_id!r} is compensation-only and may not be proposed")
        for p in check_schema(step.params, action.params):
            reasons.append(f"{tag}: {p}")
        if step.target_node not in topology.nodes:
            reasons.append(f"{tag}: target_node {step.target_node!r} is not in the topology")
        else:
            ct = topology.component_type(step.target_node)
            if ct not in action.component_types:
                reasons.append(f"{tag}: {step.action_id} does not apply to {step.target_node!r} "
                               f"({ct}); it applies to {action.component_types}")
        if step.precondition not in catalog.checks:
            reasons.append(f"{tag}: unknown precondition check {step.precondition!r}")
        elif step.precondition not in action.preconditions:
            reasons.append(f"{tag}: precondition {step.precondition!r} is not one the catalog declares "
                           f"for {step.action_id} ({action.preconditions})")
        if Reversibility(step.reversibility) != Reversibility(action.reversibility):
            reasons.append(f"{tag}: reversibility {step.reversibility!r} does not match the catalog's "
                           f"{action.reversibility!r} for {step.action_id}")
        try:
            expected = catalog.compensation_for(step)
        except CatalogError as e:
            expected = None
            reasons.append(f"{tag}: {e}")
        if step.compensation != expected:
            reasons.append(f"{tag}: compensation must be exactly the catalog's: "
                           f"{expected.model_dump() if expected else None}, got "
                           f"{step.compensation.model_dump() if step.compensation else None}")
        ceiling = action.executor.grants.timeout_s
        if step.timeout_s > ceiling:
            reasons.append(f"{tag}: timeout_s {step.timeout_s} exceeds the catalog ceiling {ceiling} for {step.action_id}")
        predicates_declared.add(action.verification)

    pred = plan.verification.predicate
    if pred not in catalog.predicates:
        reasons.append(f"verification: unknown predicate {pred!r} (known: {sorted(catalog.predicates)})")
    elif predicates_declared and pred not in predicates_declared:
        reasons.append(f"verification: predicate {pred!r} is not one the plan's actions declare "
                       f"({sorted(predicates_declared)})")

    expected_br = expected_blast_radius([s.target_node for s in plan.steps], topology)
    if set(plan.blast_radius.nodes) != set(expected_br) or len(plan.blast_radius.nodes) != len(set(plan.blast_radius.nodes)):
        reasons.append(f"blast_radius {plan.blast_radius.nodes} does not equal the topology's {expected_br}")

    if reasons:
        raise PlanValidationError(reasons)
    return plan
