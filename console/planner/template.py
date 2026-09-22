"""The deterministic template planner: a finding pattern maps to one step."""

from __future__ import annotations

from typing import Iterable, Optional

from console.catalog import Catalog, CatalogError
from console.contracts import BlastRadius, Finding, Plan, PlannerProvenance, Step, Verification
from console.topology import Topology

from .templates import TEMPLATES, VERIFICATION_WINDOW_S, Template
from .validate import expected_blast_radius, validate_plan


class NoTemplate(LookupError):
    """No template covers this finding; the loop can only report."""


class _Unresolvable(Exception):
    pass


class TemplatePlanner:
    def __init__(self, catalog: Catalog, topology: Topology, service_map: Optional[dict[str, str]] = None,
                 golden: Optional[dict[str, dict[str, str]]] = None, disabled: Iterable[str] = ()):
        self.catalog = catalog
        self.topology = topology
        self.service_map = dict(service_map or {})
        self.golden = dict(golden or {})          # node_id -> {file, key, value[, service]}
        self.disabled = set(disabled)             # actions a template may not use (it falls back)

    def service_of(self, node_id: str) -> str:
        return self.service_map.get(node_id, node_id)

    # --- param sources

    def _replica_of(self, node_id: str) -> str:
        for e in self.topology.edges:
            if e["source"] == node_id and str(e.get("type", "")).upper() == "REPLICATION_DEPENDENCY":
                return e["target"]
        raise _Unresolvable(f"{node_id} has no REPLICATION_DEPENDENCY edge")

    def _proxy_of(self, node_id: str) -> str:
        for d in self.topology.direct_dependents(node_id):
            if self.topology.component_type(d) == "LOAD_BALANCER":
                return d
        raise _Unresolvable(f"no LOAD_BALANCER depends directly on {node_id}")

    def _resolve(self, source: str, node_id: str) -> str:
        if source == "service":
            return self.service_of(node_id)
        if source == "node":
            return node_id
        if source == "replica_service":
            return self.service_of(self._replica_of(node_id))
        if source == "proxy_service":
            return self.service_of(self._proxy_of(node_id))
        if source in ("golden_file", "golden_key", "golden_value"):
            g = self.golden.get(node_id)
            if not g or source[len("golden_"):] not in g:
                raise _Unresolvable(f"no golden config for {node_id}")
            return g[source[len("golden_"):]]
        raise CatalogError(f"template param source {source!r} is not known")

    def _pick(self, template: Template, node_id: str) -> tuple[Template, dict[str, str]]:
        t: Optional[Template] = template
        tried = []
        while t is not None:
            try:
                if t.action_id in self.disabled:
                    raise _Unresolvable(f"action {t.action_id} is disabled")
                return t, {k: self._resolve(src, node_id) for k, src in t.params.items()}
            except _Unresolvable as e:
                tried.append(f"{t.template_id}: {e}")
                t = t.fallback
        raise NoTemplate("; ".join(tried))

    # --- planning

    def template_for(self, finding: Finding) -> Template:
        rc = finding.root_cause
        if rc.node_id not in self.topology.nodes:
            raise NoTemplate(f"root cause {rc.node_id!r} is not in the topology")
        ct = self.topology.component_type(rc.node_id)
        if ct != rc.component_type:
            raise NoTemplate(f"finding says {rc.node_id} is {rc.component_type}, topology says {ct}")
        t = TEMPLATES.get((ct, rc.state))
        if t is None:
            raise NoTemplate(f"no template for ({ct}, {rc.state})")
        return t

    def plan(self, finding: Finding) -> Plan:
        rc = finding.root_cause
        template, params = self._pick(self.template_for(finding), rc.node_id)
        action = self.catalog.get(template.action_id)
        step = Step(action_id=action.action_id, target_node=rc.node_id, params=params,
                    reversibility=action.reversibility,
                    compensation=self.catalog.compensation_for_action(action.action_id, params),
                    precondition=action.preconditions[0],
                    timeout_s=action.executor.grants.timeout_s)
        nodes = expected_blast_radius([rc.node_id], self.topology)
        plan = Plan(finding_id=finding.id, steps=[step],
                    blast_radius=BlastRadius(nodes=nodes, count=len(nodes)),
                    verification=Verification(predicate=action.verification, window_s=VERIFICATION_WINDOW_S),
                    planner=PlannerProvenance(kind="template", template_id=template.template_id))
        # the template planner is not trusted either: its output goes through the same gate
        return validate_plan(plan.model_dump(mode="json"), self.catalog, finding, self.topology)
