"""Infrastructure graph built on top of networkx."""

from __future__ import annotations

import copy
from typing import Any

import networkx as nx

from sable_sim.core.component import Component, ComponentState, ComponentType
from sable_sim.core.dependency import Dependency, DependencyType


class InfrastructureGraph:
    """Directed graph of infrastructure components and their dependencies."""

    def __init__(self) -> None:
        self._g = nx.DiGraph()

    # ---- mutators --------------------------------------------------------

    def add_component(self, component: Component) -> None:
        """Add a component as a node."""
        self._g.add_node(component.id, component=component)

    def add_dependency(self, dependency: Dependency) -> None:
        """Add a dependency as a directed edge (source → target)."""
        self._g.add_edge(
            dependency.source_id,
            dependency.target_id,
            dependency=dependency,
        )
        # Update in/out lists on the component objects
        src = self.get_component(dependency.source_id)
        tgt = self.get_component(dependency.target_id)
        if src and dependency.target_id not in src.dependencies_out:
            src.dependencies_out.append(dependency.target_id)
        if tgt and dependency.source_id not in tgt.dependencies_in:
            tgt.dependencies_in.append(dependency.source_id)

    # ---- single-item lookups --------------------------------------------

    def get_component(self, component_id: str) -> Component | None:
        """Look up a component by ID."""
        data = self._g.nodes.get(component_id)
        if data is None:
            return None
        return data.get("component")

    def get_dependency(self, source_id: str, target_id: str) -> Dependency | None:
        """Look up a dependency edge."""
        data = self._g.edges.get((source_id, target_id))
        if data is None:
            return None
        return data.get("dependency")

    # ---- neighbour queries -----------------------------------------------

    def get_dependents(self, component_id: str) -> list[Component]:
        """Components that *depend on* this one.

        Edges are source→target meaning "source depends on target", so
        dependents of X are its **predecessors** in the digraph.
        """
        out: list[Component] = []
        for pred in self._g.predecessors(component_id):
            c = self.get_component(pred)
            if c is not None:
                out.append(c)
        return out

    def get_dependencies(self, component_id: str) -> list[Component]:
        """Components this one *depends on*.

        Edges are source→target meaning "source depends on target", so
        the dependencies of X are its **successors** in the digraph.
        """
        out: list[Component] = []
        for succ in self._g.successors(component_id):
            c = self.get_component(succ)
            if c is not None:
                out.append(c)
        return out

    def get_dependencies_of_type(
        self, component_id: str, dep_type: DependencyType
    ) -> list[tuple[Component, Dependency]]:
        """Return (target_component, dependency) pairs for outgoing edges of *dep_type*.

        Since edges go source→target ("source depends on target"), this
        looks at successors of *component_id*.
        """
        results: list[tuple[Component, Dependency]] = []
        for succ in self._g.successors(component_id):
            dep = self.get_dependency(component_id, succ)
            if dep is not None and dep.type == dep_type:
                c = self.get_component(succ)
                if c is not None:
                    results.append((c, dep))
        return results

    # ---- bulk queries ----------------------------------------------------

    def get_components_by_type(self, comp_type: ComponentType) -> list[Component]:
        """All components of a given type."""
        return [
            d["component"]
            for _, d in self._g.nodes(data=True)
            if d.get("component") is not None and d["component"].type == comp_type
        ]

    def get_components_by_state(self, state: ComponentState) -> list[Component]:
        """All components in a given state."""
        return [
            d["component"]
            for _, d in self._g.nodes(data=True)
            if d.get("component") is not None and d["component"].state == state
        ]

    def get_all_components(self) -> list[Component]:
        """Every component in the graph."""
        return [
            d["component"]
            for _, d in self._g.nodes(data=True)
            if d.get("component") is not None
        ]

    def get_all_dependencies(self) -> list[Dependency]:
        """Every dependency edge."""
        return [
            d["dependency"]
            for _, _, d in self._g.edges(data=True)
            if d.get("dependency") is not None
        ]

    # ---- path / connectivity ---------------------------------------------

    def has_alternate_path(
        self,
        source_id: str,
        target_id: str,
        excluded: set[str] | None = None,
    ) -> bool:
        """True if a path exists from *source_id* to *target_id* after
        removing the nodes in *excluded*."""
        view = self._g.copy()
        if excluded:
            view.remove_nodes_from(excluded)
        if source_id not in view or target_id not in view:
            return False
        return nx.has_path(view, source_id, target_id)

    # ---- properties -------------------------------------------------------

    @property
    def node_count(self) -> int:
        return self._g.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._g.number_of_edges()

    # ---- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Full serialization with *nodes* and *edges* lists."""
        return {
            "nodes": [self.get_component(n).to_dict() for n in self._g.nodes if self.get_component(n)],
            "edges": [d["dependency"].to_dict() for _, _, d in self._g.edges(data=True) if "dependency" in d],
        }

    def copy(self) -> InfrastructureGraph:
        """Deep copy."""
        new = InfrastructureGraph()
        for comp in self.get_all_components():
            new.add_component(comp.copy())
        for dep in self.get_all_dependencies():
            new.add_dependency(
                Dependency(
                    source_id=dep.source_id,
                    target_id=dep.target_id,
                    type=dep.type,
                    criticality=dep.criticality,
                    bandwidth_sensitivity=dep.bandwidth_sensitivity,
                    latency_sensitivity=dep.latency_sensitivity,
                )
            )
        return new
