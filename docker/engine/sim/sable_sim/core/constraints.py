"""Validation constraints ensuring generated topologies are realistic."""

from __future__ import annotations

import logging

from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.component import ComponentType
from sable_sim.core.dependency import DependencyType

logger = logging.getLogger(__name__)


class TopologyConstraints:
    """Validates infrastructure topologies against realism rules."""

    def validate(self, graph: InfrastructureGraph) -> tuple[bool, list[str]]:
        """Validate topology. Returns (is_valid, list_of_violations)."""
        violations: list[str] = []
        violations.extend(self._check_connectivity(graph))
        violations.extend(self._check_hierarchy(graph))
        violations.extend(self._check_dependencies(graph))
        violations.extend(self._check_impossible_connections(graph))
        violations.extend(self._check_redundancy(graph))
        return len(violations) == 0, violations

    def _check_connectivity(self, graph: InfrastructureGraph) -> list[str]:
        """Every component should have at least one dependency edge."""
        violations: list[str] = []
        for comp in graph.get_all_components():
            if not comp.dependencies_in and not comp.dependencies_out:
                # Monitoring servers may be source-only
                if comp.type != ComponentType.MONITORING_SERVER:
                    violations.append(
                        f"{comp.id} ({comp.type}) is isolated — no dependencies"
                    )
        return violations

    def _check_hierarchy(self, graph: InfrastructureGraph) -> list[str]:
        """Access switches should connect to core switches, not to each other."""
        violations: list[str] = []
        access_switches = graph.get_components_by_type(ComponentType.ACCESS_SWITCH)
        for sw in access_switches:
            net_deps = graph.get_dependencies_of_type(
                sw.id, DependencyType.NETWORK_PATH
            )
            has_core_uplink = any(
                c.type == ComponentType.CORE_SWITCH for c, _ in net_deps
            )
            if not has_core_uplink:
                violations.append(
                    f"{sw.id} has no NETWORK_PATH to a CORE_SWITCH"
                )
        return violations

    def _check_dependencies(self, graph: InfrastructureGraph) -> list[str]:
        """All non-infrastructure components should have DNS dependency."""
        violations: list[str] = []
        infra_types = {
            ComponentType.CORE_SWITCH,
            ComponentType.ACCESS_SWITCH,
            ComponentType.WAN_LINK,
            ComponentType.INTERNET_GATEWAY,
        }
        for comp in graph.get_all_components():
            if comp.type in infra_types:
                continue
            dns_deps = graph.get_dependencies_of_type(
                comp.id, DependencyType.DNS_DEPENDENCY
            )
            if not dns_deps:
                violations.append(f"{comp.id} ({comp.type}) has no DNS dependency")
        return violations

    def _check_impossible_connections(
        self, graph: InfrastructureGraph
    ) -> list[str]:
        """Check for obviously impossible topologies."""
        violations: list[str] = []
        for dep in graph.get_all_dependencies():
            src = graph.get_component(dep.source_id)
            tgt = graph.get_component(dep.target_id)
            if src is None or tgt is None:
                violations.append(
                    f"Edge references missing node: {dep.source_id} → {dep.target_id}"
                )
                continue
            # Servers shouldn't directly connect to internet gateway
            if (
                src.type in (ComponentType.SERVER_PHYSICAL, ComponentType.SERVER_VIRTUAL)
                and tgt.type == ComponentType.INTERNET_GATEWAY
                and dep.type == DependencyType.NETWORK_PATH
            ):
                violations.append(
                    f"{src.id} directly connected to {tgt.id} via network "
                    f"— should go through switches/firewall"
                )
        return violations

    def _check_redundancy(self, graph: InfrastructureGraph) -> list[str]:
        """Warn (not error) if critical services lack redundancy."""
        violations: list[str] = []
        critical_types = {
            ComponentType.CORE_SWITCH: 2,
            ComponentType.DNS_SERVER: 2,
            ComponentType.DOMAIN_CONTROLLER: 2,
        }
        total_nodes = graph.node_count
        if total_nodes < 20:
            return violations  # small topologies get a pass

        for ctype, min_count in critical_types.items():
            actual = len(graph.get_components_by_type(ctype))
            if actual < min_count:
                violations.append(
                    f"Only {actual} {ctype} (expected >= {min_count} for redundancy)"
                )
        return violations
