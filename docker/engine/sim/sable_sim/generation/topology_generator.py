"""Random infrastructure topology generator with constraint validation."""

from __future__ import annotations

import logging
from typing import Any

from sable_sim.core.graph import InfrastructureGraph
from sable_sim.core.component import (
    Component,
    ComponentType,
    ComponentState,
    DEFAULT_PROPERTIES,
)
from sable_sim.core.dependency import Dependency, DependencyType, Criticality
from sable_sim.generation.constraints import TopologyConstraints
from sable_sim.utils.random import SeededRandom

logger = logging.getLogger(__name__)


class TopologyGenerator:
    """Generates random but realistic infrastructure topologies."""

    def __init__(self, rng: SeededRandom) -> None:
        self._rng = rng
        self._constraints = TopologyConstraints()
        self._id_counters: dict[str, int] = {}

    def _gen_id(self, prefix: str) -> str:
        """Generate unique component ID like 'sw-core-01'."""
        count = self._id_counters.get(prefix, 0) + 1
        self._id_counters[prefix] = count
        return f"{prefix}-{count:02d}"

    def _make_component(
        self, comp_id: str, comp_type: ComponentType, extra: dict | None = None
    ) -> Component:
        props = dict(DEFAULT_PROPERTIES.get(comp_type, {}))
        if extra:
            props.update(extra)
        # Add random variation ±20 %
        for k, v in list(props.items()):
            if isinstance(v, (int, float)) and v > 0:
                props[k] = round(v * self._rng.uniform(0.8, 1.2), 4)
        return Component(id=comp_id, type=comp_type, properties=props)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        node_count_target: int = 50,
        ensure_redundancy: bool = True,
    ) -> InfrastructureGraph:
        """Generate a random realistic topology with ~*node_count_target* nodes."""
        self._id_counters.clear()
        graph = InfrastructureGraph()

        # Scale factors
        n = max(node_count_target, 10)
        n_core = 2 if ensure_redundancy else 1
        n_access = max(2, n // 10)
        n_phys_servers = max(1, n // 8)
        n_hypervisors = max(1, n_phys_servers // 2)
        n_vms = max(1, n // 6)
        n_dns = 2 if ensure_redundancy else 1
        n_dc = 2 if ensure_redundancy else 1

        # 1. Core switches
        cores = [
            self._make_component(self._gen_id("sw-core"), ComponentType.CORE_SWITCH)
            for _ in range(n_core)
        ]
        for c in cores:
            graph.add_component(c)
        # Inter-core links
        for i in range(len(cores)):
            for j in range(i + 1, len(cores)):
                graph.add_dependency(Dependency(
                    cores[i].id, cores[j].id,
                    DependencyType.NETWORK_PATH, Criticality.REDUNDANT,
                ))
                graph.add_dependency(Dependency(
                    cores[j].id, cores[i].id,
                    DependencyType.NETWORK_PATH, Criticality.REDUNDANT,
                ))

        # 2. Perimeter: firewall, router, internet gateway, WAN link
        fw = self._make_component(self._gen_id("fw"), ComponentType.FIREWALL)
        router = self._make_component(self._gen_id("rtr"), ComponentType.ROUTER)
        igw = self._make_component(self._gen_id("igw"), ComponentType.INTERNET_GATEWAY)
        wan = self._make_component(self._gen_id("wan"), ComponentType.WAN_LINK)
        for c in [fw, router, igw, wan]:
            graph.add_component(c)

        # WAN → IGW → Router → Firewall → Core
        graph.add_dependency(Dependency(wan.id, igw.id, DependencyType.NETWORK_PATH, Criticality.HARD))
        graph.add_dependency(Dependency(igw.id, router.id, DependencyType.NETWORK_PATH, Criticality.HARD))
        graph.add_dependency(Dependency(router.id, fw.id, DependencyType.NETWORK_PATH, Criticality.HARD))
        for core in cores:
            graph.add_dependency(Dependency(fw.id, core.id, DependencyType.NETWORK_PATH, Criticality.HARD))

        # 3. Access switches → core
        access_switches = [
            self._make_component(self._gen_id("sw-acc"), ComponentType.ACCESS_SWITCH)
            for _ in range(n_access)
        ]
        for sw in access_switches:
            graph.add_component(sw)
            # Uplink to 1 or 2 core switches
            uplinks = self._rng.sample(cores, min(len(cores), self._rng.randint(1, 2)))
            for core in uplinks:
                graph.add_dependency(Dependency(
                    sw.id, core.id, DependencyType.NETWORK_PATH,
                    Criticality.REDUNDANT if len(uplinks) > 1 else Criticality.HARD,
                ))

        # 4. Physical servers → access switches
        phys_servers: list[Component] = []
        for _ in range(n_phys_servers):
            srv = self._make_component(self._gen_id("srv-phys"), ComponentType.SERVER_PHYSICAL)
            graph.add_component(srv)
            sw = self._rng.choice(access_switches)
            graph.add_dependency(Dependency(srv.id, sw.id, DependencyType.NETWORK_PATH, Criticality.HARD))
            phys_servers.append(srv)

        # 5. Hypervisors on physical servers
        hypervisors: list[Component] = []
        for i in range(n_hypervisors):
            hv = self._make_component(self._gen_id("hv"), ComponentType.HYPERVISOR)
            graph.add_component(hv)
            host = phys_servers[i % len(phys_servers)]
            graph.add_dependency(Dependency(hv.id, host.id, DependencyType.HOSTING_DEPENDENCY, Criticality.HARD))
            hypervisors.append(hv)

        # 6. VMs on hypervisors
        vms: list[Component] = []
        for _ in range(n_vms):
            vm = self._make_component(self._gen_id("srv-vm"), ComponentType.SERVER_VIRTUAL)
            graph.add_component(vm)
            hv = self._rng.choice(hypervisors)
            graph.add_dependency(Dependency(vm.id, hv.id, DependencyType.HOSTING_DEPENDENCY, Criticality.HARD))
            vms.append(vm)

        # 7. Storage
        n_arrays = max(1, n // 25)
        arrays: list[Component] = []
        for _ in range(n_arrays):
            arr = self._make_component(self._gen_id("stor-arr"), ComponentType.STORAGE_ARRAY)
            graph.add_component(arr)
            sw = self._rng.choice(access_switches)
            graph.add_dependency(Dependency(arr.id, sw.id, DependencyType.NETWORK_PATH, Criticality.HARD))
            arrays.append(arr)
            # Storage targets
            for _ in range(self._rng.randint(1, 3)):
                tgt = self._make_component(self._gen_id("stor-tgt"), ComponentType.STORAGE_TARGET)
                graph.add_component(tgt)
                graph.add_dependency(Dependency(tgt.id, arr.id, DependencyType.STORAGE_DEPENDENCY, Criticality.HARD))

        # Hypervisors → storage
        for hv in hypervisors:
            arr = self._rng.choice(arrays)
            graph.add_dependency(Dependency(hv.id, arr.id, DependencyType.STORAGE_DEPENDENCY, Criticality.HARD))

        # 8. DNS servers
        dns_servers: list[Component] = []
        for _ in range(n_dns):
            dns = self._make_component(self._gen_id("dns"), ComponentType.DNS_SERVER)
            graph.add_component(dns)
            sw = self._rng.choice(access_switches)
            graph.add_dependency(Dependency(dns.id, sw.id, DependencyType.NETWORK_PATH, Criticality.HARD))
            dns_servers.append(dns)

        # 9. DHCP
        dhcp = self._make_component(self._gen_id("dhcp"), ComponentType.DHCP_SERVER)
        graph.add_component(dhcp)
        graph.add_dependency(Dependency(
            dhcp.id, self._rng.choice(access_switches).id,
            DependencyType.NETWORK_PATH, Criticality.HARD,
        ))

        # 10. Domain controllers
        dcs: list[Component] = []
        for _ in range(n_dc):
            dc = self._make_component(self._gen_id("dc"), ComponentType.DOMAIN_CONTROLLER)
            graph.add_component(dc)
            sw = self._rng.choice(access_switches)
            graph.add_dependency(Dependency(dc.id, sw.id, DependencyType.NETWORK_PATH, Criticality.HARD))
            dcs.append(dc)
        # DC replication
        for i in range(len(dcs)):
            for j in range(i + 1, len(dcs)):
                graph.add_dependency(Dependency(
                    dcs[i].id, dcs[j].id,
                    DependencyType.REPLICATION_DEPENDENCY, Criticality.SOFT,
                ))

        # 11. Monitoring
        mon = self._make_component(self._gen_id("mon"), ComponentType.MONITORING_SERVER)
        graph.add_component(mon)
        graph.add_dependency(Dependency(
            mon.id, self._rng.choice(access_switches).id,
            DependencyType.NETWORK_PATH, Criticality.HARD,
        ))

        # 12. Application services
        n_apps = max(1, n // 15)
        apps: list[Component] = []
        for _ in range(n_apps):
            app = self._make_component(self._gen_id("app"), ComponentType.APPLICATION_SERVICE)
            graph.add_component(app)
            # Hosted on a VM
            if vms:
                vm = self._rng.choice(vms)
                graph.add_dependency(Dependency(
                    app.id, vm.id, DependencyType.HOSTING_DEPENDENCY, Criticality.HARD,
                ))
            apps.append(app)

        # 13. Load balancer
        lb = self._make_component(self._gen_id("lb"), ComponentType.LOAD_BALANCER)
        graph.add_component(lb)
        graph.add_dependency(Dependency(
            lb.id, self._rng.choice(access_switches).id,
            DependencyType.NETWORK_PATH, Criticality.HARD,
        ))
        for app in apps:
            graph.add_dependency(Dependency(
                app.id, lb.id, DependencyType.SERVICE_DEPENDENCY, Criticality.SOFT,
            ))

        # 14. Certificate authority
        ca = self._make_component(self._gen_id("ca"), ComponentType.CERTIFICATE_AUTHORITY)
        graph.add_component(ca)
        graph.add_dependency(Dependency(
            ca.id, self._rng.choice(access_switches).id,
            DependencyType.NETWORK_PATH, Criticality.HARD,
        ))

        # --- Wire service dependencies ---

        # DNS dependencies for all non-network components
        _infra_types = {
            ComponentType.CORE_SWITCH, ComponentType.ACCESS_SWITCH,
            ComponentType.WAN_LINK, ComponentType.INTERNET_GATEWAY,
        }
        for comp in graph.get_all_components():
            if comp.type in _infra_types:
                continue
            if comp.type == ComponentType.DNS_SERVER:
                continue
            dns = self._rng.choice(dns_servers)
            crit = Criticality.REDUNDANT if len(dns_servers) > 1 else Criticality.SOFT
            graph.add_dependency(Dependency(
                comp.id, dns.id, DependencyType.DNS_DEPENDENCY, crit,
            ))

        # Auth dependencies for domain-joined components
        auth_types = {
            ComponentType.SERVER_PHYSICAL, ComponentType.SERVER_VIRTUAL,
            ComponentType.HYPERVISOR, ComponentType.APPLICATION_SERVICE,
            ComponentType.VDI_BROKER, ComponentType.VDI_HOST,
        }
        for comp in graph.get_all_components():
            if comp.type not in auth_types:
                continue
            dc = self._rng.choice(dcs)
            crit = Criticality.REDUNDANT if len(dcs) > 1 else Criticality.SOFT
            graph.add_dependency(Dependency(
                comp.id, dc.id, DependencyType.AUTHENTICATION_DEPENDENCY, crit,
            ))

        # Monitoring dependencies
        for comp in graph.get_all_components():
            if comp.id == mon.id:
                continue
            if self._rng.random() < 0.9:  # 90% have monitoring
                graph.add_dependency(Dependency(
                    comp.id, mon.id, DependencyType.MONITORING_DEPENDENCY, Criticality.SOFT,
                ))

        # Validate
        valid, violations = self._constraints.validate(graph)
        if not valid:
            logger.warning("Topology has %d constraint violations:", len(violations))
            for v in violations:
                logger.warning("  - %s", v)

        logger.info(
            "Generated topology: %d nodes, %d edges",
            graph.node_count,
            graph.edge_count,
        )
        return graph
