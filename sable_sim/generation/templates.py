"""Pre-built topology templates for common environments."""

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
from sable_sim.utils.random import SeededRandom

logger = logging.getLogger(__name__)


class TemplateGenerator:
    """Generates topologies from pre-defined templates."""

    def __init__(self, rng: SeededRandom) -> None:
        self._rng = rng
        self._id_counters: dict[str, int] = {}

    def _gen_id(self, prefix: str) -> str:
        count = self._id_counters.get(prefix, 0) + 1
        self._id_counters[prefix] = count
        return f"{prefix}-{count:02d}"

    def _mc(
        self, comp_id: str, comp_type: ComponentType, extra: dict | None = None
    ) -> Component:
        """Make component with default properties + optional overrides + noise."""
        props = dict(DEFAULT_PROPERTIES.get(comp_type, {}))
        if extra:
            props.update(extra)
        for k, v in list(props.items()):
            if isinstance(v, (int, float)) and v > 0:
                props[k] = round(v * self._rng.uniform(0.85, 1.15), 4)
        return Component(id=comp_id, type=comp_type, properties=props)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, template_name: str, **kwargs: Any) -> InfrastructureGraph:
        """Generate topology from a named template."""
        generators = {
            "small_office": self.small_office,
            "enterprise_campus": self.enterprise_campus,
            "vdi_environment": self.vdi_environment,
            "healthcare_network": self.healthcare_network,
        }
        if template_name not in generators:
            raise ValueError(
                f"Unknown template: {template_name}. "
                f"Available: {list(generators.keys())}"
            )
        self._id_counters.clear()
        return generators[template_name](**kwargs)

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def small_office(self, endpoints: int = 30) -> InfrastructureGraph:
        """Small office: firewall, 1 core switch, 2 access switches, 1 server."""
        g = InfrastructureGraph()

        # Perimeter
        fw = self._mc(self._gen_id("fw"), ComponentType.FIREWALL)
        igw = self._mc(self._gen_id("igw"), ComponentType.INTERNET_GATEWAY)
        wan = self._mc(self._gen_id("wan"), ComponentType.WAN_LINK)
        for c in [fw, igw, wan]:
            g.add_component(c)
        g.add_dependency(Dependency(wan.id, igw.id, DependencyType.NETWORK_PATH, Criticality.HARD))
        g.add_dependency(Dependency(igw.id, fw.id, DependencyType.NETWORK_PATH, Criticality.HARD))

        # Core switch
        core = self._mc(self._gen_id("sw-core"), ComponentType.CORE_SWITCH)
        g.add_component(core)
        g.add_dependency(Dependency(fw.id, core.id, DependencyType.NETWORK_PATH, Criticality.HARD))

        # Access switches
        access = [
            self._mc(self._gen_id("sw-acc"), ComponentType.ACCESS_SWITCH)
            for _ in range(2)
        ]
        for sw in access:
            g.add_component(sw)
            g.add_dependency(Dependency(sw.id, core.id, DependencyType.NETWORK_PATH, Criticality.HARD))

        # Server
        srv = self._mc(self._gen_id("srv-phys"), ComponentType.SERVER_PHYSICAL)
        g.add_component(srv)
        g.add_dependency(Dependency(srv.id, access[0].id, DependencyType.NETWORK_PATH, Criticality.HARD))

        # DNS / DHCP / DC (all on the server)
        dns = self._mc(self._gen_id("dns"), ComponentType.DNS_SERVER)
        dhcp = self._mc(self._gen_id("dhcp"), ComponentType.DHCP_SERVER)
        dc = self._mc(self._gen_id("dc"), ComponentType.DOMAIN_CONTROLLER)
        for svc in [dns, dhcp, dc]:
            g.add_component(svc)
            g.add_dependency(Dependency(svc.id, srv.id, DependencyType.HOSTING_DEPENDENCY, Criticality.HARD))
            g.add_dependency(Dependency(svc.id, access[0].id, DependencyType.NETWORK_PATH, Criticality.HARD))

        # Monitoring
        mon = self._mc(self._gen_id("mon"), ComponentType.MONITORING_SERVER)
        g.add_component(mon)
        g.add_dependency(Dependency(mon.id, srv.id, DependencyType.HOSTING_DEPENDENCY, Criticality.HARD))
        g.add_dependency(Dependency(mon.id, access[0].id, DependencyType.NETWORK_PATH, Criticality.HARD))

        # Wire service deps
        self._wire_service_deps(g, [dns], [dc], mon)

        logger.info("small_office: %d nodes, %d edges", g.node_count, g.edge_count)
        return g

    def enterprise_campus(self, endpoints: int = 1000) -> InfrastructureGraph:
        """Enterprise campus with redundant core, HA firewall, server farm."""
        g = InfrastructureGraph()

        # Perimeter
        fw1 = self._mc(self._gen_id("fw"), ComponentType.FIREWALL)
        fw2 = self._mc(self._gen_id("fw"), ComponentType.FIREWALL)
        rtr1 = self._mc(self._gen_id("rtr"), ComponentType.ROUTER)
        rtr2 = self._mc(self._gen_id("rtr"), ComponentType.ROUTER)
        igw = self._mc(self._gen_id("igw"), ComponentType.INTERNET_GATEWAY)
        wan1 = self._mc(self._gen_id("wan"), ComponentType.WAN_LINK)
        wan2 = self._mc(self._gen_id("wan"), ComponentType.WAN_LINK)
        for c in [fw1, fw2, rtr1, rtr2, igw, wan1, wan2]:
            g.add_component(c)
        g.add_dependency(Dependency(wan1.id, igw.id, DependencyType.NETWORK_PATH, Criticality.REDUNDANT))
        g.add_dependency(Dependency(wan2.id, igw.id, DependencyType.NETWORK_PATH, Criticality.REDUNDANT))
        g.add_dependency(Dependency(igw.id, rtr1.id, DependencyType.NETWORK_PATH, Criticality.REDUNDANT))
        g.add_dependency(Dependency(igw.id, rtr2.id, DependencyType.NETWORK_PATH, Criticality.REDUNDANT))
        g.add_dependency(Dependency(rtr1.id, fw1.id, DependencyType.NETWORK_PATH, Criticality.HARD))
        g.add_dependency(Dependency(rtr2.id, fw2.id, DependencyType.NETWORK_PATH, Criticality.HARD))

        # 2 Core switches (redundant)
        core1 = self._mc(self._gen_id("sw-core"), ComponentType.CORE_SWITCH)
        core2 = self._mc(self._gen_id("sw-core"), ComponentType.CORE_SWITCH)
        for c in [core1, core2]:
            g.add_component(c)
        g.add_dependency(Dependency(core1.id, core2.id, DependencyType.NETWORK_PATH, Criticality.REDUNDANT))
        g.add_dependency(Dependency(core2.id, core1.id, DependencyType.NETWORK_PATH, Criticality.REDUNDANT))
        g.add_dependency(Dependency(fw1.id, core1.id, DependencyType.NETWORK_PATH, Criticality.HARD))
        g.add_dependency(Dependency(fw2.id, core2.id, DependencyType.NETWORK_PATH, Criticality.HARD))
        cores = [core1, core2]

        # Access switches
        n_acc = max(4, endpoints // 50)
        access = []
        for _ in range(n_acc):
            sw = self._mc(self._gen_id("sw-acc"), ComponentType.ACCESS_SWITCH)
            g.add_component(sw)
            g.add_dependency(Dependency(sw.id, core1.id, DependencyType.NETWORK_PATH, Criticality.REDUNDANT))
            g.add_dependency(Dependency(sw.id, core2.id, DependencyType.NETWORK_PATH, Criticality.REDUNDANT))
            access.append(sw)

        # Physical servers
        n_phys = max(4, endpoints // 100)
        phys_servers = []
        for _ in range(n_phys):
            srv = self._mc(self._gen_id("srv-phys"), ComponentType.SERVER_PHYSICAL)
            g.add_component(srv)
            g.add_dependency(Dependency(srv.id, self._rng.choice(access).id, DependencyType.NETWORK_PATH, Criticality.HARD))
            phys_servers.append(srv)

        # Hypervisors
        n_hv = max(2, n_phys // 2)
        hypervisors = []
        for i in range(n_hv):
            hv = self._mc(self._gen_id("hv"), ComponentType.HYPERVISOR)
            g.add_component(hv)
            host = phys_servers[i % len(phys_servers)]
            g.add_dependency(Dependency(hv.id, host.id, DependencyType.HOSTING_DEPENDENCY, Criticality.HARD))
            hypervisors.append(hv)

        # VMs
        n_vms = max(4, endpoints // 50)
        vms = []
        for _ in range(n_vms):
            vm = self._mc(self._gen_id("srv-vm"), ComponentType.SERVER_VIRTUAL)
            g.add_component(vm)
            hv = self._rng.choice(hypervisors)
            g.add_dependency(Dependency(vm.id, hv.id, DependencyType.HOSTING_DEPENDENCY, Criticality.HARD))
            vms.append(vm)

        # Storage
        stor1 = self._mc(self._gen_id("stor-arr"), ComponentType.STORAGE_ARRAY)
        stor2 = self._mc(self._gen_id("stor-arr"), ComponentType.STORAGE_ARRAY)
        for arr in [stor1, stor2]:
            g.add_component(arr)
            g.add_dependency(Dependency(arr.id, self._rng.choice(access).id, DependencyType.NETWORK_PATH, Criticality.HARD))
            for _ in range(2):
                tgt = self._mc(self._gen_id("stor-tgt"), ComponentType.STORAGE_TARGET)
                g.add_component(tgt)
                g.add_dependency(Dependency(tgt.id, arr.id, DependencyType.STORAGE_DEPENDENCY, Criticality.HARD))

        for hv in hypervisors:
            arr = self._rng.choice([stor1, stor2])
            g.add_dependency(Dependency(hv.id, arr.id, DependencyType.STORAGE_DEPENDENCY, Criticality.HARD))

        # Infrastructure services
        dns1 = self._mc(self._gen_id("dns"), ComponentType.DNS_SERVER)
        dns2 = self._mc(self._gen_id("dns"), ComponentType.DNS_SERVER)
        dhcp = self._mc(self._gen_id("dhcp"), ComponentType.DHCP_SERVER)
        dc1 = self._mc(self._gen_id("dc"), ComponentType.DOMAIN_CONTROLLER)
        dc2 = self._mc(self._gen_id("dc"), ComponentType.DOMAIN_CONTROLLER)
        ca = self._mc(self._gen_id("ca"), ComponentType.CERTIFICATE_AUTHORITY)
        mon = self._mc(self._gen_id("mon"), ComponentType.MONITORING_SERVER)

        for svc in [dns1, dns2, dhcp, dc1, dc2, ca, mon]:
            g.add_component(svc)
            if vms:
                g.add_dependency(Dependency(svc.id, self._rng.choice(vms).id, DependencyType.HOSTING_DEPENDENCY, Criticality.HARD))
            g.add_dependency(Dependency(svc.id, self._rng.choice(access).id, DependencyType.NETWORK_PATH, Criticality.HARD))

        g.add_dependency(Dependency(dc1.id, dc2.id, DependencyType.REPLICATION_DEPENDENCY, Criticality.SOFT))

        # Load balancers
        lb1 = self._mc(self._gen_id("lb"), ComponentType.LOAD_BALANCER)
        lb2 = self._mc(self._gen_id("lb"), ComponentType.LOAD_BALANCER)
        for lb in [lb1, lb2]:
            g.add_component(lb)
            g.add_dependency(Dependency(lb.id, self._rng.choice(access).id, DependencyType.NETWORK_PATH, Criticality.HARD))

        # Application services
        n_apps = max(2, endpoints // 200)
        apps = []
        for _ in range(n_apps):
            app = self._mc(self._gen_id("app"), ComponentType.APPLICATION_SERVICE)
            g.add_component(app)
            if vms:
                g.add_dependency(Dependency(app.id, self._rng.choice(vms).id, DependencyType.HOSTING_DEPENDENCY, Criticality.HARD))
            g.add_dependency(Dependency(app.id, self._rng.choice([lb1, lb2]).id, DependencyType.SERVICE_DEPENDENCY, Criticality.SOFT))
            apps.append(app)

        # Wire service deps
        self._wire_service_deps(g, [dns1, dns2], [dc1, dc2], mon)

        logger.info("enterprise_campus: %d nodes, %d edges", g.node_count, g.edge_count)
        return g

    def vdi_environment(self, sessions: int = 500) -> InfrastructureGraph:
        """VDI environment: enterprise base + VDI brokers, hosts, dedicated storage."""
        g = self.enterprise_campus(endpoints=sessions * 2)

        # Add VDI brokers
        access = g.get_components_by_type(ComponentType.ACCESS_SWITCH)
        vms = g.get_components_by_type(ComponentType.SERVER_VIRTUAL)
        hypervisors = g.get_components_by_type(ComponentType.HYPERVISOR)

        broker1 = self._mc(self._gen_id("vdi-broker"), ComponentType.VDI_BROKER)
        broker2 = self._mc(self._gen_id("vdi-broker"), ComponentType.VDI_BROKER)
        for b in [broker1, broker2]:
            g.add_component(b)
            if vms:
                g.add_dependency(Dependency(b.id, self._rng.choice(vms).id, DependencyType.HOSTING_DEPENDENCY, Criticality.HARD))
            if access:
                g.add_dependency(Dependency(b.id, self._rng.choice(access).id, DependencyType.NETWORK_PATH, Criticality.HARD))

        # VDI hosts
        n_hosts = max(4, sessions // 25)
        for _ in range(n_hosts):
            host = self._mc(self._gen_id("vdi-host"), ComponentType.VDI_HOST)
            g.add_component(host)
            if hypervisors:
                g.add_dependency(Dependency(host.id, self._rng.choice(hypervisors).id, DependencyType.HOSTING_DEPENDENCY, Criticality.HARD))
            # VDI hosts depend on broker
            g.add_dependency(Dependency(host.id, self._rng.choice([broker1, broker2]).id, DependencyType.SERVICE_DEPENDENCY, Criticality.HARD))
            # Storage
            arrays = g.get_components_by_type(ComponentType.STORAGE_ARRAY)
            if arrays:
                g.add_dependency(Dependency(host.id, self._rng.choice(arrays).id, DependencyType.STORAGE_DEPENDENCY, Criticality.HARD))

        # Re-wire service deps for new components
        dns_servers = g.get_components_by_type(ComponentType.DNS_SERVER)
        dcs = g.get_components_by_type(ComponentType.DOMAIN_CONTROLLER)
        mon_servers = g.get_components_by_type(ComponentType.MONITORING_SERVER)
        mon = mon_servers[0] if mon_servers else None

        vdi_comps = g.get_components_by_type(ComponentType.VDI_BROKER) + g.get_components_by_type(ComponentType.VDI_HOST)
        for comp in vdi_comps:
            if dns_servers:
                g.add_dependency(Dependency(comp.id, self._rng.choice(dns_servers).id, DependencyType.DNS_DEPENDENCY, Criticality.REDUNDANT))
            if dcs:
                g.add_dependency(Dependency(comp.id, self._rng.choice(dcs).id, DependencyType.AUTHENTICATION_DEPENDENCY, Criticality.REDUNDANT))
            if mon:
                g.add_dependency(Dependency(comp.id, mon.id, DependencyType.MONITORING_DEPENDENCY, Criticality.SOFT))

        logger.info("vdi_environment: %d nodes, %d edges", g.node_count, g.edge_count)
        return g

    def healthcare_network(self, endpoints: int = 2000) -> InfrastructureGraph:
        """Healthcare: enterprise base + extra segmentation, EHR/PACS apps."""
        g = self.enterprise_campus(endpoints=endpoints)

        access = g.get_components_by_type(ComponentType.ACCESS_SWITCH)
        vms = g.get_components_by_type(ComponentType.SERVER_VIRTUAL)

        # Extra firewalls for segmentation
        for _ in range(2):
            seg_fw = self._mc(self._gen_id("fw-seg"), ComponentType.FIREWALL)
            g.add_component(seg_fw)
            cores = g.get_components_by_type(ComponentType.CORE_SWITCH)
            if cores:
                g.add_dependency(Dependency(seg_fw.id, cores[0].id, DependencyType.NETWORK_PATH, Criticality.HARD))

        # Healthcare-specific application services
        health_apps = ["ehr", "pacs", "pharmacy", "lab", "radiology"]
        for app_name in health_apps:
            app = self._mc(self._gen_id(f"app-{app_name}"), ComponentType.APPLICATION_SERVICE)
            g.add_component(app)
            if vms:
                g.add_dependency(Dependency(app.id, self._rng.choice(vms).id, DependencyType.HOSTING_DEPENDENCY, Criticality.HARD))
            if access:
                g.add_dependency(Dependency(app.id, self._rng.choice(access).id, DependencyType.NETWORK_PATH, Criticality.HARD))
            # Storage dependency for PACS
            if app_name == "pacs":
                arrays = g.get_components_by_type(ComponentType.STORAGE_ARRAY)
                if arrays:
                    g.add_dependency(Dependency(app.id, arrays[0].id, DependencyType.STORAGE_DEPENDENCY, Criticality.HARD))

        # Additional storage for imaging
        extra_arr = self._mc(self._gen_id("stor-arr"), ComponentType.STORAGE_ARRAY)
        g.add_component(extra_arr)
        if access:
            g.add_dependency(Dependency(extra_arr.id, self._rng.choice(access).id, DependencyType.NETWORK_PATH, Criticality.HARD))
        for _ in range(3):
            tgt = self._mc(self._gen_id("stor-tgt"), ComponentType.STORAGE_TARGET)
            g.add_component(tgt)
            g.add_dependency(Dependency(tgt.id, extra_arr.id, DependencyType.STORAGE_DEPENDENCY, Criticality.HARD))

        # Extra monitoring server for compliance
        mon2 = self._mc(self._gen_id("mon"), ComponentType.MONITORING_SERVER)
        g.add_component(mon2)
        if access:
            g.add_dependency(Dependency(mon2.id, self._rng.choice(access).id, DependencyType.NETWORK_PATH, Criticality.HARD))

        # Wire service deps for new components
        dns_servers = g.get_components_by_type(ComponentType.DNS_SERVER)
        dcs = g.get_components_by_type(ComponentType.DOMAIN_CONTROLLER)
        mon_servers = g.get_components_by_type(ComponentType.MONITORING_SERVER)
        mon = mon_servers[0] if mon_servers else None

        new_apps = [c for c in g.get_components_by_type(ComponentType.APPLICATION_SERVICE)]
        for comp in new_apps:
            existing_dns = g.get_dependencies_of_type(comp.id, DependencyType.DNS_DEPENDENCY)
            if not existing_dns and dns_servers:
                g.add_dependency(Dependency(comp.id, self._rng.choice(dns_servers).id, DependencyType.DNS_DEPENDENCY, Criticality.REDUNDANT))
            existing_auth = g.get_dependencies_of_type(comp.id, DependencyType.AUTHENTICATION_DEPENDENCY)
            if not existing_auth and dcs:
                g.add_dependency(Dependency(comp.id, self._rng.choice(dcs).id, DependencyType.AUTHENTICATION_DEPENDENCY, Criticality.REDUNDANT))
            existing_mon = g.get_dependencies_of_type(comp.id, DependencyType.MONITORING_DEPENDENCY)
            if not existing_mon and mon:
                g.add_dependency(Dependency(comp.id, mon.id, DependencyType.MONITORING_DEPENDENCY, Criticality.SOFT))

        logger.info("healthcare_network: %d nodes, %d edges", g.node_count, g.edge_count)
        return g

    # ------------------------------------------------------------------
    # Service dependency wiring
    # ------------------------------------------------------------------

    def _wire_service_deps(
        self,
        g: InfrastructureGraph,
        dns_servers: list[Component],
        dcs: list[Component],
        mon: Component,
    ) -> None:
        """Add DNS, auth, and monitoring dependencies to all applicable components."""
        _infra = {
            ComponentType.CORE_SWITCH, ComponentType.ACCESS_SWITCH,
            ComponentType.WAN_LINK, ComponentType.INTERNET_GATEWAY,
        }
        _auth_types = {
            ComponentType.SERVER_PHYSICAL, ComponentType.SERVER_VIRTUAL,
            ComponentType.HYPERVISOR, ComponentType.APPLICATION_SERVICE,
            ComponentType.VDI_BROKER, ComponentType.VDI_HOST,
        }

        for comp in g.get_all_components():
            # DNS
            if comp.type not in _infra and comp.type != ComponentType.DNS_SERVER:
                dns = self._rng.choice(dns_servers)
                crit = Criticality.REDUNDANT if len(dns_servers) > 1 else Criticality.SOFT
                g.add_dependency(Dependency(
                    comp.id, dns.id, DependencyType.DNS_DEPENDENCY, crit,
                ))

            # Auth
            if comp.type in _auth_types:
                dc = self._rng.choice(dcs)
                crit = Criticality.REDUNDANT if len(dcs) > 1 else Criticality.SOFT
                g.add_dependency(Dependency(
                    comp.id, dc.id, DependencyType.AUTHENTICATION_DEPENDENCY, crit,
                ))

            # Monitoring
            if comp.id != mon.id:
                if self._rng.random() < 0.9:
                    g.add_dependency(Dependency(
                        comp.id, mon.id, DependencyType.MONITORING_DEPENDENCY, Criticality.SOFT,
                    ))
