"""Component definitions, state machine, and failure modes."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ComponentType(StrEnum):
    """All infrastructure component types."""

    CORE_SWITCH = "CORE_SWITCH"
    ACCESS_SWITCH = "ACCESS_SWITCH"
    FIREWALL = "FIREWALL"
    ROUTER = "ROUTER"
    LOAD_BALANCER = "LOAD_BALANCER"
    SERVER_PHYSICAL = "SERVER_PHYSICAL"
    SERVER_VIRTUAL = "SERVER_VIRTUAL"
    HYPERVISOR = "HYPERVISOR"
    STORAGE_ARRAY = "STORAGE_ARRAY"
    STORAGE_TARGET = "STORAGE_TARGET"
    VDI_BROKER = "VDI_BROKER"
    VDI_HOST = "VDI_HOST"
    DNS_SERVER = "DNS_SERVER"
    DHCP_SERVER = "DHCP_SERVER"
    DOMAIN_CONTROLLER = "DOMAIN_CONTROLLER"
    CERTIFICATE_AUTHORITY = "CERTIFICATE_AUTHORITY"
    MONITORING_SERVER = "MONITORING_SERVER"
    WAN_LINK = "WAN_LINK"
    INTERNET_GATEWAY = "INTERNET_GATEWAY"
    APPLICATION_SERVICE = "APPLICATION_SERVICE"


class ComponentState(StrEnum):
    """Operational states a component can be in."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNREACHABLE = "unreachable"


@dataclass
class FailureMode:
    """A specific way a component type can fail."""

    name: str
    applicable_types: list[ComponentType]
    is_hardware: bool
    severity: float  # 0.0–1.0


@dataclass
class Component:
    """A node in the infrastructure graph."""

    id: str
    type: ComponentType
    state: ComponentState = ComponentState.HEALTHY
    health: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    dependencies_in: list[str] = field(default_factory=list)
    dependencies_out: list[str] = field(default_factory=list)

    def apply_health_change(
        self,
        delta: float,
        degradation_threshold: float = 0.5,
        failure_threshold: float = 0.2,
    ) -> ComponentState:
        """Adjust health by *delta* (negative = worse), clamp, and update state.

        Returns the new ComponentState.
        """
        self.health = max(0.0, min(1.0, self.health + delta))
        if self.health <= failure_threshold:
            self.state = ComponentState.FAILED
        elif self.health <= degradation_threshold:
            self.state = ComponentState.DEGRADED
        else:
            self.state = ComponentState.HEALTHY
        return self.state

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "id": self.id,
            "type": str(self.type),
            "state": str(self.state),
            "health": self.health,
            "properties": dict(self.properties),
            "initial_state": str(self.state),
            "initial_health": self.health,
        }

    def copy(self) -> Component:
        """Return a deep copy."""
        return Component(
            id=self.id,
            type=self.type,
            state=self.state,
            health=self.health,
            properties=copy.deepcopy(self.properties),
            dependencies_in=list(self.dependencies_in),
            dependencies_out=list(self.dependencies_out),
        )


# ---------------------------------------------------------------------------
# Failure modes per component type
# ---------------------------------------------------------------------------

def _fm(name: str, types: list[ComponentType], hw: bool, sev: float) -> FailureMode:
    return FailureMode(name=name, applicable_types=types, is_hardware=hw, severity=sev)


FAILURE_MODES: dict[ComponentType, list[FailureMode]] = {
    ComponentType.CORE_SWITCH: [
        _fm("hardware_failure", [ComponentType.CORE_SWITCH], True, 1.0),
        _fm("cpu_overload", [ComponentType.CORE_SWITCH], False, 0.6),
        _fm("spanning_tree_loop", [ComponentType.CORE_SWITCH], False, 0.8),
        _fm("firmware_crash", [ComponentType.CORE_SWITCH], False, 0.9),
    ],
    ComponentType.ACCESS_SWITCH: [
        _fm("hardware_failure", [ComponentType.ACCESS_SWITCH], True, 1.0),
        _fm("poe_exhaustion", [ComponentType.ACCESS_SWITCH], False, 0.5),
        _fm("vlan_misconfiguration", [ComponentType.ACCESS_SWITCH], False, 0.4),
        _fm("uplink_loss", [ComponentType.ACCESS_SWITCH], False, 0.7),
    ],
    ComponentType.FIREWALL: [
        _fm("session_table_exhaustion", [ComponentType.FIREWALL], False, 0.7),
        _fm("hardware_failure", [ComponentType.FIREWALL], True, 1.0),
        _fm("rule_misconfiguration", [ComponentType.FIREWALL], False, 0.5),
        _fm("ha_failover_failure", [ComponentType.FIREWALL], False, 0.6),
    ],
    ComponentType.ROUTER: [
        _fm("route_table_corruption", [ComponentType.ROUTER], False, 0.7),
        _fm("hardware_failure", [ComponentType.ROUTER], True, 1.0),
        _fm("bgp_flap", [ComponentType.ROUTER], False, 0.5),
        _fm("memory_exhaustion", [ComponentType.ROUTER], False, 0.6),
    ],
    ComponentType.LOAD_BALANCER: [
        _fm("backend_pool_exhaustion", [ComponentType.LOAD_BALANCER], False, 0.6),
        _fm("hardware_failure", [ComponentType.LOAD_BALANCER], True, 1.0),
        _fm("ssl_cert_expiry", [ComponentType.LOAD_BALANCER], False, 0.4),
        _fm("health_check_false_positive", [ComponentType.LOAD_BALANCER], False, 0.3),
    ],
    ComponentType.SERVER_PHYSICAL: [
        _fm("hardware_failure", [ComponentType.SERVER_PHYSICAL], True, 1.0),
        _fm("cpu_exhaustion", [ComponentType.SERVER_PHYSICAL], False, 0.6),
        _fm("memory_exhaustion", [ComponentType.SERVER_PHYSICAL], False, 0.6),
        _fm("disk_failure", [ComponentType.SERVER_PHYSICAL], True, 0.8),
        _fm("nic_failure", [ComponentType.SERVER_PHYSICAL], True, 0.7),
    ],
    ComponentType.SERVER_VIRTUAL: [
        _fm("resource_contention", [ComponentType.SERVER_VIRTUAL], False, 0.5),
        _fm("host_failure_cascade", [ComponentType.SERVER_VIRTUAL], False, 0.9),
        _fm("disk_exhaustion", [ComponentType.SERVER_VIRTUAL], False, 0.6),
        _fm("network_isolation", [ComponentType.SERVER_VIRTUAL], False, 0.7),
    ],
    ComponentType.HYPERVISOR: [
        _fm("overcommit_thrash", [ComponentType.HYPERVISOR], False, 0.6),
        _fm("hardware_failure", [ComponentType.HYPERVISOR], True, 1.0),
        _fm("management_plane_loss", [ComponentType.HYPERVISOR], False, 0.5),
        _fm("storage_disconnect", [ComponentType.HYPERVISOR], False, 0.8),
    ],
    ComponentType.STORAGE_ARRAY: [
        _fm("disk_failure", [ComponentType.STORAGE_ARRAY], True, 0.5),
        _fm("controller_failure", [ComponentType.STORAGE_ARRAY], True, 0.9),
        _fm("cache_battery_failure", [ComponentType.STORAGE_ARRAY], True, 0.4),
        _fm("iops_saturation", [ComponentType.STORAGE_ARRAY], False, 0.5),
        _fm("latency_spike", [ComponentType.STORAGE_ARRAY], False, 0.4),
    ],
    ComponentType.STORAGE_TARGET: [
        _fm("protocol_error", [ComponentType.STORAGE_TARGET], False, 0.6),
        _fm("session_exhaustion", [ComponentType.STORAGE_TARGET], False, 0.5),
        _fm("authentication_failure", [ComponentType.STORAGE_TARGET], False, 0.7),
        _fm("parent_array_cascade", [ComponentType.STORAGE_TARGET], False, 0.8),
    ],
    ComponentType.VDI_BROKER: [
        _fm("broker_overload", [ComponentType.VDI_BROKER], False, 0.6),
        _fm("database_corruption", [ComponentType.VDI_BROKER], False, 0.8),
        _fm("licensing_exhaustion", [ComponentType.VDI_BROKER], False, 0.5),
        _fm("pool_misconfiguration", [ComponentType.VDI_BROKER], False, 0.4),
    ],
    ComponentType.VDI_HOST: [
        _fm("resource_exhaustion", [ComponentType.VDI_HOST], False, 0.6),
        _fm("profile_corruption", [ComponentType.VDI_HOST], False, 0.5),
        _fm("gpu_failure", [ComponentType.VDI_HOST], True, 0.7),
        _fm("host_agent_crash", [ComponentType.VDI_HOST], False, 0.5),
    ],
    ComponentType.DNS_SERVER: [
        _fm("service_crash", [ComponentType.DNS_SERVER], False, 0.9),
        _fm("cache_poisoning", [ComponentType.DNS_SERVER], False, 0.6),
        _fm("zone_transfer_failure", [ComponentType.DNS_SERVER], False, 0.4),
        _fm("response_timeout", [ComponentType.DNS_SERVER], False, 0.5),
    ],
    ComponentType.DHCP_SERVER: [
        _fm("scope_exhaustion", [ComponentType.DHCP_SERVER], False, 0.5),
        _fm("service_crash", [ComponentType.DHCP_SERVER], False, 0.8),
        _fm("rogue_server_conflict", [ComponentType.DHCP_SERVER], False, 0.6),
    ],
    ComponentType.DOMAIN_CONTROLLER: [
        _fm("replication_failure", [ComponentType.DOMAIN_CONTROLLER], False, 0.5),
        _fm("ldap_timeout", [ComponentType.DOMAIN_CONTROLLER], False, 0.6),
        _fm("kerberos_failure", [ComponentType.DOMAIN_CONTROLLER], False, 0.7),
        _fm("fsmo_seizure_needed", [ComponentType.DOMAIN_CONTROLLER], False, 0.6),
    ],
    ComponentType.CERTIFICATE_AUTHORITY: [
        _fm("cert_expiry", [ComponentType.CERTIFICATE_AUTHORITY], False, 0.5),
        _fm("crl_distribution_failure", [ComponentType.CERTIFICATE_AUTHORITY], False, 0.4),
        _fm("private_key_compromise", [ComponentType.CERTIFICATE_AUTHORITY], False, 0.9),
    ],
    ComponentType.MONITORING_SERVER: [
        _fm("ingest_overload", [ComponentType.MONITORING_SERVER], False, 0.5),
        _fm("storage_exhaustion", [ComponentType.MONITORING_SERVER], False, 0.6),
        _fm("agent_disconnect", [ComponentType.MONITORING_SERVER], False, 0.4),
    ],
    ComponentType.WAN_LINK: [
        _fm("link_down", [ComponentType.WAN_LINK], True, 1.0),
        _fm("saturation", [ComponentType.WAN_LINK], False, 0.5),
        _fm("latency_spike", [ComponentType.WAN_LINK], False, 0.4),
        _fm("packet_loss", [ComponentType.WAN_LINK], False, 0.5),
    ],
    ComponentType.INTERNET_GATEWAY: [
        _fm("session_exhaustion", [ComponentType.INTERNET_GATEWAY], False, 0.6),
        _fm("hardware_failure", [ComponentType.INTERNET_GATEWAY], True, 1.0),
        _fm("isp_outage", [ComponentType.INTERNET_GATEWAY], False, 0.9),
        _fm("ddos", [ComponentType.INTERNET_GATEWAY], False, 0.7),
    ],
    ComponentType.APPLICATION_SERVICE: [
        _fm("crash", [ComponentType.APPLICATION_SERVICE], False, 0.9),
        _fm("memory_leak", [ComponentType.APPLICATION_SERVICE], False, 0.4),
        _fm("dependency_timeout", [ComponentType.APPLICATION_SERVICE], False, 0.5),
        _fm("configuration_error", [ComponentType.APPLICATION_SERVICE], False, 0.5),
        _fm("thread_pool_exhaustion", [ComponentType.APPLICATION_SERVICE], False, 0.6),
    ],
}


# ---------------------------------------------------------------------------
# Default properties per component type
# ---------------------------------------------------------------------------

DEFAULT_PROPERTIES: dict[ComponentType, dict[str, float]] = {
    ComponentType.CORE_SWITCH: {
        "port_count": 48, "throughput": 10000, "cpu_util": 0.15, "memory_util": 0.30, "uplink_count": 4
    },
    ComponentType.ACCESS_SWITCH: {
        "port_count": 24, "throughput": 1000, "cpu_util": 0.10, "vlan_count": 5, "poe_budget": 370
    },
    ComponentType.FIREWALL: {
        "throughput": 5000, "session_count": 50000, "max_sessions": 100000, "cpu_util": 0.25, "rule_count": 200
    },
    ComponentType.ROUTER: {
        "route_table_size": 500, "throughput": 5000, "cpu_util": 0.15, "bgp_neighbor_count": 4
    },
    ComponentType.LOAD_BALANCER: {
        "active_connections": 5000, "backend_count": 8, "throughput": 5000, "health_check_interval": 10
    },
    ComponentType.SERVER_PHYSICAL: {
        "cpu_cores": 16, "cpu_util": 0.30, "memory_total": 64, "memory_util": 0.40,
        "disk_iops": 5000, "disk_util": 0.50, "nic_count": 2,
    },
    ComponentType.SERVER_VIRTUAL: {
        "cpu_cores": 4, "cpu_util": 0.35, "memory_total": 16, "memory_util": 0.50,
        "disk_iops": 1000, "host_id": 0,
    },
    ComponentType.HYPERVISOR: {
        "vm_count": 10, "cpu_overcommit_ratio": 4.0, "memory_overcommit_ratio": 1.5,
        "cpu_util": 0.45, "memory_util": 0.60,
    },
    ComponentType.STORAGE_ARRAY: {
        "total_capacity": 50000, "used_capacity": 30000, "iops": 50000,
        "latency_ms": 2.0, "raid_level": 6, "disk_count": 24,
    },
    ComponentType.STORAGE_TARGET: {
        "protocol": 1, "throughput": 1000, "session_count": 50, "latency_ms": 3.0, "parent_array_id": 0
    },
    ComponentType.VDI_BROKER: {
        "session_count": 200, "max_sessions": 500, "active_pools": 3, "cpu_util": 0.30, "memory_util": 0.40
    },
    ComponentType.VDI_HOST: {
        "session_count": 20, "max_sessions": 50, "cpu_util": 0.60, "memory_util": 0.70,
        "gpu_util": 0.50, "parent_hypervisor_id": 0,
    },
    ComponentType.DNS_SERVER: {
        "query_rate": 1000, "cache_size": 50000, "zone_count": 10, "response_time_ms": 1.0
    },
    ComponentType.DHCP_SERVER: {
        "scope_utilization": 0.60, "lease_count": 500, "max_leases": 1000
    },
    ComponentType.DOMAIN_CONTROLLER: {
        "replication_status": 1, "ldap_response_ms": 5.0, "authentication_rate": 200, "fsmo_roles": 0
    },
    ComponentType.CERTIFICATE_AUTHORITY: {
        "cert_count": 500, "crl_size": 100, "ocsp_response_time": 2.0
    },
    ComponentType.MONITORING_SERVER: {
        "agent_count": 80, "metric_ingest_rate": 10000, "alert_count": 5, "storage_util": 0.40
    },
    ComponentType.WAN_LINK: {
        "bandwidth": 1000, "latency_ms": 10.0, "jitter_ms": 1.0, "packet_loss_pct": 0.01, "utilization_pct": 0.30
    },
    ComponentType.INTERNET_GATEWAY: {
        "throughput": 2000, "nat_session_count": 20000, "max_sessions": 50000, "public_ip_count": 4
    },
    ComponentType.APPLICATION_SERVICE: {
        "response_time_ms": 50.0, "request_rate": 500, "error_rate": 0.01,
        "instance_count": 2, "health_endpoint": 1,
    },
}
