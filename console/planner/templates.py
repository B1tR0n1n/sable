"""The template table: (component_type, root-cause state) -> a step template.

A Template names ONE catalog action and says how to fill each of its params
from the finding and the topology. Param sources:

  service          the compose service of the root-cause node (service_map, identity by default)
  node             the root-cause node id itself
  replica_service  the service of the node the root cause REPLICATES to
                   (a REPLICATION_DEPENDENCY edge root_cause -> replica)
  proxy_service    the service of a LOAD_BALANCER that directly depends on the root cause

When a source cannot be resolved from the topology, `fallback` is tried;
with no fallback the planner raises NoTemplate. Nothing here is executed:
the template planner builds a Plan from the row and validate_plan re-checks
it against the catalog like any other plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

FAILED_STATES = ("failed", "unreachable")
DEGRADED_STATES = ("degraded", "oscillating")


@dataclass(frozen=True)
class Template:
    template_id: str
    action_id: str
    params: dict[str, str] = field(default_factory=dict)     # param name -> source
    fallback: Optional["Template"] = None


RESTART = {"service": "service"}

_DNS_RESTART = Template("dns_failed_restart", "restart_service", RESTART)
_DNS_CLEAR = Template("dns_degraded_clear_cache", "clear_dns_cache", RESTART)
_APP_RESTART = Template("app_unhealthy_restart", "restart_service", RESTART)
_DB_RESTART = Template("db_failed_restart", "restart_service", RESTART)
_DB_FAILOVER = Template("db_failed_failover", "failover_to_replica",
                        {"primary": "service", "replica": "replica_service", "proxy": "proxy_service"},
                        fallback=_DB_RESTART)
_DB_DEGRADED = Template("db_degraded_restart", "restart_service", RESTART)
_LB_RESTART = Template("lb_unhealthy_restart", "restart_service", RESTART)
_MON_RESTART = Template("monitoring_unhealthy_restart", "restart_service", RESTART)

TEMPLATES: dict[tuple[str, str], Template] = {
    **{("DNS_SERVER", s): _DNS_RESTART for s in FAILED_STATES},
    **{("DNS_SERVER", s): _DNS_CLEAR for s in DEGRADED_STATES},
    **{("APPLICATION_SERVICE", s): _APP_RESTART for s in FAILED_STATES + DEGRADED_STATES},
    **{("SERVER_VIRTUAL", s): _DB_FAILOVER for s in FAILED_STATES},
    **{("SERVER_VIRTUAL", s): _DB_DEGRADED for s in DEGRADED_STATES},
    **{("LOAD_BALANCER", s): _LB_RESTART for s in FAILED_STATES + DEGRADED_STATES},
    **{("MONITORING_SERVER", s): _MON_RESTART for s in FAILED_STATES + DEGRADED_STATES},
    # STORAGE_ARRAY: no template — the loop reports only (NoTemplate)
}

VERIFICATION_WINDOW_S = 30
