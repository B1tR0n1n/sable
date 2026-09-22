"""The template table: (component_type, root-cause state) -> a step template.

A Template names ONE catalog action and says how to fill each of its params
from the finding, the topology and the operator's golden config. Param
sources:

  service          the compose service of the root-cause node (service_map, identity by default)
  node             the root-cause node id itself
  replica_service  the service of the node the root cause REPLICATES to
                   (a REPLICATION_DEPENDENCY edge root_cause -> replica)
  proxy_service    the service of a LOAD_BALANCER that directly depends on the root cause
  golden_file / golden_key / golden_value
                   the node's known-good config entry from the golden map
                   (lab/golden.yaml: node_id -> {file, key, value})

When a source cannot be resolved — or the template's action is disabled —
`fallback` is tried; with no fallback the planner raises NoTemplate.
Nothing here is executed: the template planner builds a Plan from the row
and validate_plan re-checks it against the catalog like any other plan.

Why these rows: a fix must be VERIFIABLE by SABLE afterwards (Phase 5
checks the blast radius is healthy). A degraded app is most often a bad
config — restoring the golden value is the one fix that both heals it and
is exactly reversible (OVERLORD's snapshot holds the file); a restart is
the fallback. A dead primary comes back with a restart; a failover only
moves the proxy and leaves the primary — and the app that talks to it —
down, so it is not a template: the LLM planner may still propose it, and
it stays in the catalog for an operator who knows the primary is gone.
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
GOLDEN = {"file": "golden_file", "key": "golden_key", "value": "golden_value", "service": "service"}

_DNS_RESTART = Template("dns_failed_restart", "restart_service", RESTART)
_DNS_CLEAR = Template("dns_degraded_clear_cache", "clear_dns_cache", RESTART, fallback=_DNS_RESTART)
_APP_RESTART = Template("app_unhealthy_restart", "restart_service", RESTART)
_APP_GOLDEN = Template("app_degraded_restore_config", "set_config_value", GOLDEN, fallback=_APP_RESTART)
_DB_RESTART = Template("db_failed_restart", "restart_service", RESTART)
_DB_DEGRADED = Template("db_degraded_restart", "restart_service", RESTART)
_LB_RESTART = Template("lb_unhealthy_restart", "restart_service", RESTART)
_LB_GOLDEN = Template("lb_degraded_restore_config", "set_config_value", GOLDEN, fallback=_LB_RESTART)
_MON_RESTART = Template("monitoring_unhealthy_restart", "restart_service", RESTART)

# kept for the LLM planner's vocabulary and for tests of the failover binding;
# not in the table (see the module docstring)
_DB_FAILOVER = Template("db_failed_failover", "failover_to_replica",
                        {"primary": "service", "replica": "replica_service", "proxy": "proxy_service"},
                        fallback=_DB_RESTART)

TEMPLATES: dict[tuple[str, str], Template] = {
    **{("DNS_SERVER", s): _DNS_RESTART for s in FAILED_STATES},
    **{("DNS_SERVER", s): _DNS_CLEAR for s in DEGRADED_STATES},
    **{("APPLICATION_SERVICE", s): _APP_RESTART for s in FAILED_STATES},
    **{("APPLICATION_SERVICE", s): _APP_GOLDEN for s in DEGRADED_STATES},
    **{("SERVER_VIRTUAL", s): _DB_RESTART for s in FAILED_STATES},
    **{("SERVER_VIRTUAL", s): _DB_DEGRADED for s in DEGRADED_STATES},
    **{("LOAD_BALANCER", s): _LB_RESTART for s in FAILED_STATES},
    **{("LOAD_BALANCER", s): _LB_GOLDEN for s in DEGRADED_STATES},
    **{("MONITORING_SERVER", s): _MON_RESTART for s in FAILED_STATES + DEGRADED_STATES},
    # STORAGE_ARRAY: no template — the loop reports only (NoTemplate)
}

VERIFICATION_WINDOW_S = 30
