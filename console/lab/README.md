# Lab (Phase 8): a safe, reproducible world to break and heal

A docker-compose stack SABLE observes through its **unchanged** Prometheus adapter and the
console fixes through the action catalog. Nothing here touches real infrastructure.

```
proxy (nginx, LOAD_BALANCER) --> db (kv primary, SERVER_VIRTUAL) ~~replication~~> db-replica
   |                              ^
   +--> app (APPLICATION_SERVICE) +      app resolves db.lab through dns (dnsmasq, DNS_SERVER)
prometheus (MONITORING_SERVER) scrapes all of them; blackbox probes dns:53 and http://proxy:80/health
```

| service | image | port | metrics |
|---|---|---|---|
| dns | `images/dnsmasq` (alpine:3.20 + dnsmasq) | 53/udp (host 15353) | none: blackbox `dns_db_lab` probe, instance `dns:53` |
| app | python:3.12-alpine, `services/app.py` | 8000 (host 18000) | `up`, `app_dependency_ok{dep="dns"\|"db"}`, `app_requests_total`, `app_errors_total` |
| db, db-replica | python:3.12-alpine, `services/kv.py` | 8100 (host 18100 / 18101) | `up`, `kv_role{role=…}`, `kv_keys`, `kv_replication_ok` (replica) |
| proxy | nginx:1.27-alpine, `proxy/default.conf.template` | 80 (host 18080) | `/metrics` shim (`nginx_up`), `/stub_status`; `/health` = the fronted primary's |
| prometheus | prom/prometheus:v2.53.0 | 9090 (host 9090) | itself |
| blackbox | prom/blackbox-exporter:v0.25.0 | 9115 (host 19115) | probes; not a topology node |

Fixed IPs on `172.28.0.0/24` (dns .10, app .20, db .30, db-replica .31, proxy .40,
prometheus .50, blackbox .51) so dnsmasq holds static A records (`config/lab.hosts`) and
nginx's upstream survives restarts. The app container's resolver is the dns container
(`dns: [172.28.0.10]`): Docker's embedded DNS answers container names and forwards
`*.lab` to dnsmasq, so a dead or poisoned dnsmasq really breaks the app.

## Prerequisites (WSL2 + Docker Desktop, or any Linux docker)

- Docker Engine with the compose plugin (`docker compose version` ≥ v2). On Docker Desktop
  enable WSL integration for your distro; run everything from the WSL2 shell, in a checkout
  that lives on the Linux filesystem (not `/mnt/c`, bind mounts and exec bits suffer there).
- `curl`, `python3`; `dig` is optional (`make lab-status` uses it for the DNS line).
- Host ports 9090, 15353/udp, 18000, 18080, 18100, 18101, 19115 free. SABLE's server keeps
  8080 and the console 7780; nothing in the lab uses those.

## Run it

```sh
make -C console lab-up          # build dnsmasq image, start, wait for every healthcheck
make -C console lab-status      # compose state, /health per service, Prometheus targets & probes
make -C console lab-fault FAULT=poison_dns          # stop_service [ARGS=<service>] | corrupt_config | kill_primary | poison_dns
make -C console lab-reset       # faults/heal_all.sh: pristine configs + recreate + wait
make -C console lab-down        # remove containers, network, volumes
make -C console lab-e2e         # end-to-end suite (needs SABLE + console, below)
make -C console test            # pytest console/tests (test_lab.py validates this directory statically)
```

Prometheus UI: <http://localhost:9090> — try `probe_success or on(instance, job) up`.

## Where SABLE must point

SABLE reads the lab through `adapters/prometheus.py` only; the live monitor is the bridge
(`docker/live_monitor.py` polls Prometheus, scores health, encodes, POSTs `/api/live_tick`).

1. **SABLE server**: `python docker/server.py` (port 8080; inside Docker it binds
   127.0.0.1, INTEGRATION-NOTES D12).
2. **Topology**: `console/lab/topology.yaml` must be the topology SABLE loads. The
   mechanism (`docker/server.py:47,73-83`): `TOPOLOGY_DIR` is **hard-coded** to
   `<sable>/adapters/topologies` (no env var), `_load_topology_metadata()` iterates
   `sorted(TOPOLOGY_DIR.glob("*.y*ml"))` and **returns after the first file**, and node
   index → id/type is by **position** in that file's `nodes` list. So either:
   - running from source: put the lab file first in that directory, e.g.
     `cp console/lab/topology.yaml adapters/topologies/00-lab.yaml` (`00-lab.yaml` sorts
     before `msp_demo.yaml`; remove it afterwards), or move `msp_demo.yaml` aside; or
   - in the SABLE container: bind-mount a directory containing only this file over the
     path `server.py` computes (`Path(__file__).parent.parent/adapters/topologies` →
     `/adapters/topologies` when `server.py` is at `/app/server.py`):
     `-v $PWD/console/lab:/adapters/topologies:ro` mounts `topology.yaml` as the only YAML
     there (`sable_prometheus.yaml` has no `nodes`, harmless but keep it in mind).

   On the live path the server also replaces `_topo_nodes` with the posted `node_ids`
   whenever the counts differ (`server.py`, `live_tick`), so ids stay right even with
   `msp_demo.yaml` loaded, but component types become `UNKNOWN` and `/api/topology` shows
   the wrong graph. `topology.yaml` lists nodes in **sorted id order** because the encoder
   posts them sorted (`adapters/encode.py:138`); `test_lab.py` asserts that order.
3. **Live monitor**: `docker/live_monitor.py` builds `PrometheusConfig(url=…)` only and
   never reads a `node_map` or `queries` from its YAML, so run plainly it auto-discovers node
   ids such as `app_app`. Use the wrapper (SABLE unchanged; needs SABLE's deps: numpy,
   requests, pyyaml, torch stack for the encoder):

   ```sh
   python3 console/lab/live_monitor_lab.py --config console/lab/sable_prometheus.yaml [--dry-run]
   ```

   `sable_prometheus.yaml` carries the keys `live_monitor.py` reads (`prometheus_url`,
   `sable_url`, `poll_interval`, `topology`, `health_overrides`) plus a `prometheus:` block
   (`url`, `auto_discover: false`, explicit `node_map`, `queries`) the wrapper turns into
   the adapter's `PrometheusConfig`.

### How SABLE sees the lab (metrics → node states)

- `node_map` keys each Prometheus `instance` to a topology node: `dns:53→dns`,
  `app:8000→app`, `db:8100→db`, `db-replica:8100→db-replica`, `proxy:80→proxy`,
  `prometheus:9090→prometheus`. `_run_queries` reads the `instance` label of every result.
- query `up` = `probe_success or on(instance, job) up`: the blackbox dns probe (which
  validates that `db.lab` answers `172.28.0.30`) stands in for dnsmasq's missing exporter;
  every other node keeps Prometheus's synthetic `up`. `_mark_unreachable`: `up < 1` ⇒
  `reachable=False` ⇒ scorer state **unreachable** (health 0).
- query `healthy` = `min by (instance, job) (app_dependency_ok)`: `HealthConfig` already
  treats `healthy` as binary; with `up` (weight 2) fine and `healthy` 0 (weight 1) the app
  scores 0.67 ⇒ **degraded** (it is reachable, but broken).
- The live monitor posts `ground_truth` states with the encoded tick; SABLE's engine
  produces the recommendation/root cause the console turns into a Finding.

| fault | Prometheus shows | SABLE states |
|---|---|---|
| `stop_service dns` | `probe_success{instance="dns:53"} 0`; `app_dependency_ok{dep="dns"} 0` | dns unreachable, app degraded |
| `corrupt_config` | `app_dependency_ok{dep="db"} 0`, `app_db_target_info{host="nowhere.lab"}`; dns/db `up 1` | app degraded only |
| `kill_primary` | `up{instance="db:8100"} 0`; `app_dependency_ok{dep="db"} 0`; `probe_success{instance="http://proxy:80/health"} 0` | db unreachable, app degraded |
| `poison_dns` | `probe_success{instance="dns:53"} 0` (answer is 172.28.0.99); `app_dependency_ok{dep="db"} 0` | dns unreachable, app degraded |

## Where the console must point

`CONSOLE_LAB=1` enables `POST /api/lab/fault`, which runs `console/lab/faults/<name>.sh`.
The catalog's executor context is `lab_dir = console/lab`, `lab_compose =
console/lab/docker-compose.yml`; every binding is `docker compose -f <lab_compose> …`, so
the console (or OVERLORD's executor) runs on the host that owns the docker socket. The
console's `Topology` must be loaded from `console/lab/topology.yaml` (same positional
order as SABLE's). `SABLE_URL` defaults to `http://127.0.0.1:8080`.

Catalog ↔ lab conventions (console/catalog/actions.yaml):

| action | what it does to the lab | heals |
|---|---|---|
| `restart_service {service}` | `docker compose restart <service>` (also starts a stopped one) | stop_service, kill_primary, poison_dns (dns entrypoint regenerates records) |
| `set_config_value {file: config/app.conf, key: db_host, value: db.lab, service: app}` | rewrites the key=value line, restarts app | corrupt_config |
| `clear_dns_cache {service: dns}` | `docker compose kill -s HUP dns`: the entrypoint (PID 1) regenerates `/run/dnsmasq/hosts` from pristine `lab.hosts` and HUPs dnsmasq | poison_dns |
| `failover_to_replica {primary: db, replica: db-replica, proxy: proxy}` | sets `UPSTREAM_HOST=db-replica` in `proxy/upstream.env`, restarts proxy (the container sources the file at start, then nginx's envsubst renders the template) | the proxy's path during kill_primary; app still talks to `db.lab` directly, so `restart_service db` is the plan that turns everything green |

Why poison lives in the container, not in `dnsmasq.conf`: dnsmasq re-reads hosts files on
SIGHUP but never its config file, so a poison in the bind-mounted config could not be
healed by any catalog action.

## Faults

Each script in `faults/` is `set -euo pipefail`, finds the lab from its own path, is
idempotent and prints one line per thing it did. `heal_all.sh` restores every
`config/*.pristine` and `proxy/*.pristine` over its live twin and recreates the stack;
`make lab-reset` runs it. `stop_service.sh` takes the service as `$1` (or `$SERVICE`) and
defaults to `dns`, so the console's `{"name": "stop_service"}` stops dns.

## End-to-end suite

`make -C console lab-e2e` runs `python3 -m console.lab.e2e` against the console at
`CONSOLE_URL` (default `http://127.0.0.1:7780`), using only documented endpoints: for each
of `stop_service dns → dns`, `corrupt_config → app`, `kill_primary → db`,
`poison_dns → dns` it heals the lab, injects, waits for the Finding, approves its Plan as
`e2e`, and requires a Receipt with `verification.status == "pass"` and the Finding
`closed`. The negative scenario re-injects `corrupt_config` while `set_config_value` is
unavailable and requires `fail` + `rollback.performed` + `reopened`. The API has no
"force a wrong plan" knob, so that phase needs the console started with
`CONSOLE_DISABLE_ACTIONS=set_config_value`; pass `E2E_CONSOLE_CMD="python3 -m
console.server"` and the runner starts and restarts the console itself. `--only`,
`--skip-negative`, `--list`; timeouts via `E2E_FINDING_TIMEOUT`, `E2E_RECEIPT_TIMEOUT`.
