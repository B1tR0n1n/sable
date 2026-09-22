# OVERLORD x SABLE console: lab targets (Phase 8) and tests. Use from anywhere:
#     make -C console lab-up
# Every path is relative to this Makefile's directory; nothing is added to SABLE's root.

CONSOLE_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ROOT        := $(patsubst %/,%,$(dir $(CONSOLE_DIR)))
LAB         := $(CONSOLE_DIR)/lab
COMPOSE     := docker compose -f $(LAB)/docker-compose.yml
PYTHON      ?= python3
FAULTS      := stop_service corrupt_config kill_primary poison_dns heal_all

.PHONY: help install up down status logs lab-up lab-down lab-reset lab-fault lab-status lab-e2e test

help:
	@echo "install     one-shot bootstrap: OVERLORD, deps, Linux node + UI, docker + weights checks (console/install.sh)"
	@echo "up          start everything: lab, OVERLORD daemon, SABLE, live monitor, console (console/run.sh)"
	@echo "down        stop everything"
	@echo "status      what is running, the console state, the lab state"
	@echo "logs        tail every process log (console/.run/*.log)"
	@echo "lab-up      build + start the lab and wait for every healthcheck"
	@echo "lab-down    stop and remove the lab (containers, network, volumes)"
	@echo "lab-reset   restore pristine configs, recreate the stack (faults/heal_all.sh)"
	@echo "lab-fault   inject a fault: make lab-fault FAULT=<$(FAULTS)> [ARGS=dns]"
	@echo "lab-status  compose state, /health of each service, Prometheus targets and probes"
	@echo "lab-e2e     run the end-to-end suite against the console server (console/lab/e2e.py)"
	@echo "test        pytest console/tests"

lab-up:
	$(COMPOSE) up -d --build --wait --wait-timeout 120

lab-down:
	$(COMPOSE) down -v --remove-orphans

lab-reset:
	bash $(LAB)/faults/heal_all.sh

lab-fault:
	@test -n "$(FAULT)" || { echo "usage: make lab-fault FAULT=<$(FAULTS)> [ARGS=...]"; exit 2; }
	@test -f "$(LAB)/faults/$(FAULT).sh" || { echo "no such fault: $(FAULT) (have: $(FAULTS))"; exit 2; }
	bash $(LAB)/faults/$(FAULT).sh $(ARGS)

lab-status:
	bash $(LAB)/status.sh

# The console server (Phase 7) must be running with CONSOLE_LAB=1, or pass
# E2E_CONSOLE_CMD="python3 -m console.server" so e2e.py starts/restarts it itself (the
# negative scenario needs it restarted with CONSOLE_DISABLE_ACTIONS=set_config_value).
lab-e2e:
	cd $(ROOT) && CONSOLE_LAB=1 LAB_DIR=$(LAB) E2E_CONSOLE_CMD="$${E2E_CONSOLE_CMD:-$(PYTHON) -m console.server}" $(PYTHON) -m console.lab.e2e

test:
	cd $(ROOT) && $(PYTHON) -m pytest console/tests -q

install:
	bash $(CONSOLE_DIR)/install.sh

up:
	bash $(CONSOLE_DIR)/run.sh up

down:
	bash $(CONSOLE_DIR)/run.sh down

status:
	bash $(CONSOLE_DIR)/run.sh status

logs:
	bash $(CONSOLE_DIR)/run.sh logs
