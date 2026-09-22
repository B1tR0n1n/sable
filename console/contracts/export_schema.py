#!/usr/bin/env python3
"""Regenerate contracts/schema/*.schema.json and contracts/fixtures/*.json.

    python -m console.contracts.export_schema

The fixtures are one valid example per contract, built from the models so
they can never drift from them; the test suite checks both stay in sync."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import (
    Approval,
    BlastRadius,
    Compensation,
    Confidence,
    Evidence,
    Finding,
    Gate,
    NodeRef,
    Plan,
    PlannerProvenance,
    Receipt,
    Rollback,
    Snapshot,
    Step,
    StepResult,
    Verification,
    VerificationResult,
    export_schemas,
)

HERE = Path(__file__).parent
T0 = datetime(2026, 9, 22, 14, 0, 0, tzinfo=timezone.utc)


def fixtures() -> dict[str, object]:
    finding = Finding(
        id="fnd-example000001", created_at=T0, site_id="lab-1",
        detection_mode="live_feed",
        root_cause=NodeRef(node_id="dns-1", component_type="DNS_SERVER", state="failed"),
        affected_nodes=[NodeRef(node_id="app-erp", component_type="APPLICATION_SERVICE", state="degraded"),
                        NodeRef(node_id="vdi-broker", component_type="VDI_BROKER", state="degraded")],
        evidence=[Evidence(metric="up", value=0.0, unit="bool", threshold=1.0, timestamp=T0),
                  Evidence(metric="dns_query_latency_ms", value=0.0, unit="ms", timestamp=T0)],
        confidence=Confidence(score=0.91, method="engine_native"),
        severity="high",
        summary="dns-1 is failed; app-erp and vdi-broker depend on it and are degraded.",
        summary_generated=True,
        engine_version="sable@62595f4",
    )
    plan = Plan(
        id="pln-example000001", finding_id=finding.id, created_at=T0,
        steps=[Step(step_id="s1", action_id="restart_service", target_node="dns-1",
                    params={"service": "dnsmasq"}, reversibility="compensable",
                    compensation=Compensation(action_id="restart_service", params={"service": "dnsmasq"}),
                    precondition="service_exists", timeout_s=60)],
        blast_radius=BlastRadius(nodes=["dns-1", "app-erp", "vdi-broker"], count=3),
        verification=Verification(predicate="node_healthy", window_s=30),
        gate=Gate(decision="human", rule_id="default-deny", reason="shipped default: every cell is human"),
        planner=PlannerProvenance(kind="template", template_id="failed_service_restart"),
    )
    receipt = Receipt(
        id="rcp-example000001", plan_id=plan.id, finding_id=finding.id, created_at=T0,
        session_id="20260922-140000-abc123",
        approvals=[Approval(actor="operator@lab", decision="approve", timestamp=T0)],
        steps=[StepResult(step_id="s1", started_at=T0, ended_at=T0, status="ok",
                          output_digest="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                          session_id="20260922-140000-abc123")],
        snapshots=[Snapshot(node_id="dns-1", snapshot_ref="20260922-140000-abc123")],
        verification=VerificationResult(status="pass", observed={"dns-1": "healthy"}, checked_at=T0),
        rollback=Rollback(performed=False),
    ).seal(prev_receipt_hash=None)
    return {"finding": finding, "plan": plan, "receipt": receipt}


def main(argv=None) -> int:
    schemas = export_schemas(HERE / "schema")
    fx_dir = HERE / "fixtures"
    fx_dir.mkdir(exist_ok=True)
    for name, obj in fixtures().items():
        (fx_dir / f"{name}.json").write_text(
            json.dumps(obj.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    for name, p in schemas.items():
        print(f"schema  {p.relative_to(HERE.parent.parent)}")
    for name in fixtures():
        print(f"fixture {(fx_dir / (name + '.json')).relative_to(HERE.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
