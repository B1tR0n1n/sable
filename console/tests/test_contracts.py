"""Phase 1 acceptance: round-trip serialisation, exported schemas, one
fixture per object — and the contract invariants the later phases rely on."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from console.contracts import (
    Approval, BlastRadius, Compensation, Confidence, Finding, Gate, NodeRef, Plan,
    Receipt, Reversibility, Step, StepResult, Verification, VerificationResult,
)
from console.contracts.export_schema import fixtures
from console.contracts.models import verify_chain

ROOT = Path(__file__).resolve().parents[1] / "contracts"
T0 = datetime(2026, 9, 22, 14, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("name", ["finding", "plan", "receipt"])
def test_round_trip(name):
    obj = fixtures()[name]
    dumped = obj.model_dump(mode="json")
    back = type(obj).model_validate(dumped)
    assert back == obj
    assert type(obj).model_validate_json(json.dumps(dumped)) == obj


@pytest.mark.parametrize("name", ["finding", "plan", "receipt"])
def test_schema_exported_and_fixture_present(name):
    schema = ROOT / "schema" / f"{name}.schema.json"
    fixture = ROOT / "fixtures" / f"{name}.json"
    assert schema.is_file(), "run: python -m console.contracts.export_schema"
    assert fixture.is_file()
    s = json.loads(schema.read_text())
    assert s.get("title") == name.capitalize()
    assert s.get("additionalProperties") is False        # extra="forbid" is in the schema
    # the committed fixture is exactly what the models produce today
    assert json.loads(fixture.read_text()) == fixtures()[name].model_dump(mode="json")


def test_extra_fields_are_rejected_not_repaired():
    data = fixtures()["finding"].model_dump(mode="json")
    data["definitely_not_a_field"] = 1
    with pytest.raises(ValidationError):
        Finding.model_validate(data)


def test_plan_reversibility_is_worst_step():
    p = fixtures()["plan"]
    assert p.reversibility == Reversibility.compensable
    p2 = p.model_copy(update={"steps": p.steps + [Step(
        step_id="s2", action_id="set_config_value", target_node="dns-1", params={},
        reversibility="irreversible", precondition="none", timeout_s=5)]})
    assert p2.reversibility == Reversibility.irreversible


def test_plan_invariants():
    base = fixtures()["plan"]
    with pytest.raises(ValidationError):             # compensable without a compensation
        Step(step_id="x", action_id="a", target_node="n", reversibility="compensable",
             precondition="none", timeout_s=1)
    with pytest.raises(ValidationError):             # blast radius count must match
        BlastRadius(nodes=["a"], count=2)
    with pytest.raises(ValidationError):             # delay needs delay_s
        Gate(decision="delay", rule_id="r", reason="x")
    with pytest.raises(ValidationError):             # a plan needs at least one step
        Plan(finding_id="f", steps=[], blast_radius=BlastRadius(), verification=Verification(predicate="p", window_s=1))
    dup = [base.steps[0], base.steps[0].model_copy()]
    with pytest.raises(ValidationError):             # unique step ids
        base.model_copy(update={"steps": dup}); Plan.model_validate(base.model_copy(update={"steps": dup}).model_dump(mode="json"))


def test_confidence_mc_dropout_requires_samples():
    with pytest.raises(ValidationError):
        Confidence(score=0.5, method="mc_dropout")
    assert Confidence(score=0.5, method="mc_dropout", samples=10).samples == 10
    with pytest.raises(ValidationError):
        Confidence(score=1.5, method="heuristic")


def test_receipt_hash_chain_detects_tampering():
    r1 = fixtures()["receipt"]
    assert r1.verify_hash()
    r2 = Receipt(plan_id="pln-2", steps=[StepResult(step_id="s1", status="ok")],
                 verification=VerificationResult(status="fail", checked_at=T0), created_at=T0)
    r2.seal(prev_receipt_hash=r1.receipt_hash)
    assert verify_chain([r1, r2]) == (True, None)
    tampered = r1.model_copy(update={"approvals": [Approval(actor="mallory", decision="approve", timestamp=T0)]})
    assert not tampered.verify_hash()
    assert verify_chain([tampered, r2]) == (False, 0)
    assert verify_chain([r2]) == (False, 0)            # r2's prev points at a receipt that is not there


def test_finding_dedup_key():
    f = fixtures()["finding"]
    assert f.dedup_key == ("lab-1", "dns-1", "failed")
    g = f.model_copy(update={"root_cause": NodeRef(node_id="dns-1", component_type="DNS_SERVER", state="degraded")})
    assert g.dedup_key != f.dedup_key
