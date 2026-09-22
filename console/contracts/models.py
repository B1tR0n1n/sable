"""Finding, Plan, Receipt — the console's data contracts.

Field lists follow PLAN.md Phase 1 exactly. Where the later phases need a
field the plan did not spell out (a Finding's open/closed status for dedup and
verification, a Plan's planner provenance, a Receipt's link to the audit
chain), it is added with a default and marked `# additive` so the plan's
shape remains a strict subset.

Conventions
  ids          strings; generated (uuid4 hex, prefixed) when not supplied
  timestamps   timezone-aware datetimes, serialised as ISO-8601
  hashes       sha256 hex over canonical JSON (sorted keys, no whitespace)
  strictness   extra="forbid" on every model
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def canonical_json(data: Any) -> str:
    """Deterministic JSON for hashing: sorted keys, compact separators,
    datetimes already rendered by pydantic's JSON mode."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True, validate_assignment=True)


# ---------------------------------------------------------------- vocabularies


class DetectionMode(str, Enum):
    live_feed = "live_feed"                # diagnosis rests on current telemetry
    unmonitored_gap = "unmonitored_gap"    # diagnosis rests on the ABSENCE of telemetry


class ConfidenceMethod(str, Enum):
    mc_dropout = "mc_dropout"              # agreement/variance over MC-dropout samples
    engine_native = "engine_native"        # the engine's own per-node confidence (max softmax)
    heuristic = "heuristic"                # anything else, declared as such


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class FindingStatus(str, Enum):            # additive: dedup (Phase 2), verification (Phase 5), lifecycle
    open = "open"
    closed = "closed"                      # a verified fix; the receipt is attached
    reopened = "reopened"                  # a fix failed verification and was compensated
    resolved = "resolved"                  # cleared on its own — no action was taken (a transient blip)
    escalated = "escalated"                # still open; re-planning is capped, a human decides


class Reversibility(str, Enum):
    reversible = "reversible"              # undone exactly (a snapshot restores it)
    compensable = "compensable"            # undone by a compensating action
    irreversible = "irreversible"          # cannot be undone

    @property
    def rank(self) -> int:
        return {"reversible": 0, "compensable": 1, "irreversible": 2}[self.value]


class GateDecision(str, Enum):
    auto = "auto"
    delay = "delay"
    human = "human"
    human_plus = "human_plus"
    report_only = "report_only"


class StepStatus(str, Enum):
    pending = "pending"
    running = "running"
    ok = "ok"
    failed = "failed"
    skipped = "skipped"
    precondition_failed = "precondition_failed"
    timed_out = "timed_out"


class VerificationStatus(str, Enum):
    passed = "pass"
    failed = "fail"
    inconclusive = "inconclusive"


# ---------------------------------------------------------------- Finding


class NodeRef(_Strict):
    node_id: str
    component_type: str
    state: str


class Evidence(_Strict):
    metric: str
    value: float
    unit: str = ""
    threshold: Optional[float] = None
    timestamp: datetime


class Confidence(_Strict):
    score: float = Field(ge=0.0, le=1.0)
    method: ConfidenceMethod
    samples: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _samples_only_for_mc(self) -> "Confidence":
        if self.method == ConfidenceMethod.mc_dropout.value and not self.samples:
            raise ValueError("mc_dropout confidence must state its sample count")
        return self


class Finding(_Strict):
    """Emitted by SABLE. `summary` may be model-written; `summary_generated`
    says so — generated text is always flagged (ground rule 8)."""
    id: str = Field(default_factory=lambda: _id("fnd"))
    created_at: datetime = Field(default_factory=now_utc)
    site_id: str
    detection_mode: DetectionMode
    root_cause: NodeRef
    affected_nodes: list[NodeRef] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: Confidence
    severity: Severity
    summary: str = ""
    summary_generated: bool = False
    engine_version: str
    # additive
    status: FindingStatus = FindingStatus.open
    updated_at: Optional[datetime] = None
    occurrences: int = Field(default=1, ge=1)
    receipt_ids: list[str] = Field(default_factory=list)

    @property
    def dedup_key(self) -> tuple[str, str, str]:
        """A persisting condition is the same finding: same site, same root
        cause node, same root-cause state."""
        return (self.site_id, self.root_cause.node_id, self.root_cause.state)


# ---------------------------------------------------------------- Plan


class Compensation(_Strict):
    action_id: str
    params: dict[str, Any] = Field(default_factory=dict)


class Step(_Strict):
    step_id: str = Field(default_factory=lambda: _id("stp"))
    action_id: str                         # MUST exist in the catalog (validated in Phase 3)
    target_node: str
    params: dict[str, Any] = Field(default_factory=dict)
    reversibility: Reversibility
    compensation: Optional[Compensation] = None
    precondition: str                      # a check name from the catalog
    timeout_s: int = Field(gt=0)

    @model_validator(mode="after")
    def _compensable_needs_compensation(self) -> "Step":
        if self.reversibility == Reversibility.compensable.value and self.compensation is None:
            raise ValueError("a compensable step must name its compensation")
        return self


class BlastRadius(_Strict):
    nodes: list[str] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _count_matches(self) -> "BlastRadius":
        if self.count != len(self.nodes):
            raise ValueError("blast_radius.count must equal len(nodes)")
        return self


class Verification(_Strict):
    predicate: str                         # a catalog verification predicate
    window_s: int = Field(ge=0)


class Gate(_Strict):
    decision: GateDecision
    rule_id: str
    reason: str
    delay_s: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _delay_needs_seconds(self) -> "Gate":
        if self.decision == GateDecision.delay.value and self.delay_s is None:
            raise ValueError("a delay decision must carry delay_s")
        return self


class PlannerProvenance(_Strict):          # additive: which planner, and the model call if any
    kind: str                              # "template" | "llm"
    template_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_sha256: Optional[str] = None
    output_sha256: Optional[str] = None


class Plan(_Strict):
    id: str = Field(default_factory=lambda: _id("pln"))
    finding_id: str
    created_at: datetime = Field(default_factory=now_utc)
    steps: list[Step] = Field(min_length=1)
    blast_radius: BlastRadius
    verification: Verification
    gate: Optional[Gate] = None
    # additive
    planner: Optional[PlannerProvenance] = None

    @property
    def reversibility(self) -> Reversibility:
        """The plan's overall reversibility is its WORST step."""
        return max((Reversibility(s.reversibility) for s in self.steps), key=lambda r: r.rank)

    @field_validator("steps")
    @classmethod
    def _unique_step_ids(cls, steps: list[Step]) -> list[Step]:
        ids = [s.step_id for s in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step_id values must be unique within a plan")
        return steps


# ---------------------------------------------------------------- Receipt


class Approval(_Strict):
    actor: str
    decision: str                          # "approve" | "reject" | "hold" | "execute_now"
    timestamp: datetime = Field(default_factory=now_utc)

    @field_validator("decision")
    @classmethod
    def _known_decision(cls, v: str) -> str:
        if v not in ("approve", "reject", "hold", "execute_now"):
            raise ValueError("decision must be approve | reject | hold | execute_now")
        return v


class StepResult(_Strict):
    step_id: str
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    status: StepStatus
    output_digest: Optional[str] = None    # sha256 of the step's output
    session_id: Optional[str] = None       # additive: the OVERLORD session that ran it
    error: Optional[str] = None            # additive


class Snapshot(_Strict):
    node_id: str
    snapshot_ref: str                      # an OVERLORD session id, or a catalog-defined ref


class VerificationResult(_Strict):
    status: VerificationStatus
    observed: dict[str, Any] = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=now_utc)


class RollbackStep(_Strict):
    step_id: str                           # the step being compensated
    action_id: str                         # the compensation action run
    status: StepStatus
    session_id: Optional[str] = None
    error: Optional[str] = None


class Rollback(_Strict):
    performed: bool = False
    steps: list[RollbackStep] = Field(default_factory=list)


class Receipt(_Strict):
    """Produced by the executor after execution and verification. The hash
    chain: `receipt_hash` is sha256 over the canonical JSON of every field
    but itself and `audit_ref`, including `prev_receipt_hash` — so a changed
    receipt, or a removed one, breaks every hash after it. `audit_ref` is
    excluded because it points at OVERLORD's audit entry, which itself
    commits to `receipt_hash` (the two chains reference each other; one
    direction must be outside the hash)."""
    id: str = Field(default_factory=lambda: _id("rcp"))
    plan_id: str
    session_id: Optional[str] = None       # the OVERLORD session (first step's, or the plan's)
    approvals: list[Approval] = Field(default_factory=list)
    steps: list[StepResult] = Field(default_factory=list)
    snapshots: list[Snapshot] = Field(default_factory=list)
    verification: Optional[VerificationResult] = None
    rollback: Rollback = Field(default_factory=Rollback)
    prev_receipt_hash: Optional[str] = None
    receipt_hash: Optional[str] = None
    # additive
    finding_id: Optional[str] = None
    created_at: datetime = Field(default_factory=now_utc)
    audit_ref: Optional[dict[str, Any]] = None   # OVERLORD audit chain {seq, hash} of the closing entry

    HASH_EXCLUDES: ClassVar[frozenset[str]] = frozenset({"receipt_hash", "audit_ref"})

    def compute_hash(self) -> str:
        data = self.model_dump(mode="json", exclude=set(self.HASH_EXCLUDES))
        return hashlib.sha256(canonical_json(data).encode()).hexdigest()

    def seal(self, prev_receipt_hash: Optional[str] = None) -> "Receipt":
        """Set prev and compute this receipt's hash. Returns self."""
        if prev_receipt_hash is not None:
            self.prev_receipt_hash = prev_receipt_hash
        self.receipt_hash = self.compute_hash()
        return self

    def verify_hash(self) -> bool:
        return self.receipt_hash is not None and self.receipt_hash == self.compute_hash()


def verify_chain(receipts: list[Receipt]) -> tuple[bool, Optional[int]]:
    """Walk receipts in order: each hash must recompute, and each prev must
    equal its predecessor's hash. Returns (ok, index_of_first_break)."""
    prev = None
    for i, r in enumerate(receipts):
        if not r.verify_hash() or r.prev_receipt_hash != prev:
            return False, i
        prev = r.receipt_hash
    return True, None


# ---------------------------------------------------------------- schema export


CONTRACTS = {"finding": Finding, "plan": Plan, "receipt": Receipt}


def export_schemas(out_dir: str | Path) -> dict[str, Path]:
    """Write <name>.schema.json for each contract. Returns {name: path}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, model in CONTRACTS.items():
        p = out / f"{name}.schema.json"
        p.write_text(json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n")
        written[name] = p
    return written
