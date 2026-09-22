"""The three data contracts everything else speaks (PLAN.md, Phase 1).

    Finding   emitted by SABLE: what is wrong, with what confidence
    Plan      produced by the planner from the action catalog, annotated by the gate
    Receipt   produced after execution and verification, hash-chained

Pydantic v2, `extra="forbid"` throughout: an object with a field the contract
does not name is rejected, never repaired silently (ground rule 4). JSON
Schema for each is exported to contracts/schema/ by `export_schemas()`.
"""

from .models import (  # noqa: F401
    Approval,
    BlastRadius,
    Compensation,
    Confidence,
    ConfidenceMethod,
    DetectionMode,
    Evidence,
    Finding,
    FindingStatus,
    Gate,
    GateDecision,
    NodeRef,
    Plan,
    PlannerProvenance,
    Receipt,
    Reversibility,
    Rollback,
    RollbackStep,
    Severity,
    Snapshot,
    Step,
    StepResult,
    StepStatus,
    Verification,
    VerificationResult,
    VerificationStatus,
    canonical_json,
    export_schemas,
    now_utc,
)
