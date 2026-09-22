"""The autonomy gate (PLAN.md Phase 6).

A Policy is a matrix of confidence band x plan reversibility -> decision,
loaded from YAML. `decide(finding, plan)` turns a Finding and a Plan into a
Gate; `apply` writes that Gate into a copy of the Plan.

Algorithm
  1. band  = band_for(finding.confidence.score)           (>= high, >= medium, else low)
  2. if finding.detection_mode == unmonitored_gap:
         band = band lowered by `unmonitored_gap_cap`     (floored at low)
  3. decision = matrix[band][plan.reversibility]          (reversibility = the plan's WORST step)
  4. hard rule: irreversible plan and decision == auto -> human
  5. delay decisions carry `delay_s`
  6. rule_id names the rule that produced the decision (hard > cap > matrix);
     when the policy is the shipped all-`human` default it is prefixed
     `default-deny:` so a reader can tell "nobody configured this" from
     "the operator chose human here".

The shipped default (policy.yaml) is every cell `human`. The plan's target
matrix is policy.target.yaml; the operator opts into it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from console.contracts.models import (
    DetectionMode,
    Finding,
    Gate,
    GateDecision,
    Plan,
    Reversibility,
)

HERE = Path(__file__).parent
DEFAULT_POLICY_PATH = HERE / "policy.yaml"
TARGET_POLICY_PATH = HERE / "policy.target.yaml"

Band = Literal["high", "medium", "low"]
BANDS: tuple[Band, ...] = ("high", "medium", "low")          # best -> worst
REVERSIBILITIES: tuple[str, ...] = tuple(r.value for r in Reversibility)

HARD_RULE_IRREVERSIBLE = "hard:irreversible_never_auto"
DEFAULT_DENY_PREFIX = "default-deny:"


class PolicyError(ValueError):
    """A policy file that cannot be loaded or does not describe a valid policy."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True, frozen=True)


class Bands(_Strict):
    """Score thresholds. score >= high -> high; score >= medium -> medium; else low."""
    high: float = Field(ge=0.0, le=1.0)
    medium: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> "Bands":
        if not self.high > self.medium:
            raise ValueError(
                f"bands must be ordered high > medium (got high={self.high}, medium={self.medium})")
        return self


class MatrixRow(_Strict):
    """One confidence band's decisions, one per plan reversibility. Every
    key is required: a missing cell is a policy error, not an implicit deny."""
    reversible: GateDecision
    compensable: GateDecision
    irreversible: GateDecision

    def cell(self, reversibility: Union[str, Reversibility]) -> str:
        key = reversibility.value if isinstance(reversibility, Reversibility) else str(reversibility)
        if key not in REVERSIBILITIES:
            raise PolicyError(f"unknown reversibility {key!r}")
        return getattr(self, key)


class Matrix(_Strict):
    high: MatrixRow
    medium: MatrixRow
    low: MatrixRow

    def row(self, band: str) -> MatrixRow:
        if band not in BANDS:
            raise PolicyError(f"unknown confidence band {band!r}")
        return getattr(self, band)


class HardRules(_Strict):
    """Rules the matrix cannot override. They are listed so the policy file
    says them out loud, but they cannot be switched off — a hard rule that
    can be disabled is not hard."""
    irreversible_never_auto: bool = True

    @model_validator(mode="after")
    def _cannot_disable(self) -> "HardRules":
        if not self.irreversible_never_auto:
            raise ValueError("hard_rules.irreversible_never_auto cannot be disabled")
        return self


class Policy(_Strict):
    version: Literal[1] = 1
    bands: Bands
    matrix: Matrix
    delay_s: int = Field(ge=0)
    unmonitored_gap_cap: int = Field(ge=0, le=len(BANDS) - 1)
    hard_rules: HardRules = Field(default_factory=HardRules)

    # ------------------------------------------------------------ loading

    @classmethod
    def load(cls, path: Union[str, Path, None] = None) -> "Policy":
        """Load a policy YAML file; `path=None` loads the shipped default.
        Raises PolicyError with the file name and every validation error."""
        p = Path(path) if path is not None else DEFAULT_POLICY_PATH
        try:
            text = p.read_text()
        except OSError as e:
            raise PolicyError(f"cannot read policy file {p}: {e}") from e
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise PolicyError(f"policy file {p} is not valid YAML: {e}") from e
        if not isinstance(raw, dict):
            raise PolicyError(f"policy file {p} must be a mapping at top level, got {type(raw).__name__}")
        try:
            return cls.model_validate(raw)
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(str(x) for x in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors())
            raise PolicyError(f"invalid policy file {p}: {problems}") from e

    @classmethod
    def default(cls) -> "Policy":
        """The shipped default: every cell `human`."""
        return cls.load(DEFAULT_POLICY_PATH)

    @classmethod
    def target(cls) -> "Policy":
        """The plan's target matrix (policy.target.yaml) the operator opts into."""
        return cls.load(TARGET_POLICY_PATH)

    # ------------------------------------------------------------ queries

    @property
    def is_default_deny(self) -> bool:
        """True when every cell is `human` — i.e. nobody has configured autonomy."""
        return all(
            self.matrix.row(b).cell(r) == GateDecision.human.value
            for b in BANDS for r in REVERSIBILITIES)

    def band_for(self, score: float) -> Band:
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"confidence score must be within [0, 1], got {score}")
        if score >= self.bands.high:
            return "high"
        if score >= self.bands.medium:
            return "medium"
        return "low"

    def lower_band(self, band: Band, by: int) -> Band:
        """`band` lowered `by` bands, floored at low."""
        return BANDS[min(BANDS.index(band) + by, len(BANDS) - 1)]

    # ------------------------------------------------------------ decision

    def decide(self, finding: Finding, plan: Plan) -> Gate:
        score = finding.confidence.score
        score_band = self.band_for(score)
        reversibility = plan.reversibility.value
        reason = f"Confidence {score:.2f} is in the {score_band} band"

        # 2. unmonitored-gap cap
        band = score_band
        capped = False
        if finding.detection_mode == DetectionMode.unmonitored_gap.value and self.unmonitored_gap_cap > 0:
            band = self.lower_band(score_band, self.unmonitored_gap_cap)
            capped = band != score_band
            if capped:
                reason += (", but the finding rests on an unmonitored gap (absence of telemetry), "
                           f"so it is capped to {band}")
            else:
                reason += ", and the finding rests on an unmonitored gap (low is already the floor)"

        # 3. matrix lookup
        decision = self.matrix.row(band).cell(reversibility)
        rule_id = f"cap:unmonitored_gap\u2192{band}" if capped else f"matrix:{band}:{reversibility}"
        article = "an" if reversibility[0] in "aeiou" else "a"
        reason += f"; with {article} {reversibility} plan the matrix says {decision}"

        # 4. hard rule
        if (reversibility == Reversibility.irreversible.value
                and decision == GateDecision.auto.value
                and self.hard_rules.irreversible_never_auto):
            decision = GateDecision.human.value
            rule_id = HARD_RULE_IRREVERSIBLE
            reason += ", but an irreversible plan can never run automatically, so it needs a human"

        # 5. delay countdown
        delay_s: Optional[int] = self.delay_s if decision == GateDecision.delay.value else None
        if delay_s is not None:
            reason += f" \u2014 a {delay_s}s countdown the operator can execute now, hold, or reject"

        # 6. default-deny marker
        if self.is_default_deny:
            rule_id = DEFAULT_DENY_PREFIX + rule_id
            reason += " (shipped default policy: every cell requires a human)"

        return Gate(decision=decision, rule_id=rule_id, reason=reason + ".", delay_s=delay_s)

    def apply(self, finding: Finding, plan: Plan) -> Plan:
        """A copy of `plan` with `gate` set; the input is left untouched."""
        gate = self.decide(finding, plan)
        return plan.model_copy(update={"gate": gate}, deep=True)



# ---------------------------------------------------------------- module API

_default_policy: Optional[Policy] = None


def load_policy(path: Union[str, Path, None] = None) -> Policy:
    """Load a policy file (the shipped default when `path` is None)."""
    return Policy.load(path)


def _policy(policy: Optional[Policy]) -> Policy:
    global _default_policy
    if policy is not None:
        return policy
    if _default_policy is None:
        _default_policy = Policy.default()
    return _default_policy


def decide(finding: Finding, plan: Plan, policy: Optional[Policy] = None) -> Gate:
    """Gate decision under `policy` (the shipped default when omitted)."""
    return _policy(policy).decide(finding, plan)


def apply(finding: Finding, plan: Plan, policy: Optional[Policy] = None) -> Plan:
    """A copy of `plan` with its gate written, under `policy` (default when omitted)."""
    return _policy(policy).apply(finding, plan)
