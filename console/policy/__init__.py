"""The autonomy gate (PLAN.md Phase 6): confidence band x plan reversibility
-> decision, from console/policy/policy.yaml.

    Policy          the loaded matrix; `.decide(finding, plan) -> Gate`,
                    `.apply(finding, plan) -> Plan` (copy, with gate set)
    load_policy     Policy.load(path) — the shipped all-`human` default when path is None
    decide / apply  module-level conveniences using the shipped default policy
    PolicyError     a policy file that cannot be loaded or is invalid

Shipped default: every cell is `human`. The plan's target matrix is
policy.target.yaml; an operator opts into it.
"""

from .gate import (  # noqa: F401
    BANDS,
    DEFAULT_DENY_PREFIX,
    DEFAULT_POLICY_PATH,
    HARD_RULE_IRREVERSIBLE,
    TARGET_POLICY_PATH,
    Bands,
    HardRules,
    Matrix,
    MatrixRow,
    Policy,
    PolicyError,
    apply,
    decide,
    load_policy,
)

__all__ = [
    "BANDS", "DEFAULT_DENY_PREFIX", "DEFAULT_POLICY_PATH", "HARD_RULE_IRREVERSIBLE",
    "TARGET_POLICY_PATH", "Bands", "HardRules", "Matrix", "MatrixRow", "Policy",
    "PolicyError", "apply", "decide", "load_policy",
]
