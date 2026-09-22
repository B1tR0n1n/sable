"""PROPOSE: Findings become Plans, using only catalog actions (PLAN.md Phase 3).

    TemplatePlanner   deterministic (component_type, state) -> step template; ships first
    LLMPlanner        a model picks and parameterises catalog actions through one
                      tool-less completion; output is validated, never repaired
    validate_plan     the gate both go through: catalog + topology + contract
"""

from console.catalog import Catalog, CatalogError  # noqa: F401

from .llm import LLMPlanner  # noqa: F401
from .template import NoTemplate, TemplatePlanner  # noqa: F401
from .validate import PlanValidationError, expected_blast_radius, validate_plan  # noqa: F401

__all__ = ["Catalog", "CatalogError", "TemplatePlanner", "NoTemplate", "LLMPlanner",
           "validate_plan", "PlanValidationError", "expected_blast_radius"]
