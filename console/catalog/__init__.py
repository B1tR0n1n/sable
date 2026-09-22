"""The action catalog: the closed set of things a Plan may ask OVERLORD to do.

Ground rule 4 — the LLM never invents actions. A planner picks an `action_id`
from `actions.yaml` and fills its params; the catalog fixes everything else
on a Step (reversibility, compensation, preconditions, verification, the
executor binding). `console.planner.validate_plan` re-checks a Plan against
this module; nothing is repaired, invalid input is rejected with reasons.

    cat = Catalog.load()                              # console/catalog/actions.yaml
    cat.actions_for("DNS_SERVER")                     # proposable ActionSpecs
    cat.validate_params("restart_service", {...})     # CatalogError on violation
    cat.render("restart_service", params, context)    # {argv, then, grants, target_dir}
    cat.compensation_for(step)                        # Compensation | None
    cat.render_check("service_exists", params, context)

Templates use `{name}` placeholders. Names resolve from, in this order of
ownership: the executor's context (`lab_dir`, `lab_compose`; declared under
`context:` in the YAML), the catalog's own `scripts_dir`, and the step's
params. Load-time validation guarantees every placeholder in a binding is
one of those, and that no param shadows a context name.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from console.contracts import Compensation, Reversibility, Step

try:  # SABLE's vocabulary (adapters/base.py is stdlib-only); a copy if unavailable
    from adapters.base import COMPONENT_TYPES as KNOWN_COMPONENT_TYPES
except Exception:  # pragma: no cover
    KNOWN_COMPONENT_TYPES = [
        "CORE_SWITCH", "ACCESS_SWITCH", "FIREWALL", "ROUTER", "LOAD_BALANCER",
        "SERVER_PHYSICAL", "SERVER_VIRTUAL", "HYPERVISOR", "STORAGE_ARRAY",
        "STORAGE_TARGET", "VDI_BROKER", "VDI_HOST", "DNS_SERVER", "DHCP_SERVER",
        "DOMAIN_CONTROLLER", "CERTIFICATE_AUTHORITY", "MONITORING_SERVER",
        "WAN_LINK", "INTERNET_GATEWAY", "APPLICATION_SERVICE",
    ]

HERE = Path(__file__).resolve().parent
DEFAULT_PATH = HERE / "actions.yaml"
SCRIPTS_DIR = HERE / "scripts"
CATALOG_CONTEXT_KEYS = ("scripts_dir",)          # filled by the catalog itself, never by a planner

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_BRACE = re.compile(r"[{}]")


class CatalogError(ValueError):
    """The catalog is malformed, or a request against it is invalid."""


# ---------------------------------------------------------------- minimal JSON Schema


_JSON_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "integer": int, "number": (int, float), "null": type(None),
}


def _type_ok(value: Any, typ: str) -> bool:
    if typ in ("integer", "number") and isinstance(value, bool):
        return False                              # bool is an int in Python, not in JSON Schema
    py = _JSON_TYPES.get(typ)
    if py is None:
        raise CatalogError(f"unsupported schema type {typ!r}")
    return isinstance(value, py)


def check_schema(value: Any, schema: dict[str, Any], path: str = "params") -> list[str]:
    """A deliberately small JSON-Schema checker: type, required, properties,
    additionalProperties (bool), enum, pattern, items, minLength/maxLength,
    minimum/maximum. Returns a list of violations (empty = valid). Anything
    else in a schema is ignored — keep catalog schemas within this subset."""
    out: list[str] = []
    typ = schema.get("type")
    if typ is not None:
        types = typ if isinstance(typ, list) else [typ]
        if not any(_type_ok(value, t) for t in types):
            out.append(f"{path}: expected {'|'.join(types)}, got {type(value).__name__}")
            return out                            # further checks assume the type
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{path}: {value!r} is not one of {schema['enum']!r}")
    if isinstance(value, str):
        pat = schema.get("pattern")
        if pat is not None and re.search(pat, value) is None:
            out.append(f"{path}: {value!r} does not match pattern {pat!r}")
        if "minLength" in schema and len(value) < schema["minLength"]:
            out.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            out.append(f"{path}: longer than maxLength {schema['maxLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            out.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            out.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for req in schema.get("required") or []:
            if req not in value:
                out.append(f"{path}: missing required property {req!r}")
        for k, v in value.items():
            if k in props:
                out.extend(check_schema(v, props[k], f"{path}.{k}"))
            elif schema.get("additionalProperties", True) is False:
                out.append(f"{path}: unexpected property {k!r} (allowed: {sorted(props)})")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            out.extend(check_schema(item, schema["items"], f"{path}[{i}]"))
    return out


# ---------------------------------------------------------------- models


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class ExpectSpec(_Strict):
    exit_code: int = 0
    stdout_contains: Optional[str] = None


class CheckSpec(_Strict):
    """A read-only precondition, run through OVERLORD before the step."""
    description: str
    argv: list[str] = Field(min_length=1)
    expect: ExpectSpec = Field(default_factory=ExpectSpec)

    def placeholders(self) -> set[str]:
        names = _placeholders_in(self.argv)
        if self.expect.stdout_contains:
            names |= _placeholders_in([self.expect.stdout_contains])
        return names


class PredicateSpec(_Strict):
    """A verification predicate SABLE evaluates after execution (Phase 5)."""
    description: str


class Grants(_Strict):
    """The OVERLORD grants the executor requests for the step (G8). The
    daemon's policy is the ceiling; a step may ask for less, never more."""
    jail: bool = True
    net: Literal["none", "proxy", "host"] = "none"
    net_allow: list[str] = Field(default_factory=list)
    timeout_s: int = Field(gt=0)

    @model_validator(mode="after")
    def _allowlist_only_with_proxy(self) -> "Grants":
        if self.net_allow and self.net != "proxy":
            raise ValueError("net_allow only applies to net=proxy")
        return self


class ExecutorBinding(_Strict):
    argv: list[str] = Field(min_length=1)
    then: Optional[list[str]] = None      # optional follow-up command, same session
    grants: Grants
    target_dir: str = "{lab_dir}"

    def placeholders(self) -> set[str]:
        return _placeholders_in(self.argv) | _placeholders_in(self.then or []) | _placeholders_in([self.target_dir])


class CompensationSpec(_Strict):
    action_id: str
    params_from: list[str] = Field(default_factory=list)


class ActionSpec(_Strict):
    description: str
    component_types: list[str] = Field(min_length=1)
    params: dict[str, Any]                # JSON Schema (the subset check_schema supports)
    reversibility: Reversibility
    compensation: Optional[CompensationSpec] = None
    preconditions: list[str] = Field(min_length=1)
    verification: str
    verification_window_s: Optional[int] = Field(default=None, ge=0)   # None: the planner's default
    executor: ExecutorBinding
    compensation_only: bool = False
    action_id: str = ""                   # filled from the mapping key at load

    @property
    def param_names(self) -> set[str]:
        return set((self.params.get("properties") or {}).keys())

    @property
    def required_params(self) -> set[str]:
        return set(self.params.get("required") or [])

    @model_validator(mode="after")
    def _shape(self) -> "ActionSpec":
        unknown = [t for t in self.component_types if t not in KNOWN_COMPONENT_TYPES]
        if unknown:
            raise ValueError(f"unknown component types {unknown}")
        if self.params.get("type") != "object" or not isinstance(self.params.get("properties"), dict):
            raise ValueError("params must be a JSON Schema of type object with properties")
        if self.params.get("additionalProperties", True) is not False:
            raise ValueError("params must set additionalProperties: false")
        compensable = self.reversibility == Reversibility.compensable.value
        if compensable and self.compensation is None:
            raise ValueError("a compensable action must name its compensation")
        if not compensable and self.compensation is not None:
            raise ValueError("only a compensable action carries a compensation")
        if self.compensation is not None:
            missing = set(self.compensation.params_from) - self.param_names
            if missing:
                raise ValueError(f"compensation.params_from names unknown params {sorted(missing)}")
        return self


class CatalogSpec(_Strict):
    version: int
    context: list[str] = Field(default_factory=lambda: ["lab_dir", "lab_compose"])
    checks: dict[str, CheckSpec]
    predicates: dict[str, PredicateSpec]
    actions: dict[str, ActionSpec]

    @model_validator(mode="after")
    def _cross_references(self) -> "CatalogSpec":
        problems: list[str] = []
        ctx = set(self.context) | set(CATALOG_CONTEXT_KEYS)
        if len(set(self.context)) != len(self.context) or set(self.context) & set(CATALOG_CONTEXT_KEYS):
            problems.append("context names must be unique and must not include the catalog's own keys")
        for aid, a in self.actions.items():
            a.action_id = aid
            if a.param_names & ctx:
                problems.append(f"{aid}: params {sorted(a.param_names & ctx)} shadow context names")
            for c in a.preconditions:
                if c not in self.checks:
                    problems.append(f"{aid}: precondition {c!r} is not a known check")
                else:
                    extra = self.checks[c].placeholders() - ctx - a.param_names
                    if extra:
                        problems.append(f"{aid}: check {c!r} needs placeholders {sorted(extra)} the action cannot fill")
            if a.verification not in self.predicates:
                problems.append(f"{aid}: verification {a.verification!r} is not a known predicate")
            extra = a.executor.placeholders() - ctx - a.param_names
            if extra:
                problems.append(f"{aid}: executor needs placeholders {sorted(extra)} the action cannot fill")
            if a.compensation is not None:
                target = self.actions.get(a.compensation.action_id)
                if target is None:
                    problems.append(f"{aid}: compensation names unknown action {a.compensation.action_id!r}")
                else:
                    given = set(a.compensation.params_from)
                    if not target.required_params <= given:
                        problems.append(f"{aid}: compensation {target.action_id} requires "
                                        f"{sorted(target.required_params - given)} which params_from does not carry")
                    if not given <= target.param_names:
                        problems.append(f"{aid}: compensation params_from carries "
                                        f"{sorted(given - target.param_names)} which {target.action_id} does not accept")
        if problems:
            raise ValueError("; ".join(problems))
        return self


def _placeholders_in(parts: list[str]) -> set[str]:
    names: set[str] = set()
    for p in parts:
        if not isinstance(p, str):
            raise ValueError(f"template part {p!r} is not a string")
        stripped = _PLACEHOLDER.sub("", p)
        if _BRACE.search(stripped):
            raise ValueError(f"template part {p!r} has a brace that is not a {{name}} placeholder")
        names |= set(_PLACEHOLDER.findall(p))
    return names


# ---------------------------------------------------------------- the catalog


class Catalog:
    def __init__(self, spec: CatalogSpec, path: Path):
        self.spec = spec
        self.path = path
        self.scripts_dir = path.parent / "scripts"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Catalog":
        p = Path(path) if path else DEFAULT_PATH
        try:
            data = yaml.safe_load(p.read_text())
        except (OSError, yaml.YAMLError) as e:
            raise CatalogError(f"cannot read catalog {p}: {e}") from e
        if not isinstance(data, dict):
            raise CatalogError(f"catalog {p} is not a mapping")
        try:
            spec = CatalogSpec.model_validate(data)
        except ValidationError as e:
            raise CatalogError(f"catalog {p} is invalid: {e}") from e
        return cls(spec, p)

    # --- lookup

    @property
    def context_keys(self) -> list[str]:
        """The names the executor must supply to render()."""
        return list(self.spec.context)

    @property
    def actions(self) -> dict[str, ActionSpec]:
        return self.spec.actions

    @property
    def checks(self) -> dict[str, CheckSpec]:
        return self.spec.checks

    @property
    def predicates(self) -> dict[str, PredicateSpec]:
        return self.spec.predicates

    def get(self, action_id: str) -> ActionSpec:
        a = self.spec.actions.get(action_id)
        if a is None:
            raise CatalogError(f"unknown action {action_id!r} (known: {sorted(self.spec.actions)})")
        return a

    def has_action(self, action_id: str) -> bool:
        return action_id in self.spec.actions

    def actions_for(self, component_type: str) -> list[ActionSpec]:
        """Proposable actions for a component type: compensation_only ones
        are never proposed, only run as compensations."""
        return [a for a in self.spec.actions.values()
                if not a.compensation_only and component_type in a.component_types]

    # --- params

    def validate_params(self, action_id: str, params: dict[str, Any]) -> None:
        a = self.get(action_id)
        if not isinstance(params, dict):
            raise CatalogError(f"{action_id}: params must be an object")
        problems = check_schema(params, a.params)
        if problems:
            raise CatalogError(f"{action_id}: " + "; ".join(problems))

    def compensation_for_action(self, action_id: str, params: dict[str, Any]) -> Optional[Compensation]:
        """The compensation the catalog prescribes for `action_id` run with
        `params`, or None when the action has none (reversible/irreversible).
        Built from `params_from`, so it can be computed before the Step exists
        (the Step contract requires a compensable step to carry it)."""
        a = self.get(action_id)
        if a.compensation is None:
            return None
        missing = [k for k in a.compensation.params_from if k not in params]
        if missing:
            raise CatalogError(f"{action_id}: compensation needs params {missing} the step does not carry")
        return Compensation(action_id=a.compensation.action_id,
                            params={k: params[k] for k in a.compensation.params_from})

    def compensation_for(self, step: Step) -> Optional[Compensation]:
        """The compensation the catalog prescribes for this step (see above)."""
        return self.compensation_for_action(step.action_id, step.params)

    # --- rendering

    def _names(self, action: ActionSpec, params: dict[str, Any], context: dict[str, Any]) -> dict[str, str]:
        missing = [k for k in self.spec.context if k not in context]
        if missing:
            raise CatalogError(f"{action.action_id}: context is missing {missing}")
        names = {k: str(context[k]) for k in self.spec.context}
        names["scripts_dir"] = str(self.scripts_dir)
        for k, v in params.items():
            if isinstance(v, (dict, list)):
                raise CatalogError(f"{action.action_id}: param {k!r} is not a scalar")
            names[k] = str(v)
        return names

    def render(self, action_id: str, params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Fill the executor binding: {argv, then, grants, target_dir}. Params
        are validated first; an unresolved placeholder is a CatalogError."""
        a = self.get(action_id)
        self.validate_params(action_id, params)
        names = self._names(a, params, context)
        ex = a.executor
        return {
            "argv": _fill(ex.argv, names),
            "then": _fill(ex.then, names) if ex.then else None,
            "grants": ex.grants.model_dump(),
            "target_dir": _fill([ex.target_dir], names)[0],
        }

    def render_check(self, check_id: str, action_id: str, params: dict[str, Any],
                     context: dict[str, Any]) -> dict[str, Any]:
        """Fill a precondition check for a step: {argv, expect}."""
        c = self.spec.checks.get(check_id)
        if c is None:
            raise CatalogError(f"unknown check {check_id!r} (known: {sorted(self.spec.checks)})")
        a = self.get(action_id)
        self.validate_params(action_id, params)
        names = self._names(a, params, context)
        expect = c.expect.model_dump()
        if expect.get("stdout_contains"):
            expect["stdout_contains"] = _fill([expect["stdout_contains"]], names)[0]
        return {"argv": _fill(c.argv, names), "expect": expect}


def _fill(parts: list[str], names: dict[str, str]) -> list[str]:
    out = []
    for p in parts:
        def sub(m: re.Match) -> str:
            k = m.group(1)
            if k not in names:
                raise CatalogError(f"unknown placeholder {{{k}}} in {p!r}")
            return names[k]
        out.append(_PLACEHOLDER.sub(sub, p))
    return out


__all__ = [
    "ActionSpec", "Catalog", "CatalogError", "CatalogSpec", "CheckSpec", "CompensationSpec",
    "ExecutorBinding", "ExpectSpec", "Grants", "PredicateSpec", "check_schema",
    "DEFAULT_PATH", "SCRIPTS_DIR", "KNOWN_COMPONENT_TYPES",
]
