"""Phase 6 acceptance: every cell of the target matrix, the band boundaries,
the unmonitored-gap cap, the hard rule, the shipped default-deny, the delay
countdown, `apply` purity, and invalid policy files."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from console.contracts import (
    BlastRadius, Compensation, Confidence, Finding, Gate, NodeRef, Plan, Reversibility, Step,
    Verification,
)
from console.policy import (
    DEFAULT_DENY_PREFIX, HARD_RULE_IRREVERSIBLE, Policy, PolicyError, apply, decide, load_policy,
)
from console.policy.gate import DEFAULT_POLICY_PATH, TARGET_POLICY_PATH

T0 = datetime(2026, 9, 22, 14, 0, 0, tzinfo=timezone.utc)
BANDS = ("high", "medium", "low")
REVS = ("reversible", "compensable", "irreversible")

# a representative score per band under the shipped thresholds (high 0.85, medium 0.60)
SCORE = {"high": 0.91, "medium": 0.70, "low": 0.40}

TARGET = {
    "high":   {"reversible": "auto",        "compensable": "delay",       "irreversible": "human"},
    "medium": {"reversible": "human",       "compensable": "human",       "irreversible": "human_plus"},
    "low":    {"reversible": "report_only", "compensable": "report_only", "irreversible": "report_only"},
}


# ---------------------------------------------------------------- builders


def mk_finding(score: float, mode: str = "live_feed") -> Finding:
    return Finding(
        id="fnd-gate00000001", created_at=T0, site_id="lab-1", detection_mode=mode,
        root_cause=NodeRef(node_id="dns-1", component_type="DNS_SERVER", state="failed"),
        confidence=Confidence(score=score, method="engine_native"),
        severity="high", engine_version="sable@test",
    )


def _step(step_id: str, reversibility: str) -> Step:
    comp = Compensation(action_id="restart_service", params={"service": "dnsmasq"}) \
        if reversibility == "compensable" else None
    return Step(step_id=step_id, action_id="restart_service", target_node="dns-1",
                reversibility=reversibility, compensation=comp,
                precondition="service_exists", timeout_s=60)


def mk_plan(worst: str) -> Plan:
    """A plan whose WORST step has reversibility `worst`. Every plan also
    carries a reversible step, so the property really is taking the max."""
    steps = [_step("s1", "reversible")]
    if worst != "reversible":
        steps.append(_step("s2", worst))
    return Plan(id="pln-gate00000001", finding_id="fnd-gate00000001", created_at=T0,
                steps=steps, blast_radius=BlastRadius(nodes=["dns-1"], count=1),
                verification=Verification(predicate="node_healthy", window_s=30))


def write_policy(tmp_path: Path, **overrides) -> Path:
    """The target policy as YAML with `overrides` spliced in (dotted keys)."""
    import yaml
    raw = yaml.safe_load(TARGET_POLICY_PATH.read_text())
    for dotted, value in overrides.items():
        cur = raw
        parts = dotted.split(".")
        for k in parts[:-1]:
            cur = cur[k]
        if value is ...:
            del cur[parts[-1]]
        else:
            cur[parts[-1]] = value
    p = tmp_path / "policy.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


@pytest.fixture(scope="module")
def target() -> Policy:
    return Policy.target()


@pytest.fixture(scope="module")
def default() -> Policy:
    return Policy.default()


# ---------------------------------------------------------------- the target matrix, every cell


@pytest.mark.parametrize("band", BANDS)
@pytest.mark.parametrize("rev", REVS)
def test_target_matrix_cell(target, band, rev):
    plan = mk_plan(rev)
    assert plan.reversibility == Reversibility(rev)
    gate = target.decide(mk_finding(SCORE[band]), plan)
    assert gate.decision == TARGET[band][rev]
    assert gate.rule_id == f"matrix:{band}:{rev}"
    assert not gate.rule_id.startswith(DEFAULT_DENY_PREFIX)
    assert band in gate.reason and rev in gate.reason
    assert (gate.delay_s == 120) if gate.decision == "delay" else (gate.delay_s is None)


# ---------------------------------------------------------------- band boundaries


@pytest.mark.parametrize("score, band", [
    (1.00, "high"), (0.85, "high"),                       # exactly at threshold counts as the band
    (0.8499, "medium"), (0.60, "medium"),
    (0.5999, "low"), (0.00, "low"),
])
def test_band_boundaries(target, default, score, band):
    assert target.band_for(score) == band
    assert default.band_for(score) == band


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_band_rejects_out_of_range(target, score):
    with pytest.raises(ValueError):
        target.band_for(score)


def test_band_thresholds_are_configurable(tmp_path):
    p = Policy.load(write_policy(tmp_path, **{"bands.high": 0.95, "bands.medium": 0.90}))
    assert p.band_for(0.94) == "medium"
    assert p.band_for(0.89) == "low"


# ---------------------------------------------------------------- unmonitored-gap cap


@pytest.mark.parametrize("score_band, capped_band", [
    ("high", "medium"), ("medium", "low"), ("low", "low"),
])
@pytest.mark.parametrize("rev", REVS)
def test_unmonitored_gap_cap(target, score_band, capped_band, rev):
    gate = target.decide(mk_finding(SCORE[score_band], mode="unmonitored_gap"), mk_plan(rev))
    assert gate.decision == TARGET[capped_band][rev]
    if capped_band != score_band:
        assert gate.rule_id == f"cap:unmonitored_gap→{capped_band}"
        assert "unmonitored gap" in gate.reason and "capped" in gate.reason
    else:                                                      # low stays low: nothing was capped
        assert gate.rule_id == f"matrix:low:{rev}"
        assert "unmonitored gap" in gate.reason


def test_cap_never_lifts_a_decision(target):
    """A gap finding can only ever get LESS autonomy than a live one."""
    order = ["auto", "delay", "human", "human_plus", "report_only"]
    for band in BANDS:
        for rev in REVS:
            live = target.decide(mk_finding(SCORE[band]), mk_plan(rev)).decision
            gap = target.decide(mk_finding(SCORE[band], "unmonitored_gap"), mk_plan(rev)).decision
            assert order.index(gap) >= order.index(live)


def test_cap_of_two_bands(tmp_path):
    p = Policy.load(write_policy(tmp_path, unmonitored_gap_cap=2))
    gate = p.decide(mk_finding(SCORE["high"], "unmonitored_gap"), mk_plan("reversible"))
    assert gate.decision == "report_only"
    assert gate.rule_id == "cap:unmonitored_gap→low"


# ---------------------------------------------------------------- hard rule


def test_hard_rule_irreversible_never_auto(tmp_path):
    """The target matrix already says human for high/irreversible, so build a
    policy that says `auto` there: the gate must still refuse."""
    p = Policy.load(write_policy(tmp_path, **{"matrix.high.irreversible": "auto"}))
    assert p.matrix.high.irreversible == "auto"
    gate = p.decide(mk_finding(SCORE["high"]), mk_plan("irreversible"))
    assert gate.decision == "human"
    assert gate.rule_id == HARD_RULE_IRREVERSIBLE
    assert "irreversible" in gate.reason and "never" in gate.reason
    # the same policy still grants auto to a reversible plan — the rule is about the plan, not the band
    assert p.decide(mk_finding(SCORE["high"]), mk_plan("reversible")).decision == "auto"


def test_hard_rule_cannot_be_disabled(tmp_path):
    with pytest.raises(PolicyError, match="irreversible_never_auto"):
        Policy.load(write_policy(tmp_path, **{"hard_rules.irreversible_never_auto": False}))


def test_hard_rule_beats_the_cap_too(tmp_path):
    """auto for irreversible in the medium row, reached via the cap: still refused."""
    p = Policy.load(write_policy(tmp_path, **{"matrix.medium.irreversible": "auto"}))
    gate = p.decide(mk_finding(SCORE["high"], "unmonitored_gap"), mk_plan("irreversible"))
    assert gate.decision == "human"
    assert gate.rule_id == HARD_RULE_IRREVERSIBLE


# ---------------------------------------------------------------- shipped default: default-deny


def test_shipped_default_is_the_default_path(default):
    assert Policy.load() == default
    assert load_policy() == default
    assert Policy.load(DEFAULT_POLICY_PATH) == default
    assert default.is_default_deny
    assert not Policy.target().is_default_deny


@pytest.mark.parametrize("band", BANDS)
@pytest.mark.parametrize("rev", REVS)
@pytest.mark.parametrize("mode", ["live_feed", "unmonitored_gap"])
def test_shipped_default_every_cell_is_human(default, band, rev, mode):
    gate = default.decide(mk_finding(SCORE[band], mode), mk_plan(rev))
    assert gate.decision == "human"
    assert gate.rule_id.startswith(DEFAULT_DENY_PREFIX)
    assert gate.delay_s is None
    assert "shipped default" in gate.reason
    # the module-level conveniences use the shipped default when no policy is given
    assert decide(mk_finding(SCORE[band], mode), mk_plan(rev)) == gate


def test_shipped_default_rule_id_still_names_the_cell(default):
    gate = default.decide(mk_finding(SCORE["high"]), mk_plan("compensable"))
    assert gate.rule_id == "default-deny:matrix:high:compensable"
    gate = default.decide(mk_finding(SCORE["high"], "unmonitored_gap"), mk_plan("compensable"))
    assert gate.rule_id == "default-deny:cap:unmonitored_gap→medium"


# ---------------------------------------------------------------- delay


def test_delay_carries_delay_s(target, tmp_path):
    gate = target.decide(mk_finding(SCORE["high"]), mk_plan("compensable"))
    assert gate.decision == "delay" and gate.delay_s == 120
    assert "120s" in gate.reason and "countdown" in gate.reason
    p = Policy.load(write_policy(tmp_path, delay_s=45))
    assert p.decide(mk_finding(SCORE["high"]), mk_plan("compensable")).delay_s == 45


def test_non_delay_decisions_carry_no_delay(target):
    for band in BANDS:
        for rev in REVS:
            gate = target.decide(mk_finding(SCORE[band]), mk_plan(rev))
            if gate.decision != "delay":
                assert gate.delay_s is None


# ---------------------------------------------------------------- apply


def test_apply_returns_new_plan_and_leaves_input_untouched(target):
    finding = mk_finding(SCORE["high"])
    plan = mk_plan("reversible")
    before = plan.model_dump(mode="json")
    gated = target.apply(finding, plan)
    assert gated is not plan
    assert plan.gate is None
    assert plan.model_dump(mode="json") == before
    assert isinstance(gated.gate, Gate)
    assert gated.gate == target.decide(finding, plan)
    assert gated.gate.rule_id == "matrix:high:reversible"
    # everything but the gate is identical
    assert gated.model_dump(mode="json", exclude={"gate"}) == {k: v for k, v in before.items() if k != "gate"}
    # the gated plan still validates as a Plan (the gate round-trips)
    assert Plan.model_validate(gated.model_dump(mode="json")) == gated
    # module-level apply uses the shipped default
    assert apply(finding, plan).gate.rule_id.startswith(DEFAULT_DENY_PREFIX)
    assert plan.gate is None


def test_apply_replaces_an_existing_gate(target):
    finding = mk_finding(SCORE["low"])
    plan = mk_plan("reversible")
    stale = Gate(decision="auto", rule_id="stale", reason="a previous evaluation")
    plan = plan.model_copy(update={"gate": stale})
    gated = target.apply(finding, plan)
    assert plan.gate == stale
    assert gated.gate.decision == "report_only"


# ---------------------------------------------------------------- invalid policy files


@pytest.mark.parametrize("overrides, match", [
    ({"matrix.medium.compensable": ...}, "matrix.medium.compensable"),            # missing cell
    ({"matrix.low": ...}, "matrix.low"),                                           # missing row
    ({"matrix.high.reversible": "yolo"}, "matrix.high.reversible"),                # bad decision word
    ({"matrix.high.reversible": "automatic"}, "matrix.high.reversible"),
    ({"bands.high": 0.50, "bands.medium": 0.70}, "high > medium"),                 # thresholds inverted
    ({"bands.high": 0.70, "bands.medium": 0.70}, "high > medium"),                 # equal is not ordered
    ({"bands.high": 1.5}, "bands.high"),                                           # out of [0, 1]
    ({"bands.medium": ...}, "bands.medium"),                                       # missing threshold
    ({"delay_s": -1}, "delay_s"),
    ({"unmonitored_gap_cap": 3}, "unmonitored_gap_cap"),                           # more bands than exist
    ({"unmonitored_gap_cap": -1}, "unmonitored_gap_cap"),
    ({"version": 2}, "version"),
    ({"matrix.high.sideways": "human"}, "sideways"),                               # unknown key (extra=forbid)
    ({"surprise": 1}, "surprise"),
    ({"hard_rules.irreversible_never_auto": False}, "cannot be disabled"),
])
def test_invalid_policy_fails_to_load(tmp_path, overrides, match):
    path = write_policy(tmp_path, **overrides)
    with pytest.raises(PolicyError, match=match) as ei:
        Policy.load(path)
    assert str(path) in str(ei.value)                    # the error names the file


def test_missing_and_malformed_policy_files(tmp_path):
    with pytest.raises(PolicyError, match="cannot read"):
        Policy.load(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("matrix: [unclosed\n")
    with pytest.raises(PolicyError, match="not valid YAML"):
        Policy.load(bad)
    lst = tmp_path / "list.yaml"
    lst.write_text("- 1\n- 2\n")
    with pytest.raises(PolicyError, match="mapping"):
        Policy.load(lst)


def test_policy_error_is_a_value_error():
    assert issubclass(PolicyError, ValueError)


def test_policy_is_immutable(target):
    with pytest.raises(Exception):
        target.delay_s = 1            # frozen: a loaded policy cannot drift during a run
