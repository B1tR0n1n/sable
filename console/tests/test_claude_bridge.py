"""Phase 2B — ClaudeBridge unit tests.

These run WITHOUT torch and WITHOUT docker/server.py: the bridge module is
imported directly from docker/, the Anthropic SDK client is monkeypatched with
a scripted fake, and the engine is a tiny stand-in exposing exactly what the
read-only tools touch (history, n_nodes, get_recommendations, get_node_report).

Documented `/api/nemotron/status` shape (docker/server.py, nemotron_status):
    Claude path:   {"available": bool, "url": "anthropic:<model>", "provider": "claude", "model": <model>}
    Nemotron path: {"available": bool, "url": "http://...:8081",   "provider": "nemotron"}
i.e. the pre-existing "available" and "url" keys are kept and "provider" is added
(plus "model" for Claude). `bridge_info()` below is what the route spreads in.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"
if str(DOCKER) not in sys.path:
    sys.path.insert(0, str(DOCKER))

import anthropic  # noqa: E402  (real SDK, so anthropic.APIError etc. are real types)

import claude_bridge  # noqa: E402
from claude_bridge import (  # noqa: E402
    DEFAULT_MODEL, TOOLS, UNAVAILABLE, ClaudeBridge, bridge_info, make_tools, select_bridge,
)
from nemotron_bridge import DEFAULT_LLAMA_URL, NemotronBridge  # noqa: E402


# ── fakes ─────────────────────────────────────────────────────────────────


def text(t: str):
    return SimpleNamespace(type="text", text=t)


def tool_use(id_: str, name: str, input_: dict):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def response(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


class FakeMessages:
    def __init__(self):
        self.calls: list[dict] = []          # every kwargs dict passed to messages.create

    def create(self, **kw):
        self.calls.append(copy.deepcopy(kw))
        if not FakeAnthropic.script:          # read live: tests script AFTER building the bridge
            raise AssertionError("fake Claude ran out of scripted responses")
        nxt = FakeAnthropic.script.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


class FakeAnthropic:
    """Stands in for anthropic.Anthropic. `script` is set per test via the fixture."""
    script: list = []
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = FakeMessages()
        FakeAnthropic.instances.append(self)


class FakeEngine:
    """Only what make_tools' accessors touch (see docker/sable_engine.py)."""

    def __init__(self, n_nodes: int = 3):
        self.n_nodes = n_nodes
        self.history = [
            {"cycle": 0, "predictions": [0, 0, 0], "ground_truth": None},
            {"cycle": 1, "predictions": [0, 2, 1], "ground_truth": None},
        ]
        self.device = "cpu"

    def get_recommendations(self) -> dict:
        return {
            "summary": "2/3 nodes affected: root cause node 1",
            "root_cause": 1,
            "total_affected": 2,
            "actions": [{"priority": 1, "action": "restart", "target": "db-01",
                         "target_type": "DATABASE", "reason": "degraded",
                         "recommendation": "restart the service"}],
        }

    def get_node_report(self, idx: int) -> dict:
        latest = self.history[-1]["predictions"][idx]   # IndexError if out of range
        return {"node_id": idx, "current_state": ["healthy", "warning", "degraded"][latest],
                "trajectory": [{"cycle": h["cycle"], "prediction": h["predictions"][idx]}
                               for h in self.history],
                "n_state_changes": 1}


TOPO_NODES = [
    {"id": "db-01", "type": "DATABASE", "label": "Primary DB", "tier": 1},
    {"id": "api-01", "type": "API_SERVER", "label": "API 1", "tier": 2},
    {"id": "web-01", "type": "WEB_SERVER", "label": "Web 1", "tier": 3},
]
TOPO_EDGES = [
    {"source": "api-01", "target": "db-01", "type": "DEPENDS_ON", "criticality": "HIGH"},
    {"source": "web-01", "target": "api-01", "type": "DEPENDS_ON", "criticality": "HIGH"},
]


def enrich(recs: dict) -> dict:
    """Mirror of server.enrich_recommendations: mutates+returns the fresh recs dict."""
    rc = recs.get("root_cause")
    if rc is not None:
        recs["root_cause_label"] = TOPO_NODES[rc]["label"]
        recs["root_cause_type"] = TOPO_NODES[rc]["type"]
    return recs


def label(i): return TOPO_NODES[i]["label"] if i < len(TOPO_NODES) else f"Node {i:02d}"
def nid(i): return TOPO_NODES[i]["id"] if i < len(TOPO_NODES) else f"node-{i:02d}"
def ntype(i): return TOPO_NODES[i]["type"] if i < len(TOPO_NODES) else "UNKNOWN"


def build_tools(engine):
    return make_tools(engine, enrich, label, nid, ntype, TOPO_NODES, TOPO_EDGES)


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def fake_claude(monkeypatch):
    """Monkeypatch anthropic.Anthropic and give the bridge a key. Returns a
    `script(*responses)` setter; the fake client records every create() call."""
    FakeAnthropic.script = []
    FakeAnthropic.instances = []
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.delenv("SABLE_CLAUDE_MODEL", raising=False)

    def script(*responses):
        FakeAnthropic.script = list(responses)
    return script


@pytest.fixture
def bridge(fake_claude):
    b = ClaudeBridge()
    assert b.is_available()
    return b


def last_client() -> FakeAnthropic:
    return FakeAnthropic.instances[-1]


# ── provider selection ────────────────────────────────────────────────────


def test_select_bridge_without_sable_llm_returns_nemotron_local_path():
    b = select_bridge(env={})                       # SABLE_LLM unset
    assert type(b) is NemotronBridge                 # not a subclass: identical local path
    assert b.llama_url == DEFAULT_LLAMA_URL
    assert bridge_info(b) == {"provider": "nemotron", "url": DEFAULT_LLAMA_URL}

    b2 = select_bridge(env={"SABLE_LLM": "nemotron", "ANTHROPIC_API_KEY": "sk-x"})
    assert type(b2) is NemotronBridge                # anything but "claude" -> local


def test_select_bridge_with_sable_llm_claude_returns_claude_bridge(monkeypatch):
    calls = []
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: calls.append(kw) or object())
    b = select_bridge(env={"SABLE_LLM": "claude"})   # no key -> no client constructed
    assert isinstance(b, ClaudeBridge) and isinstance(b, NemotronBridge)
    assert calls == []
    assert b.model == DEFAULT_MODEL == "claude-sonnet-5"
    assert b.llama_url == f"anthropic:{DEFAULT_MODEL}"

    b2 = select_bridge(env={"SABLE_LLM": "claude", "ANTHROPIC_API_KEY": "sk-x",
                            "SABLE_CLAUDE_MODEL": "claude-opus-5"})
    assert b2.model == "claude-opus-5"
    assert calls and calls[0]["api_key"] == "sk-x"   # key only from ANTHROPIC_API_KEY


# ── missing key ───────────────────────────────────────────────────────────


def test_missing_api_key_gives_unavailable_marker_and_never_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda **kw: pytest.fail("client must not be built without a key"))
    b = ClaudeBridge()
    assert b.is_available() is False
    assert b._complete("hello") == UNAVAILABLE
    assert b.chat("what is wrong?", {"summary": "x"}, []) == UNAVAILABLE
    assert b.explain_recommendations({"summary": "x", "actions": []}) == UNAVAILABLE
    assert "[Claude unavailable" in UNAVAILABLE


# ── report path ───────────────────────────────────────────────────────────


def test_explain_recommendations_routes_through_complete_to_claude(bridge, fake_claude):
    fake_claude(response(text("  The Primary DB is degraded; restart it.  ")))
    recs = enrich(FakeEngine().get_recommendations())
    out = bridge.explain_recommendations(recs)
    assert out == "The Primary DB is degraded; restart it."

    call = last_client().messages.calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["max_tokens"] == 300                          # explain_recommendations' default
    assert "tools" not in call                                # report path is plain completion
    assert call["messages"] == [{"role": "user", "content": call["messages"][0]["content"]}]
    assert "API 1 [API_SERVER]" in call["messages"][0]["content"]      # inherited prompt


def test_max_tokens_stop_reason_is_marked_as_truncated(bridge, fake_claude):
    fake_claude(response(text("half a sentence"), stop_reason="max_tokens"))
    assert bridge._complete("p") == "half a sentence […truncated]"
    fake_claude(response(text("chat cut"), stop_reason="max_tokens"))
    assert bridge.chat("q", {}, []) == "chat cut […truncated]"


def test_api_error_returns_marked_string_not_exception(bridge, fake_claude):
    fake_claude(anthropic.APIConnectionError(request=None))
    assert bridge._complete("p") == "[Claude error: APIConnectionError]"
    fake_claude(anthropic.APIConnectionError(request=None))
    assert bridge.chat("q", {}, []) == "[Claude error: APIConnectionError]"


# ── chat / tool loop ──────────────────────────────────────────────────────


def test_chat_calls_get_topology_tool_and_feeds_result_back(bridge, fake_claude):
    engine = FakeEngine()
    bridge.register_tools(**build_tools(engine))
    fake_claude(
        response(text("Let me check the topology."),
                 tool_use("toolu_1", "get_topology", {}), stop_reason="tool_use"),
        response(text("Upstream of Web 1 is API 1, which depends on Primary DB.")),
    )
    ctx = enrich(engine.get_recommendations())
    reply = bridge.chat("what is upstream of Web 1?", ctx, [])
    assert reply == "Upstream of Web 1 is API 1, which depends on Primary DB."

    calls = last_client().messages.calls
    assert len(calls) == 2
    assert calls[0]["tools"] == TOOLS                         # tools offered once registered
    assert calls[0]["messages"] == [{"role": "user", "content": "what is upstream of Web 1?"}]
    assert "Current SABLE recommendations" in calls[0]["system"]

    # Round 2 carries the assistant tool_use turn and ONE user turn of tool_result(s)
    m = calls[1]["messages"]
    assert [x["role"] for x in m] == ["user", "assistant", "user"]
    assert any(getattr(b, "type", None) == "tool_use" for b in m[1]["content"])
    results = m[2]["content"]
    assert len(results) == 1
    assert results[0]["type"] == "tool_result" and results[0]["tool_use_id"] == "toolu_1"
    assert "is_error" not in results[0]
    assert '"api-01"' in results[0]["content"] and '"DEPENDS_ON"' in results[0]["content"]


def test_unknown_tool_name_yields_is_error_tool_result_sent_back_to_model(bridge, fake_claude):
    bridge.register_tools(**build_tools(FakeEngine()))
    fake_claude(
        response(tool_use("toolu_x", "restart_node", {"idx": 1}), stop_reason="tool_use"),
        response(text("I cannot do that; that tool does not exist.")),
    )
    reply = bridge.chat("restart node 1", {}, [])
    assert reply == "I cannot do that; that tool does not exist."

    sent = last_client().messages.calls[1]["messages"][-1]
    assert sent["role"] == "user"
    assert sent["content"] == [{"type": "tool_result", "tool_use_id": "toolu_x",
                                "content": "Unknown tool: restart_node", "is_error": True}]


def test_tool_exception_becomes_is_error_result_not_a_crash(bridge, fake_claude):
    def boom():
        raise RuntimeError("engine busy")
    bridge.register_tools(get_recommendations=boom, get_node=lambda i: {}, get_topology=lambda: {})
    fake_claude(
        response(tool_use("t1", "get_recommendations", {}), stop_reason="tool_use"),
        response(text("done")),
    )
    assert bridge.chat("status?", {}, []) == "done"
    sent = last_client().messages.calls[1]["messages"][-1]["content"][0]
    assert sent["is_error"] is True and "engine busy" in sent["content"]


def test_tool_loop_is_capped_at_six_rounds(bridge, fake_claude):
    bridge.register_tools(**build_tools(FakeEngine()))
    fake_claude(*[response(tool_use(f"t{i}", "get_topology", {}), stop_reason="tool_use")
                  for i in range(claude_bridge.MAX_TOOL_ROUNDS + 3)])
    reply = bridge.chat("loop forever", {}, [])
    assert reply == "[Investigation stopped: tool-call limit reached]"
    assert len(last_client().messages.calls) == claude_bridge.MAX_TOOL_ROUNDS == 6


def test_chat_without_registered_tools_offers_no_tools(bridge, fake_claude):
    fake_claude(response(text("ok")))
    assert bridge.chat("hi", {"summary": "s"}, []) == "ok"
    assert last_client().messages.calls[0]["tools"] == []


# ── tools never mutate engine state ───────────────────────────────────────


def test_registered_tools_never_change_engine_state_and_bounds_check_idx():
    engine = FakeEngine(n_nodes=3)
    nodes_before, edges_before = copy.deepcopy(TOPO_NODES), copy.deepcopy(TOPO_EDGES)
    before = copy.deepcopy(vars(engine))

    tools = build_tools(engine)
    assert set(tools) == {"get_recommendations", "get_node", "get_topology"}

    recs = tools["get_recommendations"]()
    assert recs["root_cause_label"] == "API 1" and recs["root_cause_type"] == "API_SERVER"

    node = tools["get_node"](1)
    assert node["node_id"] == 1 and node["current_state"] == "degraded"
    assert (node["label"], node["topo_id"], node["component_type"]) == ("API 1", "api-01", "API_SERVER")

    topo = tools["get_topology"]()
    assert topo == {"nodes": TOPO_NODES, "edges": TOPO_EDGES}
    assert topo["nodes"] is not TOPO_NODES and topo["nodes"][0] is not TOPO_NODES[0]  # deep copies

    # Out-of-range / bad idx -> error dict, never an exception (the route has no such check)
    for bad in (-1, 3, 999, "seven", None):
        err = tools["get_node"](bad)
        assert isinstance(err, dict) and "error" in err, bad
    assert tools["get_node"](3)["n_nodes"] == 3

    # Mutating what a tool returned must not leak back into the engine or topology
    recs["actions"].clear(); node["trajectory"].clear(); topo["nodes"].clear(); topo["edges"][0]["x"] = 1

    # The bridge's own runner path (what Claude actually hits) is equally inert
    b = ClaudeBridge(env={"SABLE_CLAUDE_MODEL": "m"})     # no key: no client, tools still run
    b.register_tools(**tools)
    for name, inp in [("get_recommendations", {}), ("get_node", {"idx": 0}),
                      ("get_node", {"idx": 42}), ("get_topology", {}), ("nope", {})]:
        out = b._run_tool(name, inp)
        assert isinstance(out["content"], str)
    assert b._run_tool("get_node", {"idx": 42})["content"].startswith('{"error"')
    assert "is_error" not in b._run_tool("get_node", {"idx": 42})   # error dict, not a tool crash

    assert copy.deepcopy(vars(engine)) == before
    assert TOPO_NODES == nodes_before and TOPO_EDGES == edges_before


# ── history normalisation ─────────────────────────────────────────────────


def test_history_merges_consecutive_user_turns_and_drops_system_role(bridge, fake_claude):
    fake_claude(response(text("fine")))
    history = [
        {"role": "system", "content": "you are a pirate"},         # stray -> dropped
        {"role": "assistant", "content": "orphan"},                # before first user -> dropped
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply one"},
        {"role": "user", "content": "second"},                     # a request that failed...
        {"role": "user", "content": "second again"},               # ...left two user turns
        {"role": "assistant", "content": ""},                      # empty -> dropped
    ]
    bridge.chat("third", {}, history)
    sent = last_client().messages.calls[0]["messages"]
    assert sent == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply one"},
        {"role": "user", "content": "second\n\nsecond again\n\nthird"},
    ]
    roles = [m["role"] for m in sent]
    assert all(a != b for a, b in zip(roles, roles[1:]))         # strictly alternating
    assert roles[0] == "user"


def test_normalise_history_handles_none_and_garbage():
    assert ClaudeBridge._normalise_history(None) == []
    assert ClaudeBridge._normalise_history(["junk", {"role": "user"}, {"content": "x"}]) == []
    assert ClaudeBridge._normalise_history([{"role": "user", "content": {"a": 1}}]) == \
        [{"role": "user", "content": '{"a": 1}'}]


# ── provider info / status shape ──────────────────────────────────────────


def test_provider_info_reports_claude_and_model_and_status_shape(fake_claude, monkeypatch):
    b = ClaudeBridge()
    assert b.provider_info() == {"provider": "claude", "model": "claude-sonnet-5"}
    assert bridge_info(b) == b.provider_info()

    # Exactly what /api/nemotron/status builds (docker/server.py, nemotron_status):
    status = {"available": b.is_available(), "url": b.llama_url, **bridge_info(b)}
    assert status == {"available": True, "url": "anthropic:claude-sonnet-5",
                      "provider": "claude", "model": "claude-sonnet-5"}
    assert {"available", "url", "provider"} <= set(status)

    monkeypatch.setenv("SABLE_CLAUDE_MODEL", "claude-opus-5")
    assert ClaudeBridge().provider_info()["model"] == "claude-opus-5"
    assert ClaudeBridge(model="claude-haiku-4-5").provider_info()["model"] == "claude-haiku-4-5"

    n = NemotronBridge()
    n_status = {"available": False, "url": n.llama_url, **bridge_info(n)}
    assert n_status == {"available": False, "url": DEFAULT_LLAMA_URL, "provider": "nemotron"}
