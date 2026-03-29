#!/usr/bin/env python3
"""
Project PARALLAX — GNN MCP Server (stdio transport)
=====================================================
Wraps the GNN HTTP server as an MCP tool server for Claude Code.

Install:
    claude mcp add sable-gnn -- python3 /mnt/vault/sable/pillar1/gnn_mcp_server.py

Requires: GNN server running on localhost:5070
"""

import json
import sys
import httpx

GNN_SERVER_URL = "http://localhost:5070"


def read_message() -> dict | None:
    """Read a JSON-RPC message from stdin (content-length framed)."""
    headers = {}
    while True:
        line = sys.stdin.readline()
        if not line:
            return None
        line = line.strip()
        if line == "":
            break
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()

    length = int(headers.get("content-length", 0))
    if length == 0:
        return None

    body = sys.stdin.read(length)
    return json.loads(body)


def send_message(msg: dict):
    """Write a JSON-RPC message to stdout (content-length framed)."""
    body = json.dumps(msg)
    sys.stdout.write(f"Content-Length: {len(body)}\r\n\r\n{body}")
    sys.stdout.flush()


def send_result(request_id, result: dict):
    send_message({"jsonrpc": "2.0", "id": request_id, "result": result})


def send_error(request_id, code: int, message: str):
    send_message({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


# ── Tool Definitions ──────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "gnn_suggest_links",
        "description": (
            "Use the trained SableGNN to suggest structurally-reasoned links for a thought. "
            "Unlike cosine similarity, this uses graph neural network reasoning about the "
            "knowledge graph topology to find connections. Returns predicted relation types "
            "and confidence scores."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "thought_id": {
                    "type": "string",
                    "description": "UUID of the thought to find links for",
                },
                "top_k": {
                    "type": "number",
                    "description": "Number of suggestions to return (default: 15)",
                    "default": 15,
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum link probability threshold (default: 0.5)",
                    "default": 0.5,
                },
            },
            "required": ["thought_id"],
        },
    },
    {
        "name": "gnn_score_pair",
        "description": (
            "Score a specific pair of thoughts using the GNN. Returns link probability, "
            "predicted relation type, type confidence, contradiction probability, and "
            "full type distribution across all 7 relation types."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "UUID of the source thought",
                },
                "target_id": {
                    "type": "string",
                    "description": "UUID of the target thought",
                },
            },
            "required": ["source_id", "target_id"],
        },
    },
    {
        "name": "gnn_find_contradictions",
        "description": (
            "Find thought pairs that the GNN identifies as potentially contradictory. "
            "Uses the contradiction detection head (AUC 0.96) to rank edges by "
            "contradiction probability."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "top_k": {
                    "type": "number",
                    "description": "Number of candidates to return (default: 15)",
                    "default": 15,
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum contradiction probability (default: 0.3)",
                    "default": 0.3,
                },
            },
        },
    },
    {
        "name": "gnn_stats",
        "description": "Get stats about the loaded GNN model and graph.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Tool Execution ────────────────────────────────────────────────────────


def call_gnn(endpoint: str, payload: dict | None = None, method: str = "POST") -> dict:
    """Call the GNN HTTP server."""
    try:
        with httpx.Client(timeout=30) as client:
            if method == "GET":
                resp = client.get(f"{GNN_SERVER_URL}{endpoint}")
            else:
                resp = client.post(
                    f"{GNN_SERVER_URL}{endpoint}",
                    json=payload or {},
                    headers={"Content-Type": "application/json"},
                )
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        return {"error": "GNN server not running. Start it: python gnn_server.py --port 5070"}
    except Exception as e:
        return {"error": str(e)}


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return the result as text."""
    if name == "gnn_suggest_links":
        result = call_gnn("/suggest", {
            "thought_id": arguments["thought_id"],
            "top_k": arguments.get("top_k", 15),
            "min_confidence": arguments.get("min_confidence", 0.5),
        })

    elif name == "gnn_score_pair":
        result = call_gnn("/score", {
            "source_id": arguments["source_id"],
            "target_id": arguments["target_id"],
        })

    elif name == "gnn_find_contradictions":
        result = call_gnn("/contradictions", {
            "top_k": arguments.get("top_k", 15),
            "threshold": arguments.get("threshold", 0.3),
        })

    elif name == "gnn_stats":
        result = call_gnn("/stats", method="GET")

    else:
        result = {"error": f"Unknown tool: {name}"}

    return json.dumps(result, indent=2)


# ── MCP Protocol Loop ────────────────────────────────────────────────────


def main():
    while True:
        msg = read_message()
        if msg is None:
            break

        request_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params", {})

        if method == "initialize":
            send_result(request_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "sable-gnn",
                    "version": "1.0.0",
                },
            })

        elif method == "notifications/initialized":
            pass  # No response needed for notifications

        elif method == "tools/list":
            send_result(request_id, {"tools": TOOLS})

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            try:
                result_text = execute_tool(tool_name, arguments)
                send_result(request_id, {
                    "content": [{"type": "text", "text": result_text}],
                })
            except Exception as e:
                send_result(request_id, {
                    "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                    "isError": True,
                })

        elif method == "ping":
            send_result(request_id, {})

        else:
            send_error(request_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    main()
