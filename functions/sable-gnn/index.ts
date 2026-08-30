/**
 * Project PARALLAX — SableGNN MCP Edge Function
 * ================================================
 * Supabase Edge Function that exposes the local GNN server as MCP tools
 * for Claude.ai via the connector. Proxies requests to the GNN HTTP
 * server through a cloudflared tunnel.
 *
 * Deploy:
 *   supabase functions deploy sable-gnn --no-verify-jwt
 *
 * Environment:
 *   GNN_SERVER_URL — cloudflared tunnel URL (e.g., https://xxx.trycloudflare.com)
 *   GNN_ACCESS_KEY — access key for auth (separate from CORTEX MCP_ACCESS_KEY)
 */

const GNN_SERVER_URL = Deno.env.get("GNN_SERVER_URL") || "http://localhost:5070";
const MCP_ACCESS_KEY = Deno.env.get("GNN_ACCESS_KEY") || "parallax-gnn-2026";

// ── Tool Definitions ──────────────────────────────────────────────────────

const TOOLS = [
  {
    name: "gnn_suggest_links",
    description:
      "Use the trained SableGNN to suggest structurally-reasoned links for a thought. " +
      "Unlike cosine similarity, this uses graph neural network reasoning about the " +
      "knowledge graph topology to find connections. Returns predicted relation types " +
      "and confidence scores.",
    inputSchema: {
      type: "object",
      properties: {
        thought_id: { type: "string", description: "UUID of the thought to find links for" },
        top_k: { type: "number", description: "Number of suggestions (default: 15)" },
        min_confidence: { type: "number", description: "Min link probability (default: 0.5)" },
      },
      required: ["thought_id"],
    },
  },
  {
    name: "gnn_score_pair",
    description:
      "Score a specific pair of thoughts using the GNN. Returns link probability, " +
      "predicted relation type, contradiction probability, and full type distribution.",
    inputSchema: {
      type: "object",
      properties: {
        source_id: { type: "string", description: "UUID of source thought" },
        target_id: { type: "string", description: "UUID of target thought" },
      },
      required: ["source_id", "target_id"],
    },
  },
  {
    name: "gnn_find_contradictions",
    description:
      "Find thought pairs the GNN identifies as potentially contradictory. " +
      "Uses contradiction detection head (AUC 0.96) to rank edges.",
    inputSchema: {
      type: "object",
      properties: {
        top_k: { type: "number", description: "Number of candidates (default: 15)" },
        threshold: { type: "number", description: "Min contradiction probability (default: 0.3)" },
      },
    },
  },
  {
    name: "gnn_stats",
    description: "Get stats about the loaded GNN model and graph.",
    inputSchema: { type: "object", properties: {} },
  },
];

// ── GNN Server Proxy ──────────────────────────────────────────────────────

async function callGNN(
  endpoint: string,
  payload?: Record<string, unknown>,
  method: string = "POST"
): Promise<unknown> {
  const url = `${GNN_SERVER_URL}${endpoint}`;
  try {
    const opts: RequestInit = {
      method,
      headers: { "Content-Type": "application/json" },
    };
    if (method === "POST" && payload) {
      opts.body = JSON.stringify(payload);
    }
    const resp = await fetch(url, opts);
    if (!resp.ok) {
      return { error: `GNN server returned ${resp.status}: ${await resp.text()}` };
    }
    return await resp.json();
  } catch (e) {
    return { error: `GNN server unreachable at ${GNN_SERVER_URL}: ${e}` };
  }
}

async function executeTool(
  name: string,
  args: Record<string, unknown>
): Promise<string> {
  let result: unknown;

  switch (name) {
    case "gnn_suggest_links":
      result = await callGNN("/suggest", {
        thought_id: args.thought_id,
        top_k: args.top_k ?? 15,
        min_confidence: args.min_confidence ?? 0.5,
      });
      break;

    case "gnn_score_pair":
      result = await callGNN("/score", {
        source_id: args.source_id,
        target_id: args.target_id,
      });
      break;

    case "gnn_find_contradictions":
      result = await callGNN("/contradictions", {
        top_k: args.top_k ?? 15,
        threshold: args.threshold ?? 0.3,
      });
      break;

    case "gnn_stats":
      result = await callGNN("/stats", undefined, "GET");
      break;

    default:
      result = { error: `Unknown tool: ${name}` };
  }

  return JSON.stringify(result, null, 2);
}

// ── MCP Request Handler ──────────────────────────────────────────────────

Deno.serve(async (req: Request) => {
  // Auth check
  const url = new URL(req.url);
  const key = url.searchParams.get("key") || req.headers.get("x-gnn-key");
  if (key !== MCP_ACCESS_KEY) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response(null, {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, x-gnn-key",
      },
    });
  }

  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "POST only" }), {
      status: 405,
      headers: { "Content-Type": "application/json" },
    });
  }

  const body = await req.json();
  const { method, params, id } = body;

  let result: unknown;

  switch (method) {
    case "initialize":
      result = {
        protocolVersion: "2024-11-05",
        capabilities: { tools: {} },
        serverInfo: { name: "sable-gnn", version: "1.0.0" },
      };
      break;

    case "tools/list":
      result = { tools: TOOLS };
      break;

    case "tools/call": {
      const toolName = params?.name;
      const args = params?.arguments || {};
      const text = await executeTool(toolName, args);
      result = {
        content: [{ type: "text", text }],
      };
      break;
    }

    case "ping":
      result = {};
      break;

    default:
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          id,
          error: { code: -32601, message: `Method not found: ${method}` },
        }),
        { headers: { "Content-Type": "application/json" } }
      );
  }

  // Return as SSE (same format as cortex-mcp)
  const response = JSON.stringify({ jsonrpc: "2.0", id, result });
  const sseData = `data: ${response}\n\n`;

  return new Response(sseData, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Access-Control-Allow-Origin": "*",
    },
  });
});
