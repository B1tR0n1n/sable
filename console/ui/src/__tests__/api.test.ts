import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { api, ApiError, captureTokenFromUrl, getToken, TOKEN_KEY, WsClient } from "../api";

describe("api", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("hits the documented paths with the documented bodies", async () => {
    const calls: Array<[string, RequestInit | undefined]> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        calls.push([url, init]);
        return new Response("{}", { status: 200 });
      }),
    );
    await api.state();
    await api.findings("open");
    await api.findingPlan("f1");
    await api.replan("f1", "llm");
    await api.approve("p1", { actor: "op", decision: "approve" });
    await api.receipts("f1");
    await api.verifyChain();
    await api.policy();
    await api.putPolicy({ bands: { high: 0.85, medium: 0.6 }, matrix: {} as never });
    await api.log(50);
    await api.analystChat("f1", "hi", []);
    await api.labFault("stop_service");

    expect(calls.map(([u, i]) => `${i?.method ?? "GET"} ${u}`)).toEqual([
      "GET /api/state",
      "GET /api/findings?status=open",
      "GET /api/findings/f1/plan",
      "POST /api/findings/f1/plan",
      "POST /api/plans/p1/approve",
      "GET /api/receipts?finding_id=f1",
      "GET /api/receipts/verify",
      "GET /api/policy",
      "PUT /api/policy",
      "GET /api/log?limit=50",
      "POST /api/analyst/chat",
      "POST /api/lab/fault",
    ]);
    expect(JSON.parse(String(calls[3][1]?.body))).toEqual({ planner: "llm" });
    expect(JSON.parse(String(calls[4][1]?.body))).toEqual({ actor: "op", decision: "approve" });
    expect(JSON.parse(String(calls[10][1]?.body))).toEqual({ finding_id: "f1", message: "hi", history: [] });
    expect(JSON.parse(String(calls[11][1]?.body))).toEqual({ name: "stop_service" });
    expect(api.receiptExportUrl("r1", "md")).toBe("/api/receipts/r1/export?format=md");
  });

  it("surfaces a 422 verbatim as ApiError with the body", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ error: "steps.0.action_id: not in catalog" }), { status: 422 })),
    );
    const err = await api.replan("f1", "llm").catch((e) => e as ApiError);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(422);
    expect((err as ApiError).message).toBe("steps.0.action_id: not in catalog");
    expect((err as ApiError).body).toEqual({ error: "steps.0.action_id: not in catalog" });
  });
});

describe("console token", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    localStorage.removeItem(TOKEN_KEY);
    window.history.replaceState(null, "", "/");
  });
  afterEach(() => localStorage.removeItem(TOKEN_KEY));

  function capture() {
    const calls: Array<[string, RequestInit | undefined]> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        calls.push([url, init]);
        return new Response("{}", { status: 200 });
      }),
    );
    return calls;
  }
  const auth = (init?: RequestInit) => (init?.headers as Record<string, string> | undefined)?.authorization;

  it("attaches Authorization: Bearer from localStorage to reads and writes", async () => {
    localStorage.setItem(TOKEN_KEY, "tok-1");
    const calls = capture();
    await api.state();
    await api.approve("p1", { actor: "op", decision: "approve" });
    await api.putPolicy({ bands: { high: 0.85, medium: 0.6 }, matrix: {} as never });
    expect(calls.map(([, i]) => auth(i))).toEqual(["Bearer tok-1", "Bearer tok-1", "Bearer tok-1"]);
    expect((calls[1][1]?.headers as Record<string, string>)["content-type"]).toBe("application/json");
  });

  it("sends no Authorization header without a token", async () => {
    const calls = capture();
    await api.approve("p1", { actor: "op", decision: "approve" });
    expect(auth(calls[0][1])).toBeUndefined();
    expect(getToken()).toBeNull();
  });

  it("captures ?token= from the URL, stores it and strips it from the address bar", () => {
    window.history.replaceState(null, "", "/?token=abc%20def&x=1#/receipts/r1");
    expect(captureTokenFromUrl()).toBe("abc def");
    expect(getToken()).toBe("abc def");
    expect(window.location.search).toBe("?x=1");
    expect(window.location.hash).toBe("#/receipts/r1");
    expect(window.location.pathname).toBe("/");
  });

  it("captures #token= too and leaves a clean URL", () => {
    window.history.replaceState(null, "", "/#token=xyz");
    expect(captureTokenFromUrl()).toBe("xyz");
    expect(getToken()).toBe("xyz");
    expect(window.location.hash).toBe("");
    expect(window.location.search).toBe("");
  });

  it("keeps the stored token when the URL carries none", () => {
    localStorage.setItem(TOKEN_KEY, "kept");
    expect(captureTokenFromUrl()).toBe("kept");
    expect(getToken()).toBe("kept");
  });
});

class FakeSocket {
  static instances: FakeSocket[] = [];
  url: string;
  onopen: (() => void) | null = null;
  onmessage: ((m: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(url: string) {
    this.url = url;
    FakeSocket.instances.push(this);
  }
  close() {
    this.onclose?.();
  }
}

describe("WsClient", () => {
  beforeEach(() => {
    FakeSocket.instances = [];
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", FakeSocket);
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("dispatches parsed events and reconnects with doubling backoff", () => {
    const onEvent = vi.fn();
    const status: string[] = [];
    const c = new WsClient({ url: "ws://x/ws", onEvent, onStatus: (s) => status.push(s), backoffMs: 100, maxBackoffMs: 1000 });
    c.connect();
    expect(FakeSocket.instances).toHaveLength(1);
    FakeSocket.instances[0].onopen?.();
    FakeSocket.instances[0].onmessage?.({ data: JSON.stringify({ type: "log", ts: "t", line: { ts: "t", level: "info", source: "x", text: "y" } }) });
    FakeSocket.instances[0].onmessage?.({ data: "not json" });
    expect(onEvent).toHaveBeenCalledTimes(1);
    expect(onEvent.mock.calls[0][0].type).toBe("log");

    FakeSocket.instances[0].onclose?.();
    expect(status.at(-1)).toBe("closed");
    vi.advanceTimersByTime(99);
    expect(FakeSocket.instances).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakeSocket.instances).toHaveLength(2);

    FakeSocket.instances[1].onclose?.(); // never opened → attempt 2 → 200ms
    vi.advanceTimersByTime(199);
    expect(FakeSocket.instances).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(FakeSocket.instances).toHaveLength(3);

    c.close();
    vi.advanceTimersByTime(5000);
    expect(FakeSocket.instances).toHaveLength(3);
  });
});
