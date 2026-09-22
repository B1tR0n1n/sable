import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import App from "../App";
import { store } from "../store";
import type { ConsoleState, Finding, Plan, Receipt } from "../types";
import findingFx from "./fixtures/finding.json";
import planFx from "./fixtures/plan.json";
import receiptFx from "./fixtures/receipt.json";

const finding = findingFx as Finding;
const plan = planFx as Plan;
const receipt = receiptFx as Receipt;

const state: ConsoleState = {
  sable: { url: "http://127.0.0.1:8080", ok: true, provider: "nemotron", device: "cuda:0" },
  overlord: { socket: "/run/overlord.sock", ok: true, version: "1.2.0" },
  counts: { findings_open: 1, plans_pending: 1, receipts: 1 },
  policy: {
    bands: { high: 0.85, medium: 0.6 },
    matrix: {
      high: { reversible: "human", compensable: "human", irreversible: "human" },
      medium: { reversible: "human", compensable: "human", irreversible: "human" },
      low: { reversible: "human", compensable: "human", irreversible: "human" },
    },
    default_deny: true,
  },
  lab: true,
};

class FakeSocket {
  static last: FakeSocket | null = null;
  onopen: (() => void) | null = null;
  onmessage: ((m: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(_url: string) {
    FakeSocket.last = this;
  }
  close() {}
  emit(ev: unknown) {
    this.onmessage?.({ data: JSON.stringify(ev) });
  }
}

function routes(url: string): Response {
  const json = (b: unknown, status = 200) => new Response(JSON.stringify(b), { status });
  if (url === "/api/state") return json(state);
  if (url === "/api/findings?status=open") return json([finding]);
  if (url === "/api/findings?status=closed") return json([]);
  if (url === "/api/receipts") return json([receipt]);
  if (url.startsWith("/api/log")) return json([{ ts: "2026-09-22T14:00:00Z", level: "info", source: "bridge", text: "finding ingested" }]);
  if (url === "/api/policy") return json({ ...state.policy, delay_s: 120, unmonitored_gap_cap: 1 });
  if (url === `/api/findings/${finding.id}/plan`) return json(plan);
  return json({ error: "not found" }, 404);
}

describe("App", () => {
  beforeEach(() => {
    store.reset();
    vi.stubGlobal("WebSocket", FakeSocket);
    vi.stubGlobal("fetch", vi.fn(async (url: string) => routes(url)));
    window.history.pushState(null, "", "/");
  });
  afterEach(() => vi.unstubAllGlobals());

  it("boots from /api/state, lights stations from the stream and shows default_deny + lab", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByTestId("default-deny")).toBeInTheDocument());
    expect(screen.getByText(/nemotron/)).toBeInTheDocument();
    expect(screen.getByText(/1\.2\.0/)).toBeInTheDocument();
    expect(screen.getByTestId("lab-controls")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId(`finding-${finding.id}`)).toBeInTheDocument());
    await waitFor(() => expect(screen.getByTestId("gate-rule")).toHaveTextContent("default-deny"));
    expect(screen.getByTestId("cell-high-compensable")).toHaveClass("hit");
    expect(screen.getByText("finding ingested")).toBeInTheDocument();

    act(() => {
      FakeSocket.last?.onopen?.();
      FakeSocket.last?.emit({ type: "step", ts: "t", plan_id: plan.id, step: { step_id: "s1", status: "running" } });
    });
    expect(screen.getByTestId("station-EXECUTE")).toHaveClass("lit");
    act(() => {
      FakeSocket.last?.emit({ type: "log", ts: "t", line: { ts: "2026-09-22T14:00:01Z", level: "info", source: "executor", text: "s1 restart_service dns-1" } });
    });
    expect(screen.getByText("s1 restart_service dns-1")).toBeInTheDocument();
    expect(screen.getByTestId(`receipt-${receipt.id}`)).toBeInTheDocument();
  });

  it("routes /receipts/:id to the receipt view", async () => {
    window.history.pushState(null, "", `/receipts/${receipt.id}`);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url === `/api/receipts/${receipt.id}`) return new Response(JSON.stringify(receipt), { status: 200 });
        if (url === "/api/receipts/verify") return new Response(JSON.stringify({ ok: true }), { status: 200 });
        return routes(url);
      }),
    );
    render(<App />);
    await waitFor(() => expect(screen.getByTestId("receipt-view")).toBeInTheDocument());
    expect(screen.getByTestId("receipt-hash")).toHaveTextContent(receipt.receipt_hash!);
  });
});
