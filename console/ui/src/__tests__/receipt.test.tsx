import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, within, waitFor } from "@testing-library/react";
import { ReceiptView, ReceiptRoute } from "../components/ReceiptView";
import { ReceiptsList } from "../components/ReceiptsList";
import type { Receipt } from "../types";
import receiptFx from "./fixtures/receipt.json";

const receipt = receiptFx as Receipt;

describe("ReceiptsList", () => {
  it("renders pass/fail semantics, rollback flag, counts and 12-char hash", () => {
    const failed: Receipt = {
      ...receipt,
      id: "rcp-fail",
      created_at: "2026-09-22T15:00:00Z",
      steps: [{ step_id: "s1", status: "failed", error: "exit 1" }],
      verification: { status: "fail", observed: {}, checked_at: "2026-09-22T15:00:00Z" },
      rollback: { performed: true, steps: [{ step_id: "s1", action_id: "restart_service", status: "ok" }] },
      receipt_hash: "abcdef0123456789abcdef0123456789",
    };
    const onOpen = vi.fn();
    render(<ReceiptsList receipts={[failed, receipt]} onOpen={onOpen} />);
    const ok = screen.getByTestId(`receipt-${receipt.id}`);
    expect(within(ok).getByTestId("receipt-status")).toHaveTextContent("PASS");
    expect(within(ok).getByTestId("receipt-status")).toHaveClass("sem-pass");
    expect(within(ok).getByText("75fadad48426")).toBeInTheDocument();
    expect(within(ok).getByText(/ok 1/)).toBeInTheDocument();
    expect(within(ok).queryByText("rolled back")).toBeNull();

    const bad = screen.getByTestId("receipt-rcp-fail");
    expect(within(bad).getByTestId("receipt-status")).toHaveClass("sem-fail");
    expect(within(bad).getByText("rolled back")).toBeInTheDocument();
    expect(within(bad).getByText("abcdef012345")).toBeInTheDocument();
    expect(within(bad).getByText("failed 1")).toHaveClass("sem-fail");
  });
});

describe("ReceiptView", () => {
  it("renders the full receipt: steps, digest, snapshots, approvals, hash chain, verified badge", () => {
    render(<ReceiptView receipt={receipt} chain={{ ok: true, broken_at: null }} onBack={() => {}} />);
    expect(screen.getByTestId("verification-status")).toHaveTextContent("PASS");
    expect(screen.getByTestId("verification-status")).toHaveClass("sem-pass");
    expect(screen.getByTestId("chain-badge")).toHaveTextContent("Chain verified");
    expect(screen.getByTestId("chain-badge")).toHaveClass("tag-green");
    expect(screen.getByTestId("receipt-hash")).toHaveTextContent(receipt.receipt_hash!);
    expect(screen.getByText("∅ (genesis)")).toBeInTheDocument();
    const steps = screen.getByTestId("receipt-steps");
    expect(within(steps).getByText("e3b0c44298fc1c14")).toBeInTheDocument();
    expect(within(steps).getByText("ok")).toHaveClass("sem-pass");
    expect(screen.getByText("operator@lab")).toBeInTheDocument();
    expect(screen.getAllByText("20260922-140000-abc123").length).toBeGreaterThan(0);
    expect(screen.getByTestId("rollback-flag")).toHaveTextContent("not performed");
    expect(screen.getByTestId("export-json")).toHaveAttribute("href", `/api/receipts/${receipt.id}/export?format=json`);
    expect(screen.getByTestId("export-md")).toHaveAttribute("href", `/api/receipts/${receipt.id}/export?format=md`);
  });

  it("renders fail semantics and a broken chain", () => {
    const failed: Receipt = {
      ...receipt,
      verification: { status: "fail", observed: { "dns-1": "failed" }, checked_at: receipt.created_at },
      rollback: { performed: true, steps: [{ step_id: "s1", action_id: "restart_service", status: "failed", error: "nope" }] },
    };
    render(<ReceiptView receipt={failed} chain={{ ok: false, broken_at: "rcp-000" }} onBack={() => {}} />);
    expect(screen.getByTestId("verification-status")).toHaveClass("sem-fail");
    expect(screen.getByTestId("rollback-flag")).toHaveTextContent("PERFORMED");
    expect(screen.getByTestId("chain-badge")).toHaveClass("tag-red");
    expect(screen.getByText("nope")).toBeInTheDocument();
  });
});

describe("ReceiptRoute", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("fetches /api/receipts/{id} and /api/receipts/verify", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url === "/api/receipts/verify") {
        return new Response(JSON.stringify({ ok: true, broken_at: null }), { status: 200 });
      }
      if (url === `/api/receipts/${receipt.id}`) {
        return new Response(JSON.stringify(receipt), { status: 200 });
      }
      return new Response(JSON.stringify({ error: "not found" }), { status: 404 });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<ReceiptRoute id={receipt.id} cached={null} onBack={() => {}} />);
    await waitFor(() => expect(screen.getByTestId("receipt-view")).toBeInTheDocument());
    await waitFor(() => expect(screen.getByTestId("chain-badge")).toHaveTextContent("Chain verified"));
    const urls = fetchMock.mock.calls.map((c) => c[0]);
    expect(urls).toContain(`/api/receipts/${receipt.id}`);
    expect(urls).toContain("/api/receipts/verify");
  });
});
