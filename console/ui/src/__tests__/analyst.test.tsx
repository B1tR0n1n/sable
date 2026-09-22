import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import { AnalystPanel, splitReply, chipLabel } from "../components/AnalystPanel";
import type { Finding } from "../types";
import findingFx from "./fixtures/finding.json";

const finding = findingFx as Finding;

describe("splitReply", () => {
  it("separates fenced SABLE data from generated prose", () => {
    const parts = splitReply("dns-1 is failed.\n```sable\n{\"node\": \"dns-1\", \"state\": \"failed\"}\n```\nSo restart it.");
    expect(parts).toEqual([
      { kind: "prose", text: "dns-1 is failed." },
      { kind: "sable", text: '{"node": "dns-1", "state": "failed"}' },
      { kind: "prose", text: "So restart it." },
    ]);
  });
  it("formats tool chips", () => {
    expect(chipLabel({ name: "get_topology", input: {} })).toBe("get_topology");
    expect(chipLabel({ name: "get_node", input: { idx: 3 } })).toBe("get_node idx=3");
  });
});

describe("AnalystPanel", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("posts to /api/analyst/chat and separates SABLE blocks from generated text", async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body));
      expect(body.finding_id).toBe(finding.id);
      expect(body.message).toBe("why is app-erp degraded?");
      expect(body.history).toEqual([]);
      return new Response(
        JSON.stringify({
          reply: "app-erp depends on dns-1.\n```sable\ndns-1: state=failed up=0\n```\nRestarting dnsmasq should clear it.",
          tool_calls: [
            { name: "get_topology", input: {} },
            { name: "get_node", input: { idx: 3 } },
          ],
          provider: "claude",
          generated: true,
        }),
        { status: 200 },
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<AnalystPanel open finding={finding} onClose={() => {}} />);
    expect(screen.getByTestId("provider-indicator")).toHaveTextContent("provider —");
    fireEvent.change(screen.getByTestId("analyst-input"), { target: { value: "why is app-erp degraded?" } });
    fireEvent.click(screen.getByTestId("analyst-send"));

    await waitFor(() => expect(screen.getByTestId("sable-block")).toBeInTheDocument());
    expect(fetchMock.mock.calls[0][0]).toBe("/api/analyst/chat");

    const sable = screen.getByTestId("sable-block");
    expect(sable).toHaveTextContent("dns-1: state=failed up=0");
    expect(sable).toHaveAttribute("aria-label", "SABLE data");

    const generated = screen.getAllByTestId("generated-text");
    expect(generated).toHaveLength(2);
    expect(generated[0]).toHaveTextContent("app-erp depends on dns-1.");
    expect(generated[1]).toHaveTextContent("Restarting dnsmasq should clear it.");
    for (const g of generated) expect(g).toHaveAttribute("aria-label", "ANALYST · GENERATED");
    expect(sable).not.toHaveClass("generated");

    const chips = within(screen.getByTestId("tool-chips")).getAllByText(/get_/);
    expect(chips.map((c) => c.textContent)).toEqual(["get_topology", "get_node idx=3"]);

    expect(screen.getByTestId("provider-indicator")).toHaveTextContent("claude API");
    expect(screen.getByText(/Analyst · generated · claude/)).toBeInTheDocument();
  });

  it("shows the local provider", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ reply: "ok", tool_calls: [], provider: "local", generated: true }), { status: 200 })),
    );
    render(<AnalystPanel open finding={finding} onClose={() => {}} />);
    fireEvent.change(screen.getByTestId("analyst-input"), { target: { value: "hi" } });
    fireEvent.click(screen.getByTestId("analyst-send"));
    await waitFor(() => expect(screen.getByTestId("provider-indicator")).toHaveTextContent("local"));
  });
});
